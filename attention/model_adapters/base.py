"""Shared contracts and token aggregation for model adapters."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


class TokenAlignmentError(ValueError):
    """Raised when model subtokens cannot be mapped to dataset code tokens."""


@dataclass(frozen=True)
class ModelSpec:
    """Static, auditable semantics for one model used in the paper."""

    name: str
    checkpoint: str
    family: str
    analyzed_component: str
    input_protocol: str
    subtoken_marker: str
    prefix_tokens: int
    suffix_tokens: int
    expected_transformer_layers: int
    expected_attention_heads: int
    expected_hidden_size: int
    supports_attention: bool = True
    paper_max_subtokens: int = 500

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AdapterOutput:
    """Canonical representation consumed by AST/DFG and hidden-state tasks."""

    model_name: str
    checkpoint: str
    code_tokens: List[str]
    subtokens: List[str]
    subtoken_groups: List[List[int]]
    input_tokens: List[str]
    lexical_positions: List[int]
    attention: Optional[np.ndarray]
    hidden_repr: np.ndarray
    metadata: Dict[str, Any]


def align_subtokens(
    subtokens: Sequence[str],
    code_tokens: Sequence[str],
    marker: str,
) -> List[List[int]]:
    """Align a joined-code tokenization back to CodeSearchNet lexical tokens.

    This intentionally follows the paper repository's normalization: spaces
    inside a dataset token are removed, as are leading tokenizer word-boundary
    markers.  Unlike the legacy helper, failures are explicit and diagnostic.
    """

    normalized = [token.replace(" ", "") for token in code_tokens]
    if not normalized:
        raise TokenAlignmentError("Cannot align an empty code-token sequence")
    if any(token == "" for token in normalized):
        raise TokenAlignmentError("Dataset contains an empty code token")

    groups: List[List[int]] = []
    token_index = 0
    accumulated = ""
    current: List[int] = []

    for subtoken_index, subtoken in enumerate(subtokens):
        if token_index >= len(normalized):
            raise TokenAlignmentError(
                f"Tokenizer produced trailing subtoken {subtoken!r} at index "
                f"{subtoken_index} after all {len(normalized)} code tokens"
            )
        piece = subtoken
        while marker and piece.startswith(marker):
            piece = piece[len(marker):]
        # CodeGen's tokenizer can emit a run of literal spaces as a subtoken
        # inside a quoted CSN token.  Dataset tokens are normalized by removing
        # their internal spaces, so apply the same normalization to each model
        # piece while retaining its index in the averaging group.
        piece = piece.replace(" ", "")
        accumulated += piece
        current.append(subtoken_index)

        expected = normalized[token_index]
        if accumulated == expected:
            groups.append(current)
            current = []
            accumulated = ""
            token_index += 1
        elif not expected.startswith(accumulated):
            raise TokenAlignmentError(
                f"Subtoken alignment diverged at code token {token_index} "
                f"({code_tokens[token_index]!r}): reconstructed "
                f"{accumulated!r} after subtoken {subtoken!r}"
            )

    if current or token_index != len(normalized):
        remaining = code_tokens[token_index:token_index + 3]
        raise TokenAlignmentError(
            f"Subtoken alignment stopped at code token "
            f"{token_index}/{len(normalized)}; next tokens: {remaining!r}"
        )
    return groups


def aggregate_attention(
    attentions: Sequence[torch.Tensor],
    lexical_positions: Sequence[int],
    groups: Sequence[Sequence[int]],
) -> np.ndarray:
    """Average attention blocks exactly as the paper repository does."""

    if not attentions or any(attention is None for attention in attentions):
        raise ValueError("Model did not return attention for every layer")
    stacked = torch.stack([attention[0] for attention in attentions])
    if stacked.ndim != 4:
        raise ValueError(f"Expected [layers, heads, seq, seq], got {stacked.shape}")

    positions = torch.as_tensor(
        lexical_positions, dtype=torch.long, device=stacked.device
    )
    lexical = stacked.index_select(2, positions).index_select(3, positions)
    targets = torch.stack(
        [lexical[:, :, :, list(group)].mean(dim=3) for group in groups],
        dim=3,
    )
    merged = torch.stack(
        [targets[:, :, list(group), :].mean(dim=2) for group in groups],
        dim=2,
    )
    return merged.detach().to(dtype=torch.float32).cpu().numpy()


def aggregate_hidden_states(
    hidden_states: Sequence[torch.Tensor],
    lexical_positions: Sequence[int],
    groups: Sequence[Sequence[int]],
) -> np.ndarray:
    """Average subtoken states into one vector per dataset code token."""

    if not hidden_states or any(state is None for state in hidden_states):
        raise ValueError("Model did not return every hidden state")
    stacked = torch.stack([state[0] for state in hidden_states])
    if stacked.ndim != 3:
        raise ValueError(f"Expected [states, seq, hidden], got {stacked.shape}")

    positions = torch.as_tensor(
        lexical_positions, dtype=torch.long, device=stacked.device
    )
    lexical = stacked.index_select(1, positions)
    merged = torch.stack(
        [lexical[:, list(group), :].mean(dim=1) for group in groups],
        dim=1,
    )
    return merged.detach().to(dtype=torch.float32).cpu().numpy()


def validate_output(output: AdapterOutput, spec: ModelSpec) -> None:
    """Fail closed when a model output cannot support downstream analyses."""

    token_count = len(output.code_tokens)
    expected_attention = (
        spec.expected_transformer_layers,
        spec.expected_attention_heads,
        token_count,
        token_count,
    )
    expected_hidden = (
        spec.expected_transformer_layers + 1,
        token_count,
        spec.expected_hidden_size,
    )
    if spec.supports_attention:
        if output.attention is None:
            raise ValueError(
                f"{spec.name} declares attention support but returned None"
            )
        if output.attention.shape != expected_attention:
            raise ValueError(
                f"Attention shape {output.attention.shape}, expected "
                f"{expected_attention} for {spec.name}"
            )
    elif output.attention is not None:
        raise ValueError(
            f"{spec.name} is hidden-only but returned attention"
        )
    if output.hidden_repr.shape != expected_hidden:
        raise ValueError(
            f"Hidden-state shape {output.hidden_repr.shape}, expected "
            f"{expected_hidden} for {spec.name}"
        )
    if len(output.subtoken_groups) != token_count:
        raise ValueError("There is not exactly one subtoken group per code token")
    flattened = [index for group in output.subtoken_groups for index in group]
    if flattened != list(range(len(output.subtokens))):
        raise ValueError("Subtoken groups are not a complete ordered partition")
    if not np.isfinite(output.hidden_repr).all():
        raise ValueError("Hidden representations contain NaN or infinity")
    if output.attention is not None:
        if not np.isfinite(output.attention).all():
            raise ValueError("Attention contains NaN or infinity")
        if (
            output.attention.min() < -1e-6
            or output.attention.max() > 1 + 1e-6
        ):
            raise ValueError("Attention contains a value outside [0, 1]")
        if spec.family == "decoder-only":
            future_attention = np.triu(output.attention, k=1)
            if np.max(np.abs(future_attention), initial=0.0) > 1e-6:
                raise ValueError("Decoder-only attention is not causal")


class ModelAdapter(ABC):
    """Base class for one loaded checkpoint and its extraction semantics."""

    def __init__(
        self,
        spec: ModelSpec,
        device: str = "cuda:0",
        local_files_only: bool = False,
    ) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device {device!r}, but CUDA is unavailable"
            )
        self.spec = spec
        self.device = torch.device(device)
        self.local_files_only = local_files_only
        self.model = None
        self.tokenizer = None

    def validate_loaded_model(self) -> None:
        """Check the downloaded checkpoint against the declared paper contract."""

        config = self.model.config

        def first_attribute(names):
            for name in names:
                value = getattr(config, name, None)
                if value is not None:
                    return int(value)
            return None

        observed = {
            "layers": first_attribute(
                ("num_hidden_layers", "num_layers", "n_layer", "encoder_layers")
            ),
            "heads": first_attribute(
                ("num_attention_heads", "num_heads", "n_head", "encoder_attention_heads")
            ),
            "hidden": first_attribute(("hidden_size", "d_model", "n_embd")),
        }
        expected = {
            "layers": self.spec.expected_transformer_layers,
            "heads": self.spec.expected_attention_heads,
            "hidden": self.spec.expected_hidden_size,
        }
        mismatches = {
            key: (observed[key], expected[key])
            for key in expected
            if observed[key] is not None and observed[key] != expected[key]
        }
        if mismatches:
            raise ValueError(
                f"Checkpoint {self.spec.checkpoint} does not match its adapter "
                f"contract (observed, expected): {mismatches}"
            )

    @abstractmethod
    def load(self) -> "ModelAdapter":
        """Load tokenizer/model once, move it to the device, and select eval."""

    @abstractmethod
    def _prepare_and_forward(
        self, code_tokens: Sequence[str]
    ) -> Tuple[List[str], List[str], List[int], Any, Dict[str, Any]]:
        """Return subtokens, inputs, lexical positions, outputs, and metadata."""

    def extract(self, code_tokens: Sequence[str]) -> AdapterOutput:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Call load() before extract()")
        subtokens, input_tokens, positions, outputs, metadata = (
            self._prepare_and_forward(code_tokens)
        )
        if len(subtokens) > self.spec.paper_max_subtokens:
            raise ValueError(
                f"{len(subtokens)} subtokens exceed the paper protocol limit "
                f"of {self.spec.paper_max_subtokens}"
            )
        groups = align_subtokens(
            subtokens, code_tokens, self.spec.subtoken_marker
        )
        attention = aggregate_attention(outputs.attentions, positions, groups)
        hidden_repr = aggregate_hidden_states(
            outputs.hidden_states, positions, groups
        )
        result = AdapterOutput(
            model_name=self.spec.name,
            checkpoint=self.spec.checkpoint,
            code_tokens=list(code_tokens),
            subtokens=subtokens,
            subtoken_groups=groups,
            input_tokens=input_tokens,
            lexical_positions=positions,
            attention=attention,
            hidden_repr=hidden_repr,
            metadata={
                "adapter_contract": self.spec.to_dict(),
                "num_subtokens": len(subtokens),
                **metadata,
            },
        )
        validate_output(result, self.spec)
        return result
