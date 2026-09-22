"""Hidden-state extraction adapter for Mamba."""

from typing import Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import AdapterOutput, ModelAdapter


def normalize_piece(token, marker="Ġ"):
    return (
        token.replace(marker, "")
        .replace("Ċ", "")
        .replace(" ", "")
        .replace("\t", "")
        .replace("\n", "")
        .strip()
    )


def align_mamba_subtokens(subtokens, code_tokens, marker):
    targets = [normalize_piece(token, marker) for token in code_tokens]
    groups = []
    ignored = []
    token_index = 0
    current = []
    reconstructed = ""

    for subtoken_index, subtoken in enumerate(subtokens):
        piece = normalize_piece(subtoken, marker)

        if piece == "":
            ignored.append(subtoken_index)
            continue

        if token_index >= len(targets):
            raise ValueError(
                f"Trailing Mamba subtoken {subtoken!r} at {subtoken_index}"
            )

        current.append(subtoken_index)
        reconstructed += piece
        expected = targets[token_index]

        if reconstructed == expected:
            groups.append(current)
            current = []
            reconstructed = ""
            token_index += 1
        elif not expected.startswith(reconstructed):
            raise ValueError(
                f"Mamba token alignment failed at token {token_index}: "
                f"expected {code_tokens[token_index]!r}, "
                f"reconstructed {reconstructed!r}"
            )

    if current or token_index != len(targets):
        raise ValueError(
            f"Mamba alignment stopped at "
            f"{token_index}/{len(targets)} code tokens"
        )

    return groups, ignored


class MambaAdapter(ModelAdapter):
    """Extract token-aligned hidden states from Mamba-370M."""

    def load(self):
        kwargs = {
            "local_files_only": self.local_files_only,
            "trust_remote_code": True,
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.spec.checkpoint,
            **kwargs,
        )

        model_kwargs = {
            **kwargs,
        }
        if self.device.type == "cuda":
            model_kwargs["dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.spec.checkpoint,
            **model_kwargs,
        ).to(self.device)

        self.model.eval()
        self.validate_loaded_model()
        return self

    def _prepare_and_forward(self, code_tokens: Sequence[str]):
        subtokens = list(
            self.tokenizer.tokenize(" ".join(code_tokens))
        )

        if len(subtokens) > self.spec.paper_max_subtokens:
            raise ValueError(
                f"{len(subtokens)} subtokens exceed limit "
                f"{self.spec.paper_max_subtokens}"
            )

        token_ids = self.tokenizer.convert_tokens_to_ids(subtokens)
        input_ids = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device,
        )

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        return subtokens, subtokens, list(range(len(subtokens))), outputs, {}

    def extract(self, code_tokens: Sequence[str]):
        subtokens, input_tokens, positions, outputs, _ = (
            self._prepare_and_forward(code_tokens)
        )

        groups, ignored = align_mamba_subtokens(
            subtokens,
            code_tokens,
            self.spec.subtoken_marker,
        )

        stacked = torch.stack(
            [state[0] for state in outputs.hidden_states]
        )

        merged = torch.stack(
            [
                stacked[:, group, :].mean(dim=1)
                for group in groups
            ],
            dim=1,
        )

        hidden_repr = (
            merged.detach()
            .to(dtype=torch.float32)
            .cpu()
            .numpy()
        )

        expected_shape = (
            self.spec.expected_transformer_layers + 1,
            len(code_tokens),
            self.spec.expected_hidden_size,
        )
        if hidden_repr.shape != expected_shape:
            raise ValueError(
                f"Mamba hidden shape {hidden_repr.shape}, "
                f"expected {expected_shape}"
            )
        if not np.isfinite(hidden_repr).all():
            raise ValueError("Mamba hidden states contain NaN or infinity")

        return AdapterOutput(
            model_name=self.spec.name,
            checkpoint=self.spec.checkpoint,
            code_tokens=list(code_tokens),
            subtokens=subtokens,
            subtoken_groups=groups,
            input_tokens=input_tokens,
            lexical_positions=positions,
            attention=None,
            hidden_repr=hidden_repr,
            metadata={
                "adapter_contract": self.spec.to_dict(),
                "representation_source": "mamba_hidden_states",
                "supports_attention": False,
                "ignored_subtoken_indices": ignored,
                "use_cache": False,
            },
        )
