"""Hugging Face adapters matching the model use described in the paper."""

from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PLBartForConditionalGeneration,
    PLBartTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaModel,
    RobertaTokenizer,
    T5ForConditionalGeneration,
)

from .base import ModelAdapter, ModelSpec


class JoinedTokenAdapter(ModelAdapter):
    """Common input tokenization used by the original repository."""

    def _joined_subtokens(self, code_tokens: Sequence[str]) -> List[str]:
        return list(self.tokenizer.tokenize(" ".join(code_tokens)))

    def _check_sequence(self, input_tokens: Sequence[str]) -> None:
        configured = getattr(self.model.config, "max_position_embeddings", None)
        if configured is not None and len(input_tokens) > configured:
            raise ValueError(
                f"Input length {len(input_tokens)} exceeds model capacity "
                f"{configured}; inputs are never silently truncated"
            )

    def _check_paper_limit(self, subtokens: Sequence[str]) -> None:
        if len(subtokens) > self.spec.paper_max_subtokens:
            raise ValueError(
                f"{len(subtokens)} subtokens exceed the paper protocol limit "
                f"of {self.spec.paper_max_subtokens}"
            )

    def _ids(self, input_tokens: Sequence[str]) -> torch.Tensor:
        ids = self.tokenizer.convert_tokens_to_ids(list(input_tokens))
        return torch.tensor([ids], dtype=torch.long, device=self.device)


class GraphCodeBERTAdapter(JoinedTokenAdapter):
    """Paper-compatible GraphCodeBERT inference over plain code tokens."""

    def load(self) -> "GraphCodeBERTAdapter":
        kwargs = {"local_files_only": self.local_files_only}
        self.tokenizer = RobertaTokenizer.from_pretrained(
            self.spec.checkpoint, **kwargs
        )
        self.model = RobertaForMaskedLM.from_pretrained(
            self.spec.checkpoint,
            attn_implementation="eager",
            **kwargs,
        ).to(self.device)
        self.model.eval()
        self.validate_loaded_model()
        return self

    def _prepare_and_forward(self, code_tokens):
        subtokens = self._joined_subtokens(code_tokens)
        self._check_paper_limit(subtokens)
        input_tokens = [self.tokenizer.cls_token, *subtokens, self.tokenizer.sep_token]
        self._check_sequence(input_tokens)
        input_ids = self._ids(input_tokens)
        with torch.inference_mode():
            outputs = self.model.roberta(
                input_ids=input_ids,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
        return (
            subtokens,
            input_tokens,
            list(range(1, 1 + len(subtokens))),
            outputs,
            {
                "representation_source": "roberta_encoder",
                "graph_inputs_at_inference": False,
                "masked_lm_head_executed": False,
            },
        )


class UniXcoderAdapter(JoinedTokenAdapter):
    """UniXcoder in the paper repository's bidirectional encoder-only mode."""

    MODE_TOKEN = "<encoder-only>"

    def load(self) -> "UniXcoderAdapter":
        kwargs = {"local_files_only": self.local_files_only}
        self.tokenizer = RobertaTokenizer.from_pretrained(
            self.spec.checkpoint, **kwargs
        )
        config = RobertaConfig.from_pretrained(self.spec.checkpoint, **kwargs)
        config.is_decoder = True
        config.output_attentions = True
        config.output_hidden_states = True
        config._attn_implementation = "eager"
        self.model = RobertaModel.from_pretrained(
            self.spec.checkpoint, config=config, **kwargs
        ).to(self.device)
        self.model.eval()
        self.validate_loaded_model()
        return self

    def _prepare_and_forward(self, code_tokens):
        subtokens = self._joined_subtokens(code_tokens)
        self._check_paper_limit(subtokens)
        input_tokens = [
            self.tokenizer.cls_token,
            self.MODE_TOKEN,
            self.tokenizer.sep_token,
            *subtokens,
            self.tokenizer.sep_token,
        ]
        # The original wrapper uses max_length=512 and four special tokens.
        if len(input_tokens) > 512:
            raise ValueError(
                f"UniXcoder input length {len(input_tokens)} exceeds 512; "
                "inputs are never silently truncated"
            )
        self._check_sequence(input_tokens)
        input_ids = self._ids(input_tokens)
        present = input_ids.ne(self.model.config.pad_token_id)
        attention_mask = present.unsqueeze(1) * present.unsqueeze(2)
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        return (
            subtokens,
            input_tokens,
            list(range(3, 3 + len(subtokens))),
            outputs,
            {
                "representation_source": "unixcoder_encoder_only",
                "mode_token": self.MODE_TOKEN,
                "attention_mask": "bidirectional_pairwise",
            },
        )


class T5EncoderAdapter(JoinedTokenAdapter):
    """CodeT5-family encoder representations, without an unused decoder pass."""

    def load(self) -> "T5EncoderAdapter":
        kwargs = {"local_files_only": self.local_files_only}
        self.tokenizer = RobertaTokenizer.from_pretrained(
            self.spec.checkpoint, **kwargs
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.spec.checkpoint,
            attn_implementation="eager",
            **kwargs,
        ).to(self.device)
        self.model.eval()
        self.validate_loaded_model()
        return self

    def _prepare_and_forward(self, code_tokens):
        subtokens = self._joined_subtokens(code_tokens)
        self._check_paper_limit(subtokens)
        input_tokens = [self.tokenizer.cls_token, *subtokens, self.tokenizer.sep_token]
        self._check_sequence(input_tokens)
        input_ids = self._ids(input_tokens)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            outputs = self.model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
        return (
            subtokens,
            input_tokens,
            list(range(1, 1 + len(subtokens))),
            outputs,
            {
                "representation_source": "encoder",
                "decoder_executed": False,
            },
        )


class PLBartEncoderAdapter(JoinedTokenAdapter):
    """PLBART encoder representations, matching the paper's analyzed side."""

    def load(self) -> "PLBartEncoderAdapter":
        kwargs = {"local_files_only": self.local_files_only}
        self.tokenizer = PLBartTokenizer.from_pretrained(
            self.spec.checkpoint, **kwargs
        )
        self.model = PLBartForConditionalGeneration.from_pretrained(
            self.spec.checkpoint,
            attn_implementation="eager",
            **kwargs,
        ).to(self.device)
        self.model.eval()
        self.validate_loaded_model()
        return self

    def _prepare_and_forward(self, code_tokens):
        subtokens = self._joined_subtokens(code_tokens)
        self._check_paper_limit(subtokens)
        input_tokens = [self.tokenizer.cls_token, *subtokens, self.tokenizer.sep_token]
        self._check_sequence(input_tokens)
        input_ids = self._ids(input_tokens)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            outputs = self.model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
        return (
            subtokens,
            input_tokens,
            list(range(1, 1 + len(subtokens))),
            outputs,
            {
                "representation_source": "encoder",
                "decoder_executed": False,
            },
        )


class CodeGenAdapter(JoinedTokenAdapter):
    """CodeGen2 decoder-only hidden states and causal self-attention."""

    def load(self) -> "CodeGenAdapter":
        kwargs = {
            "local_files_only": self.local_files_only,
            "trust_remote_code": True,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.spec.checkpoint, **kwargs
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.spec.checkpoint,
            attn_implementation="eager",
            **kwargs,
        ).to(self.device)
        self.model.eval()
        self.validate_loaded_model()
        return self

    def _prepare_and_forward(self, code_tokens):
        subtokens = self._joined_subtokens(code_tokens)
        self._check_paper_limit(subtokens)
        input_tokens = list(subtokens)
        self._check_sequence(input_tokens)
        input_ids = self._ids(input_tokens)
        with torch.inference_mode():
            outputs = self.model.transformer(
                input_ids=input_ids,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
        return (
            subtokens,
            input_tokens,
            list(range(len(subtokens))),
            outputs,
            {
                "representation_source": "causal_decoder",
                "attention_mask": "causal",
                "language_model_head_executed": False,
            },
        )


ADAPTER_CLASSES = {
    "graphcodebert": GraphCodeBERTAdapter,
    "unixcoder": UniXcoderAdapter,
    "codet5": T5EncoderAdapter,
    "plbart": PLBartEncoderAdapter,
    "codet5p_220": T5EncoderAdapter,
    "codegen": CodeGenAdapter,
}
