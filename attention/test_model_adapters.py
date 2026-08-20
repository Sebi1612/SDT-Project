import types
import unittest

import numpy as np
import torch

from model_adapters.base import (
    AdapterOutput,
    ModelAdapter,
    ModelSpec,
    TokenAlignmentError,
    aggregate_attention,
    aggregate_hidden_states,
    align_subtokens,
    validate_output,
)
from model_adapters.registry import MODEL_SPECS, get_model_spec


class DummyAdapter(ModelAdapter):
    def load(self):
        return self

    def _prepare_and_forward(self, code_tokens):
        raise NotImplementedError


class TokenAlignmentTests(unittest.TestCase):
    def test_roberta_alignment_preserves_dataset_token_nodes(self):
        groups = align_subtokens(
            ["my", "Name", "(", "Ġhello", "Ġworld"],
            ["myName", "(", "hello world"],
            "Ġ",
        )
        self.assertEqual(groups, [[0, 1], [2], [3, 4]])

    def test_sentencepiece_alignment(self):
        groups = align_subtokens(
            ["▁public", "▁get", "Value"],
            ["public", "getValue"],
            "▁",
        )
        self.assertEqual(groups, [[0], [1, 2]])

    def test_literal_whitespace_subtoken_remains_in_string_group(self):
        groups = align_subtokens(
            ["'X", "-", "Trans", "-", "Id", ":", "      ", "'"],
            ["'X-Trans-Id:      '"],
            "Ġ",
        )
        self.assertEqual(groups, [list(range(8))])

    def test_alignment_fails_at_first_divergence(self):
        with self.assertRaisesRegex(TokenAlignmentError, "diverged"):
            align_subtokens(["wrong"], ["right"], "Ġ")


class AggregationTests(unittest.TestCase):
    def test_attention_uses_mean_over_each_subtoken_block(self):
        matrix = torch.tensor(
            [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]
        )
        merged = aggregate_attention((matrix,), [0, 1, 2], [[0, 1], [2]])
        expected = np.asarray([[[[3.0, 4.5], [7.5, 9.0]]]], dtype=np.float32)
        np.testing.assert_allclose(merged, expected)

    def test_hidden_states_include_embedding_and_transformer_layers(self):
        embedding = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]
        )
        transformed = embedding + 10
        merged = aggregate_hidden_states(
            (embedding, transformed), [0, 1, 2], [[0, 1], [2]]
        )
        expected = np.asarray(
            [[[2.0, 3.0], [5.0, 6.0]], [[12.0, 13.0], [15.0, 16.0]]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(merged, expected)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.spec = ModelSpec(
            name="dummy",
            checkpoint="dummy/checkpoint",
            family="encoder-only",
            analyzed_component="encoder",
            input_protocol="code",
            subtoken_marker="Ġ",
            prefix_tokens=1,
            suffix_tokens=1,
            expected_transformer_layers=1,
            expected_attention_heads=1,
            expected_hidden_size=2,
        )

    def test_canonical_output_contract(self):
        output = AdapterOutput(
            model_name="dummy",
            checkpoint="dummy/checkpoint",
            code_tokens=["a", "b"],
            subtokens=["a", "Ġb"],
            subtoken_groups=[[0], [1]],
            input_tokens=["<s>", "a", "Ġb", "</s>"],
            lexical_positions=[1, 2],
            attention=np.zeros((1, 1, 2, 2), dtype=np.float32),
            hidden_repr=np.zeros((2, 2, 2), dtype=np.float32),
            metadata={},
        )
        validate_output(output, self.spec)

    def test_loaded_checkpoint_dimensions_are_checked(self):
        adapter = DummyAdapter(self.spec, device="cpu")
        adapter.model = types.SimpleNamespace(
            config=types.SimpleNamespace(
                num_hidden_layers=2,
                num_attention_heads=1,
                hidden_size=2,
            )
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            adapter.validate_loaded_model()

    def test_decoder_contract_rejects_future_attention(self):
        decoder_spec = ModelSpec(
            **{
                **self.spec.to_dict(),
                "family": "decoder-only",
            }
        )
        output = AdapterOutput(
            model_name="dummy",
            checkpoint="dummy/checkpoint",
            code_tokens=["a", "b"],
            subtokens=["a", "Ġb"],
            subtoken_groups=[[0], [1]],
            input_tokens=["a", "Ġb"],
            lexical_positions=[0, 1],
            attention=np.asarray([[[[1.0, 0.1], [0.9, 0.1]]]], dtype=np.float32),
            hidden_repr=np.zeros((2, 2, 2), dtype=np.float32),
            metadata={},
        )
        with self.assertRaisesRegex(ValueError, "not causal"):
            validate_output(output, decoder_spec)

    def test_codebert_cannot_be_routed_through_new_registry(self):
        with self.assertRaisesRegex(ValueError, "verified legacy"):
            get_model_spec("codebert")

    def test_paper_model_family_semantics_are_explicit(self):
        self.assertEqual(MODEL_SPECS["graphcodebert"].input_protocol, "plain_code_tokens")
        self.assertEqual(MODEL_SPECS["unixcoder"].prefix_tokens, 3)
        for name in ("codet5", "plbart", "codet5p_220"):
            self.assertEqual(MODEL_SPECS[name].analyzed_component, "encoder")
        self.assertEqual(MODEL_SPECS["plbart"].expected_transformer_layers, 6)
        self.assertEqual(MODEL_SPECS["codegen"].analyzed_component, "causal decoder")
        self.assertEqual(MODEL_SPECS["codegen"].expected_hidden_size, 4096)


if __name__ == "__main__":
    unittest.main()
