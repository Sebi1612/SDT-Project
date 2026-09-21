"""One-input CPU control for PLBART attention stochasticity (no inference rerun)."""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import PLBartForConditionalGeneration, PLBartTokenizer


OUTPUT = Path(__file__).with_name("plbart_dropout_check.json")
MODEL = "uclanlp/plbart-base"


def attentions(model, input_ids):
    with torch.no_grad():
        result = model.model.encoder(input_ids=input_ids, output_attentions=True, return_dict=True)
    return [layer.detach().cpu().numpy() for layer in result.attentions]


def layer_differences(first, second):
    return [float(np.mean(np.abs(a - b))) for a, b in zip(first, second)]


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    tokenizer = PLBartTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = PLBartForConditionalGeneration.from_pretrained(
        MODEL, local_files_only=True, output_attentions=True
    )
    model.to("cpu")
    input_ids = tokenizer("def small(x): return x + 1", return_tensors="pt").input_ids
    initial_training_mode = model.training
    default_one = attentions(model, input_ids)
    default_two = attentions(model, input_ids)
    model.train()
    training_one = attentions(model, input_ids)
    training_two = attentions(model, input_ids)
    model.eval()
    eval_one = attentions(model, input_ids)
    eval_two = attentions(model, input_ids)
    result = {
        "model": MODEL,
        "input_subtokens": int(input_ids.shape[1]),
        "initial_training_mode": initial_training_mode,
        "dropout_config": float(model.config.dropout),
        "attention_dropout_config": float(model.config.attention_dropout),
        "default_mean_absolute_attention_difference_by_layer": layer_differences(default_one, default_two),
        "forced_training_mean_absolute_attention_difference_by_layer": layer_differences(training_one, training_two),
        "eval_mean_absolute_attention_difference_by_layer": layer_differences(eval_one, eval_two),
        "interpretation": "In this installed transformers version, from_pretrained defaults to eval mode. Missing an explicit eval() call is not a supported explanation for the observed PLBART reference mismatch.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
