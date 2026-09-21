"""Save deterministic PLBART encoder attention for a fixed token-ID input."""

import argparse
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import PLBartForConditionalGeneration


FIXED_IDS = [134, 3085, 33460, 33477, 988, 111, 309, 163, 124, 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = PLBartForConditionalGeneration.from_pretrained(
        "uclanlp/plbart-base", local_files_only=True, output_attentions=True
    )
    model.eval()
    with torch.no_grad():
        result = model.model.encoder(
            input_ids=torch.tensor([FIXED_IDS]), output_attentions=True,
            return_dict=True,
        )
    values = np.stack([layer.detach().cpu().numpy() for layer in result.attentions])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, attention=values, transformers_version=transformers.__version__, input_ids=FIXED_IDS)
    print(transformers.__version__, values.shape, args.output)


if __name__ == "__main__":
    main()
