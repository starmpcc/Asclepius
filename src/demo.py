# Reference: Alpaca & Vicuna

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_prompt, modify_special_tokens


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--cache_dir", type=str)

    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        "decapoda-research/llama-7b-hf", revision="pr/7", use_fast=False
    )
    tokenizer = modify_special_tokens(tokenizer)
    subfolder = "result" if args.model_name == "zl111/ChatDoctor" else ""
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        subfolder=subfolder,
        cache_dir=args.cache_dir,
    ).to("cuda")
    prompt = get_prompt(args.model_name)

    while True:
        example = {
            "question": input("Enter instruction: "),
            "note": input("Enter input: "),
        }
        text = prompt.format_map(example)

        tokens = tokenizer.encode(text, return_tensors="pt").to("cuda")
        output = model.generate(
            tokens, max_length=2048, do_sample=True, temperature=1, num_beams=5
        )
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        try:
            answer = result[len(text) : result.index("</s>", len(text))].strip()
        except:
            answer = result[len(text) :].strip()
        print(answer)


if __name__ == "__main__":
    main()
