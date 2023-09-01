# Reference: Alpaca & Vicuna

import argparse
import io
import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_prompt, modify_special_tokens


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--cache_dir", type=str)
    return parser.parse_args()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    jdict = json.load(f)
    f.close()
    return jdict


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        "decapoda-research/llama-7b-hf", revision="pr/7", use_fast=False
    )
    subfolder = "result" if args.model_name == "zl111/ChatDoctor" else ""
    max_memory = None
    if "13" in args.model_name or "camel" in args.model_name:
        if "A100" not in torch.cuda.get_device_name():
            max_memory = {0: "40GiB", 1: "48GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        subfolder=subfolder,
        cache_dir=args.cache_dir,
        device_map="auto",
        max_memory=max_memory,
    )
    model = torch.compile(model)

    data = jload(args.input_path)

    tokenizer = modify_special_tokens(tokenizer)
    answers = []
    prompt = get_prompt(args.model_name)
    for sample in tqdm(data):
        for k, v in sample.items():
            sample[k] = v.strip("\n")
        text = prompt.format_map(sample)

        tokens = tokenizer.encode(text, return_tensors="pt").to("cuda")
        output = model.generate(
            tokens,
            max_new_tokens=400,
            num_beams=5,
            do_sample=True,
            temperature=1,
            eos_token_id=[2],
            use_cache=True,
        )
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        try:
            answer = result[len(text) : result.index("</s>", len(text))].strip()
        except:
            answer = result[len(text) :].strip()
        answers.append({"generated": answer})
    with open(args.save_path, "w") as f:
        json.dump(answers, f)


if __name__ == "__main__":
    main()
