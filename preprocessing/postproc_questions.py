import argparse
import re

import pandas as pd
import tiktoken

prompt = """You are an intelligent clinical language model.

[Discharge Summary Begin]
{note}
[Discharge Summary End]

[Instruction Begin]
{question}
[Instruction End]

Above, we provide you with a part of the discharge summary and the instruction that the healthcare professional gave about it.
Generate a response to the healthcare professional's instruction using the given discharge summary.

Here are requirements:
- Your response must be accurate and concise to the instruction.
- If the instruction is not fully answerable within the given discharge summary, explain the reason why it is unanswerable using the given information. 
- Do not say that you cannot respond as an AI model.
- Do not ask back nor rephrase the instruction.

Response:"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--save_path", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_json(args.input_path, lines=True)
    df["note"] = (
        df[0]
        .map(lambda x: x["messages"][0]["content"])
        .map(
            lambda x: re.findall(
                r"\[Discharge Summary Begin\]\n(.*)\n\[Discharge Summary End\]",
                x,
                flags=re.DOTALL | re.MULTILINE,
            )[0]
        )
    )
    df["question"] = df[1].map(lambda x: x["choices"][0]["message"]["content"])

    df["input"] = df.apply(
        lambda x: prompt.format(
            note=x["note"],
            question=x["question"],
        ),
        axis=1,
    )

    df["input"].map(
        lambda x: {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": x,
                }
            ],
            # Max of GPT-3.5-Turbo
            "max_tokens": len(tiktoken.get_encoding("cl100k_base").encode(x)) + 400,
            "temperature": 1,
        },
    ).to_json(args.save_path, orient="records", lines=True)


if __name__ == "__main__":
    main()
