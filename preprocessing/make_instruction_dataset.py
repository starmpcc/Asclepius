import argparse
import re

import pandas as pd


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
    df["question"] = (
        df[0]
        .map(lambda x: x["messages"][0]["content"])
        .map(
            lambda x: re.findall(
                r"\[Instruction Begin\]\n(.*)\n\[Instruction End\]",
                x,
                flags=re.DOTALL | re.MULTILINE,
            )[0]
        )
    )
    df["answer"] = df[1].map(lambda x: x["choices"][0]["message"]["content"])

    df = df[["note", "question", "answer"]]
    for col in df.columns:
        df = df[
            ~df[col].str.contains(
                r"(AI language)|(language model)|(clinical language)|(recognition model)|(extraction model)|(the model)|(summarization model)|(clinical model)|(generation model)|(AI model)|(NER model)",
                case=False,
                regex=True,
            )
        ]
    df = df[~df["answer"].str.endswith("?")]
    print("Total number of examples:", len(df))
    df.to_json(args.save_path, orient="records", indent=4)


if __name__ == "__main__":
    main()
