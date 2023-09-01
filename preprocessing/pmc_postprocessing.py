import argparse
import random
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
    df["case_report"] = (
        df[0]
        .map(lambda x: x["messages"][0]["content"])
        .map(
            lambda x: re.findall(
                r"\[The start of case report\]\n(.*)\n\[The end of case report\]",
                x,
                re.DOTALL | re.MULTILINE,
            )[0]
        )
    )
    df["note"] = df[1].map(lambda x: x["choices"][0]["message"]["content"])

    df["idx"] = df.apply(lambda x: random.randint(0, 7), axis=1)

    df[["case_report", "note", "idx"]].to_json(
        args.save_path, orient="records", indent=4
    )
    return


if __name__ == "__main__":
    main()
