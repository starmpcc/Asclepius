import argparse
import os

import pandas as pd
import tiktoken
from pandarallel import pandarallel

pandarallel.initialize(nb_workers=32)

prompt = """You are an intelligent clinical model.

[The start of case report]
{pmc}
[The end of case report]

Based on the patient's case report provided, please generate a synthetic discharge summary in the style of an Electronic Health Record (EHR).

Please follow these requirements:

1. Generate only the discharge summary. Do not generate any other phrases such as notification.
2. If there are any standard clinical terms used in the case report, they should be replaced with their commonly used non-standardized equivalents in the discharge summary. For example, "hypercholesterolemia" can be rewritten as "high cholesterol".
3. The discharge summary include abbreviations that are not defined in the context.
4. The discharge summary may contain minor grammatical errors.
5. Ensure that the discharge summary does not contain any clinical information or details (such as medication names, dosages, treatment plans, diagnoses, procedures, test results, etc.) that are not explicitly mentioned or defined within the given case report.
6. For patients who have not yet been discharged, create a hospital course summary rather than discharge summary.
7. While preserving the structure of the EHR and maintaining medical consistency, generate a detailed and comprehensive discharge summary.
8. The discharge summary is a comprehensive document that can be organized using several distinct headings."""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--save_path", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input_path)

    df["patient"] = df["patient"].map(lambda x: prompt.format(pmc=x))

    df["tokens"] = df["patient"].parallel_map(
        lambda x: len(tiktoken.get_encoding("cl100k_base").encode(x))
    )
    df = df.sample(frac=1, random_state=42)

    df["patient"] = df.apply(
        lambda x: {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": x["patient"],
                }
            ],
            # Max of GPT-3.5-Turbo
            "max_tokens": 4088 - x["tokens"],
            "temperature": 1,
        },
        axis=1,
    )

    df["patient"].to_json(args.save_path, orient="records", lines=True)


if __name__ == "__main__":
    main()
