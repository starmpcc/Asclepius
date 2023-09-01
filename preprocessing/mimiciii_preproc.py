import argparse

import pandas as pd

from preprocessing.convert_deid_tag import replace_list_of_notes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument(
        "--filter_index_path",
        type=str,
        default="preprocessing/mimiciii_filter_index.txt",
    )
    parser.add_argument("--save_path", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    notes = pd.read_csv(args.input_path)

    with open(args.filter_index_path, "r") as f:
        erase_index = f.readlines()
    erase_index = [int(i) for i in erase_index]

    preprocessed_notes = notes[
        (~notes["ROW_ID"].isin(erase_index))
        & (notes["CATEGORY"] == "Discharge summary")
    ]

    preprocessed_notes["TEXT"] = preprocessed_notes["TEXT"].map(
        lambda x: replace_list_of_notes([x])[0]
    )

    preprocessed_notes.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    main()
