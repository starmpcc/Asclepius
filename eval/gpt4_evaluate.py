import argparse
import json
import random
import time

import openai
import pandas as pd
from tqdm import tqdm

openai.api_key = "YOUR_API_KEY"

prompt = """You are an intelligent clinical language model. 

[Discharge Summary Begin]
{note}
[Discharge Summary End]

[Instruction Begin]
{question}
[Instruction End]

{answers}
Above, we provide you a discharge summary and the instruction that the healthcare professional gave about the discharge summary.
You are also provided with {num_samples} corresponding responses from {num_samples} different clinical models.
Your task is to read the discharge summary and the instruction carefully then find the answer to the instruction. 
Then, compare your answer with each model's response and evaluate the response based on the following criteria.

Criteria : 
1. Unacceptable (1 point): The model's response includes any incorrect or irrelevant contents. If the instruction was unanswerable, the model did not acknowledge this and outputs wrong answer.
2. Poor (2 points): The model's response does not contain any incorrect or irrelevant contents, but omits significant or crucial contents that the instruction is requiring for.
3. Satisfactory (3 points): The model's response does not contain any incorrect or irrelevant contents, but omits minor or insignificant contents that the instruction is requiring for.
4. Excellent (4 points): The model's response contains all necesarry information that the instruction is requiring for. If the instruction was unanswerable, the model correctly acknowledged this and says that it is unanswerable.

When evaluating each score based on above criteria, ensure that each judgement is not affected by other model's response.
First line must contain only {num_samples} values, which indicate the score for each model, respectively.
The {num_samples} scores are separated by a space.
Output scores without explanation.
"""


def generate_inst_prompt(note, question, samples):
    answers = ""
    for i, sample in enumerate(samples):
        sample_name = chr(65 + i)  # Alphabet A, B, C...
        answers += f"[Agent {sample_name}'s Answer Begin]\n{sample}\n[Agent {sample_name}'s Answer End]\n\n"
    return [
        {
            "role": "user",
            "content": prompt.format(
                note=note, question=question, answers=answers, num_samples=len(samples)
            ),
        }
    ]


def make_answer_gpt(message):
    for i in range(10):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-0314", messages=message, max_tokens=2048, temperature=0
            )
        except Exception as e:
            print(e)
            time.sleep(5)
            continue
        return response["choices"][0]["message"]["content"]
    return str(response)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()
    data = pd.read_json(args.input_path, orient="records", lines=True)

    answer_cols = [i for i in data.columns if "answer" in i]

    for _, row in tqdm(data.iterrows()):
        order = list(range(len(answer_cols)))
        random.shuffle(order)

        note = row["note"]
        question = row["question"]
        samples = row[answer_cols].values[order]

        prompt = generate_inst_prompt(note, question, samples)
        answer = make_answer_gpt(prompt)

        answer = answer.strip('"')
        answer = answer.strip("'")
        splitted_answer = answer.split()

        try:
            [splitted_answer[order.index(idx)] for idx in range(len(answer_cols))]
        except:
            for idx, col in enumerate(answer_cols):
                model_name = "_".join(col.split("_")[:-1])
                row[f"{model_name}_score"] = 0
        else:
            for idx, col in enumerate(answer_cols):
                model_name = "_".join(col.split("_")[:-1])
                row[f"{model_name}_score"] = splitted_answer[order.index(idx)]

        row["gpt_response"] = answer
        with open(args.save_path, "a") as f:
            f.write(json.dumps(row.to_dict()) + "\n")


if __name__ == "__main__":
    main()
