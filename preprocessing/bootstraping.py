"""
Source: https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py
NOTE: Note/Inst gen usually bounded on TPM, not RPM
thus, simply ignore rpm cases
Also, the file size can be loaded into mem
"""

import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import random
from dataclasses import dataclass  # for storing API inputs, outputs, and metadata

# imports
import aiohttp  # for making API calls concurrently
import pandas as pd

global_prompt = """You are a healthcare professional who often uses an advanced clinical language model that can analyze clinical notes.
You can ask a question and use the output from this model to aid in your decision-making process.

Let's suppose you want to ask a question about the task of "{task}" related to the given discharge summary.
How would you frame your question?

[Discharge Summary Begin]
{note}
[Discharge Summary End]

We also offer five example questions related to the task for your reference.
Although you can refer to these samples, try to create a distinct question.
Try not to repeat the verb or structure of sentences used in examples to maximize diversity.
The crucial aspect of your question is that it should be answerable using only the information available in the discharge summary.
Also, try to keep your question as brief as possible.

[Example Questions Begin]
{samples}
[Example Questions End]

Please output solely a question."""

tasks = [
    {
        "task": "Named Entity Recognition",
        "samples": [
            "Can you identify and categorize all the medical conditions?",
            "Could you extract all medications prescribed and their corresponding dosages?",
            "What were the diagnostic tests and their results?",
            "Identify all the medical procedures.",
            "Can you recognize and list any life style factors such as smoking, alcohol consumption, or exercise habits?",
        ],
        "pool": [],
    },
    {
        "task": "Abbreviation Expansion",
        "samples": [
            "What is the expanded form of the abbreviation 'COPD'?",
            "Could you decode all the abbreviations?",
            "'HTN' has been mentioned frequently in this report. Could you clarify what it means?",
            "In this medical report, what would 'CAD' typically stand for?",
            "I see 'DM' used here in relation to a patient's condition, could you provide the full term?",
        ],
        "pool": [],
    },
    {
        "task": "Relation Extraction",
        "samples": [
            "Identify the connection between the medication 'Metformin' and the patient's diagnosed condition 'Type 2 diabetes'.",
            "Establish the relation between the prescribed dosage of 'Lisinopril' and the patient's 'hypertension' management.",
            "Can you find the link between the use of 'antibiotics' and the patient's 'post-operative infection'?",
            "Determine the association between the patient's lifestyle modifications, specifically 'diet and exercise', and the improvement in 'cholesterol levels'.",
            "Decipher the relationship between the 'radiation therapy' administered and the progress of the patient's 'breast cancer'.",
        ],
        "pool": [],
    },
    {
        "task": "Temporal Information Extraction",
        "samples": [
            "Can you extract the duration of the patient's treatment?",
            "Identify the duration of the hospital stay mentioned in the discharge summary.",
            "Retrieve the timestamps of any surgical procedures.",
            "Extract the date and time of the last medication administration.",
            "Identify any temporal references to follow-up appointments or scheduled tests in the summary.",
        ],
        "pool": [],
    },
    {
        "task": "Coreference Resolution",
        "samples": [
            "Which medication does 'it' refer to in the line mentioning 'it should be taken twice daily' in the medication instructions section?",
            "Please clarify what 'this procedure' refers to in the surgeon's notes section of the discharge summary.",
            "Identify the coreferents for the pronouns used in the second paragraph.",
            "In the patient education section, when 'these exercises' are mentioned, what specific exercises are being referred to?",
            "Who does 'he' refer to in the sentence 'He is expected to recover fully' in the prognosis section?",
        ],
        "pool": [],
    },
    {
        "task": "Paraphrasing",
        "samples": [
            "The discharge summary states that the patient suffered from 'an anomalous blockage in the coronary artery.' Could you paraphrase this medical term into simpler language that the patient might understand?",
            "How would you rephrase the line in the discharge summary, 'Patient exhibits signs of acute rhinosinusitis,' to make it easier for a non-medical professional to grasp?",
            "In this discharge summary, it mentions 'diabetes mellitus type 2 with hyperglycemia.' Can you provide a paraphrase that might be more straightforward for the patient and their family?",
            "The term 'post-operative seroma' appears in the patient's discharge summary. Can you paraphrase this to a less clinical terminology?",
            "Could you translate the sentence, 'The patient's condition was complicated by acute renal failure due to ischemia,' into more common terms to aid in communicating the situation to the patient?",
        ],
        "pool": [],
    },
    {
        "task": "Summarization",
        "samples": [
            "Can you provide a succinct summary of the key clinical findings and treatment recommendations outlined in this discharge summary?",
            "Can you identify and condense any lifestyle and medication modifications recommended in the patient's discharge summary?",
            "Given the patient's discharge summary, can you extract the diagnosis and prognosis information and summarize it in layman's terms for the patient's understanding?",
            "Could you extract and summarize the patient's progress during hospitalization, as well as key notes regarding her discharge planning?",
            "What were the key findings from the lab tests, imaging, and other diagnostic procedures? Please summarize these in simple terms.",
        ],
        "pool": [],
    },
    {
        "task": "Question Answering",
        "samples": [
            "Considering the 'uncontrolled diabetes' statement in the hospital release notes, what lifestyle changes and medication revisions can be recommended?",
            "Identify all the instances suggesting 'adverse reactions' from drugs mentioned in the discharge synopsis.",
            "In light of the 'congestive heart failure' diagnosis in the patient's discharge summary, what are the subsequent tests and procedures that need to be arranged?",
            "Locate all references to 'dietary restrictions' in the discharge document and provide an explanation for each constraint.",
            "Based on the 'chronic kidney disease' mention in the discharge documents, what routine follow-up strategy and patient awareness should be put into action?",
        ],
        "pool": [],
    },
]


@dataclass
class APICaller:
    requests_filepath: str
    save_filepath: str
    request_url: str
    api_key: str
    max_requests_per_minute: float
    max_tokens_per_minute: float
    token_encoding_name: str
    max_attempts: int
    logging_level: int

    def __post_init__(self):
        self.tasks = tasks
        self.sema = asyncio.Semaphore(1)
        self.rate_limit_sleep = False
        self.api_endpoint = "chat/completions"
        if "azure" in self.request_url:
            self.request_headers = {"api-key": self.api_key}
        else:
            self.request_headers = {"Authorization": f"Bearer {self.api_key}"}
        self.task_id_generator = task_id_generator_function()

    async def run(self):
        df = pd.read_csv(self.requests_filepath)
        if "TEXT" in df.columns:
            df.rename(columns={"TEXT": "note"}, inplace=True)
        df["idx"] = df.apply(lambda x: random.randint(0, 7), axis=1)
        data = df[["note", "idx"]].to_dict(orient="records")
        tasks = [self.await_and_call(i) for i in data]
        await asyncio.gather(*tasks)

    async def await_and_call(self, request):
        """
        request: {"note":note text, "idx": task_idx}
        """
        async with self.sema:
            if self.rate_limit_sleep:
                await asyncio.sleep(10)
                self.rate_limit_sleep = False
            await asyncio.sleep(
                2048 * 60 / self.max_tokens_per_minute
            )  # Max 4000, but actually about ~2000
        task_id = next(self.task_id_generator)
        samples = random.sample(self.tasks[request["idx"]]["samples"], 3)
        pool_num_samples = min(len(self.tasks[request["idx"]]["pool"]), 2)
        samples += random.sample(self.tasks[request["idx"]]["pool"], pool_num_samples)
        samples = "\n".join(samples)

        prompt = global_prompt.format(
            task=self.tasks[request["idx"]]["task"],
            samples=samples,
            note=request["note"],
        )

        request_json = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            # Max of GPT-3.5-Turbo
            "max_tokens": 300,
            "temperature": 1,
        }

        res = await self.call_api(task_id, request_json)
        try:
            self.tasks[request["idx"]]["pool"].append(
                res["choices"][0]["message"]["content"]
            )
        except:
            # If Content Filtered
            pass

    async def call_api(
        self,
        task_id: int,
        request_json: dict,
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"Starting request #{task_id}")
        for _ in range(self.max_attempts):
            error = None
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url=self.request_url,
                        headers=self.request_headers,
                        json=request_json,
                    ) as response:
                        response = await response.json(content_type=None)
                if "error" in response:
                    logging.warning(
                        f"Request {task_id} failed with error {response['error']}"
                    )
                    error = response
                    if "Rate limit" in response["error"].get("message", ""):
                        self.rate_limit_sleep = True
                else:
                    break
            except (
                Exception
            ) as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
                logging.warning(f"Request {task_id} failed with Exception {e}")
                error = e
        if error:
            logging.error(
                f"Request {request_json} failed after all attempts. Saving errors"
            )
            append_to_jsonl([request_json, [str(error)]], self.save_filepath)
            return None
        else:
            append_to_jsonl([request_json, response], self.save_filepath)
            logging.debug(f"Request {task_id} saved to {self.save_filepath}")
            return response


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


# run script
async def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path")
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--request_url", default="https://api.openai.com/v1/embeddings")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--max_requests_per_minute", type=int, default=3_000 * 0.5)
    parser.add_argument("--max_tokens_per_minute", type=int, default=250_000 * 0.5)
    parser.add_argument("--token_encoding_name", default="cl100k_base")
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--logging_level", default=logging.INFO)
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = args.input_path.replace(".jsonl", "_results.jsonl")

    caller = APICaller(
        requests_filepath=args.input_path,
        save_filepath=args.save_path,
        request_url=args.request_url,
        api_key=args.api_key,
        max_requests_per_minute=float(args.max_requests_per_minute),
        max_tokens_per_minute=float(args.max_tokens_per_minute),
        token_encoding_name=args.token_encoding_name,
        max_attempts=int(args.max_attempts),
        logging_level=int(args.logging_level),
    )
    await caller.run()


if __name__ == "__main__":
    asyncio.run(main())
