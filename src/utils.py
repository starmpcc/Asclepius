from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import Trainer


def modify_special_tokens(tokenizer):
    tokenizer.add_special_tokens(
        {
            "pad_token": "<s>",
            "eos_token": "</s>",
            "bos_token": "<s>",
            "unk_token": "<unk>",
        }
    )

    tokenizer.eos_token_id = 2
    tokenizer.bos_token_id = 1
    tokenizer.unk_token_id = 0
    tokenizer.pad_token_id = 1

    return tokenizer


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    remove_unused_columns: bool = field(
        default=False,
    )
    dataloader_num_workers: int = field(
        default=16,
    )


PROMPT_DICT = {
    "ours": """You are an intelligent clinical languge model.
Below is a snippet of patient's discharge summary and a following instruction from healthcare professional.
Write a response that appropriately completes the instruction.
The response should provide the accurate answer to the instruction, while being concise.

[Discharge Summary Begin]
{note}
[Discharge Summary End]

[Instruction Begin]
{question}
[Instruction End] 
""",
    "alpaca": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{question}\n\n### Input:\n{note}\n\n### Response:"
    ),
    "medalpaca": (
        "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
        "\n\n### Instruction:\n{question}\n\n### Input:\n{note}\n\n### Response:\n"
    ),
    "chat": """
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: [The start of the Discharge Summary]
{note}
[The end of the Discharge Summary]
{question} ASSISTANT: 
""",
}


def get_prompt(model_name):
    if model_name in ["decapoda-research/llama-7b-hf", "chaoyi-wu/PMC_LLAMA_7B"]:
        print("Using Ours+Response Prompt")
        return PROMPT_DICT["ours"] + "\nResponse: "
    # chatdoctor, alpaca ,medalpaca
    elif model_name in [
        "chavinlo/alpaca-native",
        "zl111/ChatDoctor",
    ]:
        print("Using Alpaca Prompt")
        return PROMPT_DICT["alpaca"]
    elif model_name == "medalpaca/medalpaca-7b":
        print("Using MedAlpaca Prompt")
        return PROMPT_DICT["medalpaca"]
    elif "vicuna" in model_name or "clinical-camel" in model_name:
        print("Using Vicuna Prompt")
        return PROMPT_DICT["chat"]
    else:
        print("Using Our Prompt")
        return PROMPT_DICT["ours"]
