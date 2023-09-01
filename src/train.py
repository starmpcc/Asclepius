# Adapted From: https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from dataclasses import dataclass
from typing import Dict

import torch
import transformers
from datasets import load_from_disk

from utils import *

if "A100" in torch.cuda.get_device_name():
    from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

    replace_llama_attn_with_flash_attn()


@dataclass
class Collator(object):
    def __call__(self, instances):
        input_ids = torch.stack([torch.LongTensor(i["input_ids"]) for i in instances])
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


def make_data_module(data_args) -> Dict:
    train_dataset = load_from_disk(data_args.data_path)
    data_collator = Collator()
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    revision = (
        "pr/7" if "decapoda-research/llama" in model_args.model_name_or_path else "main"
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        revision=revision,
    )
    data_module = make_data_module(data_args=data_args)

    trainer = Trainer(model=model, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()
