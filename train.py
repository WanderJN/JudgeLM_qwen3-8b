import copy
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers

from transformers import Trainer

import sys
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
root = file.parents[2]
sys.path.append(str(root))
print(sys.path)

from dataset import LazyJudgeSupervisedDataset, DataCollatorForSupervisedDataset


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="./qwen3-8b")
    model_save_path: Optional[str] = field(default="./sft_model")


@dataclass
class DataArguments:
    data_path: str = field(
        default="./data/judgelm_train_100k.jsonl", metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    swap_aug_ratio: float = -1.0
    ref_drop_ratio: float = -1.0


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir : str = field(default="./train_result")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    num_train_epochs: int = field(default=3)
    do_train: bool = field(default=True) 
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=32)
    logging_steps: int = field(default=1)
    report_to: str = field(default="tensorboard")
    save_total_limit : int = field(default=1)             # 保存检查点总数限制
    bf16 : bool = field(default=True)
    learning_rate : float = field(default=2e-5)
    lr_scheduler_type: str = field(default='cosine')
    dataloader_num_workers : int = field(default = 1)
    dataloader_pin_memory : bool = field(default=True)   # 是否将数据加载到固定内存
    save_safetensors : bool = field(default=False)
    deepspeed :Optional[str] = field(default=None)


def translate_params_from_str_to_bool(params):
    params_class = type(params)
    params = vars(params)
    for key in params:
        # check if the value is a string
        if not isinstance(params[key], str):
            continue
        if params[key].lower() == "true":
            params[key] = True
        elif params[key].lower() == "false":
            params[key] = False
        elif params[key].lower() == "none":
            params[key] = None

    return params_class(**params)

def train():

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # set data_args from str to bool
    data_args = translate_params_from_str_to_bool(data_args)


    local_rank = training_args.local_rank
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model.config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
     # 在qwen模型中eos_token作为pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # [done]: fix 'RuntimeError: Expected q_dtype == torch::kFloat16 || (is_sm8x && q_dtype == torch::kBFloat16) to be true, but got false.'
    model = model.to(torch.bfloat16) if training_args.bf16 else model.to(torch.float16)

    # 数据集读取
    train_dataset = LazyJudgeSupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, swap_aug_ratio=data_args.swap_aug_ratio, ref_drop_ratio=data_args.ref_drop_ratio)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )
    
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=True)
    trainer.save_model(model_args.model_save_path)
    trainer.save_state()



if __name__ == "__main__":
    train()