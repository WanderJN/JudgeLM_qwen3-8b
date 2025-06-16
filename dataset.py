import copy
from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother


from utils import jlload

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


# 处理qwen数据，不需要了
# def preprocess(
#     sources,
#     tokenizer: transformers.PreTrainedTokenizer,
#     system_message: str = "You are a helpful assistant."
# ) -> Dict:

#     """
#     专门为 Qwen 模型设计的预处理函数
    
#     Qwen 对话格式:
#     <|im_start|>system
#     {system_message}<|im_end|>
#     <|im_start|>user
#     {user_message_1}<|im_end|>
#     <|im_start|>assistant
#     {assistant_message_1}<|im_end|>
#     ...
#     """
#     # Apply prompt templates
#     conversations = []
#     for source in sources:
#         # 若第一条消息不是用户消息则跳过
#         if source[0]["from"] != "human":
#             source = source[1:]
            
#         # 添加系统消息 (Qwen必须)
#         dialogue = f"<|im_start|>system\n{system_message}<|im_end|>\n"

#         # 添加用户和助手的交替对话
#         for sentence in source:
#             role = "user" if sentence["from"] == "human" else "assistant"
#             dialogue += f"<|im_start|>{role}\n{sentence['value']}<|im_end|>\n"
        
#         conversations.append(dialogue.strip())

#     # 分词：Tokenize conversations
#     input_ids = tokenizer(
#         conversations,
#         return_tensors="pt",
#         padding="max_length",
#         max_length=tokenizer.model_max_length,
#         truncation=True,
#     ).input_ids
#     targets = input_ids.clone()

#     # Qwen特殊标记
#     im_start_token = tokenizer.encode("<|im_start|>")[0]
#     im_end_token = tokenizer.encode("<|im_end|>")[0]
#     assistant_token = tokenizer.encode("assistant")[0]

#     # 对于每个对话，target只有在assistant回复部分才保留值，其他地方均为IGNORE_TOKEN_ID
#     for i, (conversation, target) in enumerate(zip(conversations, targets)):
#         # 标记所有位置为忽略 (-100)
#         target[:] = IGNORE_TOKEN_ID  # 默认使用-100，也可根据需求调整
        
#         # 查找所有assistant回复位置
#         tokens = input_ids[i]
#         msg_start = None
        
#         # 遍历token寻找assistant回复
#         for j in range(len(tokens)):
#             # 检测assistant段落开始
#             if tokens[j] == im_start_token:
#                 if j+1 < len(tokens) and tokens[j+1] == assistant_token:
#                     msg_start = j
                    
#             # 检测段落结束，计算需要计算loss的范围
#             elif tokens[j] == im_end_token and msg_start is not None:
#                 # 只保留assistant回复部分的loss
#                 target[msg_start:j] = tokens[msg_start:j]  # 取消忽略此段
#                 msg_start = None

#     return dict(
#         input_ids=input_ids,
#         labels=targets,
#         attention_mask=input_ids.attention_mask,
#     )

class LazyJudgeSupervisedDataset(Dataset):

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, swap_aug_ratio: float, ref_drop_ratio: float):
        Dataset.__init__(self)

        self.tokenizer = tokenizer
        self.swap_aug_ratio = swap_aug_ratio
        self.ref_drop_ratio = ref_drop_ratio

        print("Loading data...")
        self.list_data_dict = jlload(data_path)

        print("Formatting inputs...")

        list_data_dict_cleaned = []
        for data in self.list_data_dict:
            if data['text'] == 'error':
                continue
            list_data_dict_cleaned.append(data)
        self.list_data_dict = list_data_dict_cleaned

    def __len__(self):
        return len(self.list_data_dict)

    def swap_first_two_integers(self, s):
        # find the first space
        first_space_index = s.find(' ')
        if first_space_index != -1:
            # find the second space
            second_space_index = s.find('\n', first_space_index + 1)
            if second_space_index != -1:
                # swap the first two integers
                new_s = s[first_space_index + 1:second_space_index] + ' ' + s[:first_space_index] + '\n' + s[
                                                                                                           second_space_index + 1:]
                return new_s
        return s

    def add_reference(self, data):
        data['text'] = data['text_w_reference']
        data['score'] = data['score_w_reference']
        return data

    def swap_aug(self, data):
        data['answer1_body'], data['answer2_body'] = data['answer2_body'], data['answer1_body']
        data['answer1_id'], data['answer2_id'] = data['answer2_id'], data['answer1_id']
        data['score'] = data['score'][::-1]
        data['answer1_model_id'], data['answer2_model_id'] = data['answer2_model_id'], data['answer1_model_id']
        data['answer1_metadata'], data['answer2_metadata'] = data['answer2_metadata'], data['answer1_metadata']

        data['text'] = self.swap_first_two_integers(data['text'])
        data['text'] = data['text'].replace('Assistant 1', 'Assistant X')
        data['text'] = data['text'].replace('Assistant 2', 'Assistant 1')
        data['text'] = data['text'].replace('Assistant X', 'Assistant 2')

        return data

    def _tokenize_fn(self, strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
        """Tokenize a list of strings."""
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def get_data_sample(self, data, conv, ref_drop_flag):
        template = conv["prompt_template"]

        data_sample = [{"role": "system", "content": conv["system"]}]
        if ref_drop_flag:
            data_sample.append({"role": "user", "content": template.format(question=data['question_body'],
                                                               answer_1=data['answer1_body'],
                                                               answer_2=data['answer2_body'],
                                                               prompt=conv["prompt"]) + conv["appendix"]
                                                               })
        else:
            data_sample.append({"role": "user", "content": template.format(question=data['question_body'],
                                                               reference=data['reference']['text'],
                                                               answer_1=data['answer1_body'],
                                                               answer_2=data['answer2_body'],
                                                               prompt=conv["prompt"]) + conv["appendix"]
                                                               })
        return data_sample

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data = self.list_data_dict[i]


        # 数据模板
        conv_judge_pair = {
            "system" : 'You are a helpful and precise assistant for checking the quality of the answer.',
            "prompt" : "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.",
            "prompt_template" : "[Question]\n{question}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n\n[The End of Assistant 2's Answer]\n\n[System]\n{prompt}\n\n",
            "appendix" : "### Response:",
        }
        conv = conv_judge_pair

        # if ref_drop_ratio >= -0.5, then replace 'text' with 'text_w_reference'
        ref_drop_flag = True
        if self.ref_drop_ratio >= -0.5 and np.random.rand() < self.ref_drop_ratio:
            ref_drop_flag = False
            data = self.add_reference(data)

            conv_judge_pair_w_reference = {
                "system" : 'You are a helpful and precise assistant for checking the quality of the answer.',
                "prompt" : "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.",
                "prompt_template" : "[Question]\n{question}\n\n[Reference Answer]\n{reference}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n\n[The End of Assistant 2's Answer]\n\n[System]\n{prompt}\n\n",
                "appendix" : "### Response:",
            }
            conv = conv_judge_pair_w_reference


        if self.swap_aug_ratio >= -0.5 and np.random.rand() < self.swap_aug_ratio:
            data = self.swap_aug(data)

        # 处理问题数据为字符串
        data_sample = self.get_data_sample(data, conv, ref_drop_flag)
        data_sample = self.tokenizer.apply_chat_template(
            data_sample,
            tokenize=False,
            add_generation_prompt=False
        )

        # 处理回答数据为字符串
        target = [{"role": "assistant", "content": data['text']}]
        target = self.tokenizer.apply_chat_template(
            target,
            tokenize=False,
            add_generation_prompt=False
        )

        example = data_sample + target
        example_tokenized, source_tokenized = [self._tokenize_fn(strings, self.tokenizer) for strings in ([example], [data_sample])]
        input_ids = example_tokenized["input_ids"]
        label = copy.deepcopy(input_ids)
        source_len = source_tokenized["input_ids_lens"][0]
        label[0][:source_len] = IGNORE_TOKEN_ID

        return dict(input_ids=input_ids[0], labels=label[0])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


@dataclass
class testDataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = list([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        # 把input_id只提取成只要输入的部分，labels只提取输出的部分，用于测试模型的输出
        for i in range(len(input_ids)):
            count = 0
            
            for j in range(len(labels[i])):
                if labels[i][j] != IGNORE_TOKEN_ID:
                    count = j
                    break
            input_ids[i]=input_ids[i][:count]
            labels[i]=labels[i][count-1:]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)
        
        return dict(
            input_ids=input_ids,
            labels=labels
        )


if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/lpai/dataset/qwen-copilots/0-4-0/qwen-copilot-models/qwen3-8b",
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
    )
    dataset = LazyJudgeSupervisedDataset(data_path="/lpai/volumes/base-copilot-ali-sh/wangjianing1/JudgeLM/data/judgelm_train_100k.jsonl", tokenizer=tokenizer, swap_aug_ratio=0, ref_drop_ratio=0)
    print(dataset[0])
    print("Finish !")
    