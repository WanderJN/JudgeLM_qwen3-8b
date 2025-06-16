"""Generate answers with local models.

"""
import argparse
import json
import os
import time

import shortuuid
import torch
from tqdm import tqdm

import sys
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
root = file.parents[2]
sys.path.append(str(root))
print(sys.path)

from transformers import AutoModelForCausalLM, AutoTokenizer

from judge.common import load_questions, reorg_answer_file, KeywordsStoppingCriteria, parse_score, translate_score_to_win_list
from utils import extract_jsonl


def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    temperature,
    if_reverse_answers,
    reference_file,
    if_fast_eval
):
    print("start run_eval")
    questions = load_questions(question_file, question_begin, question_end)
    if reference_file is not None:
        references = load_questions(reference_file, question_begin, question_end)

    model = AutoModelForCausalLM.from_pretrained(model_path).eval().to("cuda:0")
    model = model.to(torch.bfloat16)  # 将模型权重转换为 bf16
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    chunk_size = 4
    ans_handles = []
    print("start ans_handles append")
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_model_answers(
                model,
                tokenizer,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                temperature,
                if_reverse_answers,
                references[i : i + chunk_size] if reference_file is not None else None,
                if_fast_eval,
            )
        )

@torch.inference_mode()
def get_model_answers(
    model,
    tokenizer,
    model_id,
    questions,
    answer_file,
    max_new_token,
    temperature,
    if_reverse_answers,
    references,
    if_fast_eval,
):
    print("start load model")
    

    for q_i, question in tqdm(enumerate(questions)):
        torch.manual_seed(q_i)
        conv_judge_pair = {
            "system" : 'You are a helpful and precise assistant for checking the quality of the answer.',
            "prompt" : "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.",
            "prompt_template" : "[Question]\n{question}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n\n[The End of Assistant 2's Answer]\n\n[System]\n{prompt}\n\n",
            "appendix" : "### Response:",
        }
        conv_judge_pair_w_reference = {
                "system" : 'You are a helpful and precise assistant for checking the quality of the answer.',
                "prompt" : "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.",
                "prompt_template" : "[Question]\n{question}\n\n[Reference Answer]\n{reference}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n\n[The End of Assistant 2's Answer]\n\n[System]\n{prompt}\n\n",
                "appendix" : "### Response:",
            }
        conv = conv_judge_pair if references is None else conv_judge_pair_w_reference
        template = conv["prompt_template"]

        # if fast eval, use the "\n" as the separator
        sep = "<|endoftext|>"     # qwen模型的结束符号
        if if_fast_eval:
            sep = "\n"

        # reverse the order of the answers
        if if_reverse_answers:
            temp_answer = question["answer1_body"]
            question["answer1_body"] = question["answer2_body"]
            question["answer2_body"] = temp_answer

        data_sample = [{"role": "system", "content": conv["system"]}]
        # combine data_sample
        if references is None:
            data_sample.append({"role": "user", "content": template.format(question=question['question_body'],
                                                               answer_1=question['answer1_body'],
                                                               answer_2=question['answer2_body'],
                                                               prompt=conv["prompt"]) + conv["appendix"]
                                                               })
        else:
            data_sample.append({"role": "user", "content": template.format(question=question['question_body'],
                                                               reference=references[q_i]['reference']['text'],
                                                               answer_1=question['answer1_body'],
                                                               answer_2=question['answer2_body'],
                                                               prompt=conv["prompt"]) + conv["appendix"]
                                                               })

        input_ids = tokenizer(data_sample).input_ids

        do_sample = False if temperature < 1e-4 else True
        stopping_criteria = KeywordsStoppingCriteria([sep], tokenizer, torch.as_tensor(input_ids))

        # generate judgements
        output_ids = model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_token,
            stopping_criteria=[stopping_criteria]
        )

        if model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]) :]

        output = tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
        )

        if sep:
            output = output[: output.find(sep)]
        output = output.strip()

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_id = shortuuid.uuid()
            question["pred_id"] = ans_id
            question["pred_text"] = output
            question["pred_model_id"] = model_id
            question["tstamp"] = time.time()
            if references is not None:
                question["reference"] = references[q_i]['reference']['text']
            fout.write(json.dumps(question) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--question-file",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=2048,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--if-reverse-answers",
        type=int,
        default=0,
        help="Whether to reverse the order of the answers.",
    )
    parser.add_argument(
        "--reference-file",
        type=str,
        default=None,
        help="The reference file for evaluation.",
    )
    parser.add_argument(
        "--if-fast-eval",
        type=int,
        default=0,
        help="Whether to use fast evaluation.",
    )
    args = parser.parse_args()
    args.if_reverse_answers = bool(args.if_reverse_answers)
    args.if_fast_eval = bool(args.if_fast_eval)
    if args.reference_file == 'None':
        args.reference_file = None
    print(f"args: {args}")

    print(f"Output to {args.answer_file}")
    
    run_eval(
        args.model_path,
        args.model_id,
        args.question_file,
        args.question_begin,
        args.question_end,
        args.answer_file,
        args.max_new_token,
        args.temperature,
        args.if_reverse_answers,
        args.reference_file,
        args.if_fast_eval
    )

    reorg_answer_file(args.answer_file)

    # statistics the judgements
    sequential_pred_answer_file_list = extract_jsonl(args.answer_file)

    sequential_pred_score_list = []
    for sequential_pred_answer_file in sequential_pred_answer_file_list:
        sequential_pred_score_list.append(parse_score(sequential_pred_answer_file['pred_text']))

    # if the score gap is less than T, we consider it as a draw
    T = 0.0
    sequential_pred_win_list = translate_score_to_win_list(sequential_pred_score_list, T)

    # get the number of 1 in sequential_pred_win_list
    win_num = sequential_pred_win_list.count(1)
    tie_num = sequential_pred_win_list.count(0)
    lose_num = sequential_pred_win_list.count(-1)

    # print the win, tie, and lose number, use format {}
    print("Assistant 1's reuslts ---> win_num: {}, tie_num: {}, lose_num: {}".format(win_num, tie_num, lose_num))
    print("Assistant 2's reuslts ---> win_num: {}, tie_num: {}, lose_num: {}".format(lose_num, tie_num, win_num))
