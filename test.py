from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from dataset import LazyJudgeSupervisedDataset, testDataCollatorForSupervisedDataset


model_path = "./sft_model"
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
print(tokenizer.eos_token_id)

model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=None,
    ).eval().to("cuda:0")
model.config.use_cache = False


"""
# 测试样例
input_data = "请介绍一下你自己"

# 需要将测试数据先改成QWen模型的prompt格式
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": input_data},
]
# 处理成输入
inputs = tokenizer.apply_chat_template(messages, tokenize=True)
input_data = torch.tensor(inputs).unsqueeze(0).to("cuda:0")

# 生成结果
generate_ids = model.generate(input_data, max_new_tokens=10)
res = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
print(res)
"""

# 加载数据
test_dataset = LazyJudgeSupervisedDataset(tokenizer=tokenizer, data_path="data/judgelm_train_100k.jsonl", swap_aug_ratio=-1.0, ref_drop_ratio=-1.0)
data_collator = testDataCollatorForSupervisedDataset(tokenizer=tokenizer)

# 获取数据
data = data_collator([test_dataset[0]])
# 将数据中的每个张量移动到GPU
data = {key: value.to('cuda') for key, value in data.items()}

# 测试第一条样例
generate_ids = model.generate(data["input_ids"], max_new_tokens=1024)
res = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
print(res)
