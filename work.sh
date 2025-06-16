# 切换工作目录，安库，设置python路径
cd /lpai/volumes/base-copilot-ali-sh/wangjianing1/JudgeLM
export PYTHONPATH=/lpai/volumes/base-copilot-ali-sh/wangjianing1/JudgeLM:$PYTHONPATH


# 环境准备，更新transformers,torchao，安装deepspeed,accelerate
python -c "from transformers import __version__; print(__version__)"
pip install transformers --upgrade
pip install --upgrade torchao
pip install deepspeed
pip install accelerate


# CUDA_VISIBLE_DEVICES=0 python train.py  # 检查单卡运行程序是否报错


# 启动单机多卡微调
deepspeed --include="localhost:0,1,2,3,4,5,6,7" \
    --master_port 60000 \
    train.py \
    --deepspeed=deepspeed_config.json \
    --data_path "data/judgelm_train_100k.jsonl" \
    --output_dir "train_result" \
    --model_save_path "sft_model" \
    --weight_decay 0. \
    --warmup_ratio 0.03


# # 训练结束使用占卡脚本占卡
# python /lpai/volumes/base-copilot-ali-sh/wangjianing1/gpu_90_work.py


# 测试微调后的模型
# python test.py