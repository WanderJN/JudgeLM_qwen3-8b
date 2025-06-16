# 切换工作目录，安库，设置python路径
cd /lpai/volumes/base-copilot-ali-sh/wangjianing1/JudgeLM
export PYTHONPATH=/lpai/volumes/base-copilot-ali-sh/wangjianing1/JudgeLM:$PYTHONPATH


# 环境准备，更新transformers,torchao，安装deepspeed,accelerate
python -c "from transformers import __version__; print(__version__)"
pip install transformers --upgrade
pip install --upgrade torchao
pip install deepspeed
pip install accelerate


# 推理测试集
python judge/model_generate.py \
    --model-path \
    --question-file \
    --question-begin  \
    --question-end \
    --max-new-token  \
    --temperature "sft_model"


# # 训练结束使用占卡脚本占卡
# python /lpai/volumes/base-copilot-ali-sh/wangjianing1/gpu_90_work.py
