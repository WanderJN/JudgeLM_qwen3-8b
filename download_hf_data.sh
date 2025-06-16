pip install -U huggingface_hub

# 建议将下面这一行写入 ~/.bashrc。若没有写入，则每次下载时都需要先输入该命令
export HF_ENDPOINT=https://hf-mirror.com

# 下载指定数据集，--repo-type dataset是指定下载数据集
huggingface-cli download --repo-type dataset BAAI/JudgeLM-100K --local-dir /mnt/pfs-guan-ssai/nlu/wangjianing1/JudgeLM/data
