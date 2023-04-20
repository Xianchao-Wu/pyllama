import torch

import json
from pathlib import Path
from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def load(
    ckpt_dir: str, # './pyllama_data/7B/'
    tokenizer_path: str, # './pyllama_data/tokenizer.model'
    local_rank: int, # 0
    world_size: int, # 1
    max_seq_len: int, # 1024
    max_batch_size: int, # 1
) -> LLaMA:
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth")) # 找到所有的*.pth，按照编号排序, [PosixPath('pyllama_data/7B/consolidated.00.pth')]
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank] # PosixPath('pyllama_data/7B/consolidated.00.pth')

    checkpoint = torch.load(ckpt_path, map_location="cpu") # 导入到cpu内存中

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read()) # {'dim': 4096, 'multiple_of': 256, 'n_heads': 32, 'n_layers': 32, 'norm_eps': 1e-06, 'vocab_size': -1}

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    ) # ModelArgs(dim=4096, n_layers=32, n_heads=32, vocab_size=-1, multiple_of=256, norm_eps=1e-06, max_batch_size=1, max_seq_len=1024)
    tokenizer = Tokenizer(model_path=tokenizer_path) # 构造tokenizer NOTE 
    model_args.vocab_size = tokenizer.n_words # 词表大小, 32000
    torch.set_default_tensor_type(torch.cuda.HalfTensor) # 半精度 NOTE 初始化Transformer的时候，缺省用半精度fp16
    model = Transformer(model_args) # 构造transformer
    torch.set_default_tensor_type(torch.FloatTensor) # fp32精度, 这是又恢复到了fp32，单精度 NOTE
    model.load_state_dict(checkpoint, strict=False) # 读取checkpoint中的训练好的参数的权重
    generator = LLaMA(model, tokenizer) # 构造LLaMA生成器
    return generator


def run(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 1024,
    max_batch_size: int = 1,
):
    local_rank = 0 # 默认使用第0个gpu
    world_size = 1 # 默认当前只有1张卡
    generator = load( # NOTE 导入模型
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )
    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",  # removed: keep only one prompt
    ] # 一个“提示”例子
    while True:
        print("Prompt:", prompts)
        results = generator.generate(
            prompts, max_gen_len=256, temperature=temperature, top_p=top_p # 0.8, 0.95
        ) # NOTE 构造生成结果
        for result in results:
            print("🦙LLaMA:", result.strip())

        user_input = input("please enter your prompts (Ctrl+C to exit): ")
        prompts = [user_input]


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="/llama_data/7B")
    parser.add_argument(
        "--tokenizer_path", type=str, default="/llama_data/tokenizer.model"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args() # Namespace(ckpt_dir='./pyllama_data/7B/', tokenizer_path='./pyllama_data/tokenizer.model')
    run(
        ckpt_dir=args.ckpt_dir, # './pyllama_data/7B/'
        tokenizer_path=args.tokenizer_path, # './pyllama_data/tokenizer.model'
        temperature=0.8,
        top_p=0.95,
        max_seq_len=1024,
        max_batch_size=1,
    )
