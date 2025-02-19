from typing import Optional, Tuple
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
import hiq


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 1
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # rsqrt = 每个元素取平方根后再取倒数 NOTE
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0): # 128, 2048, 10000.0
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) # frequencies
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64, [2048, 64]
    return freqs_cis # [2048=序列长度, 64=每个位置配备一个长度为64的向量]

# freqs_cis.shape=[8, 64],复数形式；x.shape=[1, 8, 32, 64],复数形式
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // 1
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )
        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ) # [1, 1024, 32, 128]
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ) # [1, 1024, 32, 128] 这是每个attention层里面都有自己的缓存了
        if hiq.get_env_bool("KV_CAHCHE_IN_GPU", True):
            self.cache_k = self.cache_k.cuda()
            self.cache_v = self.cache_v.cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape # [1, 8, 4096]
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x) # all in shape=[1, 8, 4096]

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim) # [1,8,4096]->[1,8,32,128]
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis) # NOTE, [1,8,32,128]

        self.cache_k = self.cache_k.to(xq) # self.cache_k和xq放同一个gpu
        self.cache_v = self.cache_v.to(xq) # self.cache_v和xq放同一个gpu

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk # 这是把xk缓存起来
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv # 这是把xv缓存起来

        keys = self.cache_k[:bsz, : start_pos + seqlen] # [1, 8, 32, 128], start_pos+seqlen=0+8=8
        values = self.cache_v[:bsz, : start_pos + seqlen] # [1, 8, 32, 128]

        xq = xq.transpose(1, 2) # [1, 32, 8, 128]
        keys = keys.transpose(1, 2) # [1, 32, 8, 128]
        values = values.transpose(1, 2) # [1, 32, 8, 128]
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim) # [1, 32, 8, 8]
        if mask is not None: # mask.shape=[1,1,8,8]
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen) NOTE causal mask, [1, 32, 8, 8]
        scores = F.softmax(scores.float(), dim=-1).type_as(xq) # [1, 32, 8, 8]
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim), [1,32,8,128]
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1) # [1,8,4096]

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs): # layer_id=0, args=ModelArgs(dim=4096, n_layers=32, n_heads=32, vocab_size=32000, multiple_of=256, norm_eps=1e-06, max_batch_size=1, max_seq_len=1024)
        super().__init__()
        self.n_heads = args.n_heads # 32
        self.dim = args.dim # 4096
        self.head_dim = args.dim // args.n_heads # 128
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
            ) # w1: 4096 -> 11008; w2: 11008 -> 4096; w3: 4096 -> 11008
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps) # root mean square layer norm
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out # [1, 8, 4096]


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params # ModelArgs(dim=4096, n_layers=32, n_heads=32, vocab_size=32000, multiple_of=256, norm_eps=1e-06, max_batch_size=1, max_seq_len=1024)
        self.vocab_size = params.vocab_size # 32000
        self.n_layers = params.n_layers # 32 层 transformer decoder

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim) # Embedding(32000, 4096)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False) # 4096 -> 32000

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        ) # (128, 1024*2) --> self.freqs_cis.shape=[2048, 64]

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h) # h.shape=[1, 8, 4096]; 
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()
