# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch

from llama.tokenizer import Tokenizer


class LLaMA:
    def __init__(self, model, tokenizer: Tokenizer):
        self.model = model # <class 'llama.model_single.Transformer'>
        self.tokenizer = tokenizer # <class 'llama.tokenizer.Tokenizer'>
        
    def _should_stop(self, tokens, prompt_tokens, stop_ids, stop_words):
        """credits go to: https://github.com/galatolofederico/vanilla-llama"""
        if stop_ids is not None:
            do_stop = [False for _ in range(len(tokens))]
            for i, (t, p) in enumerate(zip(tokens, prompt_tokens)):
                g = t[len(p):].tolist()
                for stop_id in stop_ids:
                    if stop_id in g:
                        do_stop[i] = True

            if all(do_stop):
                return True

        if stop_words is not None:
            do_stop = [False for _ in range(len(tokens))]
            for i, (t, p) in enumerate(zip(tokens, prompt_tokens)):
                t = t.clone()
                g = t[len(p):]
                g[g == self.tokenizer.pad_id] = self.tokenizer.eos_id
                g = g.tolist()
                d = self.tokenizer.decode(g)
                for stop_word in stop_words:
                    if stop_word in d:
                        do_stop[i] = True

            if all(do_stop):
                return True

        return False

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int, # 256
        temperature: float = 0.8,
        top_p: float = 0.95,
        stop_ids: List[int] = None,
        stop_words: List[str] = None,
    ) -> List[str]:
        bsz = len(prompts) # batch size = 1
        params = self.model.params # ModelArgs(dim=4096, n_layers=32, n_heads=32, vocab_size=32000, multiple_of=256, norm_eps=1e-06, max_batch_size=1, max_seq_len=1024)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        # [[1, 306, 4658, 278, 6593, 310, 2834, 338]]
        min_prompt_size = min([len(t) for t in prompt_tokens]) # 8
        max_prompt_size = max([len(t) for t in prompt_tokens]) # 8

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size) # min(1024, 264) -> 264

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long() # [1, 264], pad_id=-1
        input_text_mask = tokens != self.tokenizer.pad_id # [1, 264], 8 True and 256 False
        start_pos = min_prompt_size # 8
        prev_pos = 0
        for cur_pos in range(start_pos, total_len): # 当前的位置, starts from 8-th token
            i = tokens[:, prev_pos:cur_pos] # tensor([[   1,  306, 4658,  278, 6593,  310, 2834,  338]], device='cuda:0')
            logits = self.model(i, prev_pos) # NOTE important forward method, logits.shape=[1, 32000] 这是做了截取，只让self.model=<class 'llama.model_single.Transformer'> 返回最新一个预测出来的token
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token # tensor([[   1,  306, 4658,  278, 6593,  310, 2834,  338,  304,   -1,   -1,   -1, ...
            prev_pos = cur_pos
            
            if self._should_stop(tokens, prompt_tokens, stop_ids, stop_words):
                break

        tokens[tokens == self.tokenizer.pad_id] = self.tokenizer.eos_id
        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        #print(decoded)
        return [postprocessing(i, stop_words) for i in decoded]


def postprocessing(output_text, stop_words=None, threshold=10):
    sentences = output_text.split(".")
    filtered_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > threshold and sentence[-1] == ".":
            filtered_sentences.append(sentence)
    r = '.'.join(sentences).strip()
    if stop_words:
        for w in stop_words:
            if r.endswith(w):
                r = r[0:-len(w)].strip()
    if r[-1] != '.':
        r += '...'
    return r


def sample_top_p(probs, p): # probs.shape=[1, 32000], p=0.95
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True) # tensor([[8.6312e-01, 9.6636e-03, 8.2657e-03,  ..., 5.3256e-17, 1.4531e-17,; ||| tensor([[  304,  2560,   278,  ..., 24291, 16196, 27918]], device='cuda:0')
    probs_sum = torch.cumsum(probs_sort, dim=-1) # e.g., tensor([0.8631, 0.8728, 0.8810, 0.8888, 0.8962, 0.9033, 0.9101, 0.9161, 0.9210, ...
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True)) # tensor([0.9074, 0.0102, 0.0087, 0.0081, 0.0078, 0.0074, 0.0071, 0.0064, 0.0051,
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token # tensor([[304]], device='cuda:0')

