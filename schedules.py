import math
from typing import Any, Callable, Optional, Sequence, Union, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)

def NEG_INF_VALUE(dtype: torch.dtype):
    return torch.finfo(dtype).min

class D3MSchedule(ABC):

    @abstractmethod
    def noise_tokens(self, action_tokens, noise_tokens, t):
        raise NotImplementedError("Subclass must implement this method")

    @abstractmethod
    def step(self, prev_action_tokens, logits, t):
        raise NotImplementedError("Subclass must implement this method")

    @staticmethod
    def _sample_softmax_logits(softmax_logits, use_argmax):
        # TODO (faraz): unify this using temperature sampling
        if use_argmax:
            return torch.argmax(softmax_logits, dim=-1)
        else: 
            flattened_logits = softmax_logits.reshape(-1, softmax_logits.size(-1))
            sampled_logits   = torch.multinomial(flattened_logits, 1).squeeze(-1)
            reshaped_logits  = sampled_logits.reshape(softmax_logits.size()[:-1])
            return reshaped_logits

class VanillaD3MSchedule(D3MSchedule):

    def __init__(self, num_inference_steps: int, argmax_posterior: bool):
        super().__init__()
        self.argmax_posterior    = argmax_posterior
        self.num_inference_steps = num_inference_steps

    def noise_tokens(self, action_tokens, noise_tokens, t):
        # liner probability annealing
        prob_noise        = 1 - t # e.g. t=0 -> 100% noise, t=.75 -> 25% noise
        flip_mask         = torch.rand(action_tokens.shape, device=action_tokens.device) < prob_noise[:, None]
        noised_tokens     = torch.where(flip_mask, noise_tokens, action_tokens)
        
        return noised_tokens

    def step(self, prev_action_tokens, logits, t):
        # liner probability annealing
        prob_flip         = t + (1 / self.num_inference_steps) # e.g. for 1 step [1.0] for 10 step [0.1, 0.2, ... 0.9, 1.0]
        sampled_posterior = self._sample_softmax_logits(logits, self.argmax_posterior)
        flip_mask         = torch.rand(prev_action_tokens.shape, device=prev_action_tokens.device) < prob_flip[:, None]
        new_action_tokens = torch.where(flip_mask, sampled_posterior, prev_action_tokens)

        return new_action_tokens, t + 1 / self.num_inference_steps

class TwoBlockD3MSchedule(D3MSchedule):

    def __init__(self, num_inference_steps: int, argmax_posterior: bool):
        super().__init__()
        self.argmax_posterior    = argmax_posterior
        assert num_inference_steps == 2, "TwoBlockD3MSchedule requires num_inference_steps == 2"

    def noise_tokens(self, action_tokens, noise_tokens, t):
        """
        Vectorised (torch.compile-friendly) implementation.

        For each sample in the batch:
        • if t < 0.5  ⟶ replace the whole sequence with `noise_tokens`
        • else        ⟶ replace only the second half
        """
        B, S = action_tokens.shape
        half = S // 2
        # Masks
        mask_all        = (t < 0.5).unsqueeze(1).expand(-1, S)                    # (B, S)
        idx             = torch.arange(S, device=action_tokens.device)
        right_half_mask = idx >= half                                       # (S,)
        mask_half       = (t >= 0.5).unsqueeze(1).expand(-1, S) & right_half_mask

        mask = mask_all | mask_half                                         # (B, S)
        return torch.where(mask, noise_tokens, action_tokens)

    def step(self, prev_action_tokens, logits, t):
        """
        Stage-wise update with pure tensor ops (safe for torch.compile).
        • stage-0 (t == 0)   → fill first half
        • stage-1 (t == 0.5) → fill second half
        """
        S = prev_action_tokens.size(1)
        half = S // 2

        sampled = self._sample_softmax_logits(logits, self.argmax_posterior)
        left_update  = (t == 0).unsqueeze(1).expand(-1, half)   # (B, half)
        right_update = (t != 0).unsqueeze(1).expand(-1, half)   # (B, half)
        mask = torch.cat([left_update, right_update], dim=1)    # (B, S)

        new_action_tokens = torch.where(mask, sampled, prev_action_tokens)
        new_t = torch.where(t == 0, torch.full_like(t, 0.5), torch.full_like(t, 1.0))
        return new_action_tokens, new_t


class GreedyD3MSchedule(D3MSchedule):
    """
    This sampling strategy was explored in https://arxiv.org/abs/2502.06768
    """

    def __init__(self, num_inference_steps, pad_token_id):
        super().__init__()
        self.num_inference_steps = num_inference_steps
        self.pad_token_id = pad_token_id

    def noise_tokens(self, action_tokens, noise_tokens, t):
        # same as vanilla
        prob_noise        = 1 - t # e.g. t=0 -> 100% noise, t=.75 -> 25% noise
        flip_mask         = torch.rand(action_tokens.shape, device=action_tokens.device) < prob_noise[:, None]
        noised_tokens     = torch.where(flip_mask, noise_tokens, action_tokens)
        
        return noised_tokens

    def _get_step_size(self, prev_action_tokens, logits, t):
        assert prev_action_tokens.shape[1] % self.num_inference_steps == 0, """
        There may be unexpected behavior when num_inference_steps does not 
        divide token sequence_length evenly. Double check on this.
        """
        return  ((t*0)+1) * logits.shape[1] / self.num_inference_steps  # e.g. for 1 step [1.0] for 10 step [0.1, 0.2, ... 0.9, 1.0]

    def step(self, prev_action_tokens, logits, t):
        # NOTE (faraz): this logic prevents ever remasking - for now this is desirable.
        logits[:, :, self.pad_token_id] = NEG_INF_VALUE(logits.dtype) # ensures that mask token is never sampled
        num_tokens_to_unmask = self._get_step_size(prev_action_tokens, logits, t)

        # Mask logits for already unmasked tokens so that they are not picked by greedy sampler
        masked_logits = logits.clone()
        masked_logits[prev_action_tokens != self.pad_token_id] = NEG_INF_VALUE(masked_logits.dtype)

        new_action_tokens = prev_action_tokens.clone()

        for i in range(logits.size(0)):
            max_values, argmax_indices = torch.max(masked_logits[i], dim=-1)
            k = int(num_tokens_to_unmask[i].item())
            if k == 0:
                continue  # Nothing to unmask for this sample in this step

            topk_indices = torch.topk(max_values, k=k, dim=-1).indices

            new_action_tokens[i, topk_indices] = argmax_indices[topk_indices]

            masked_logits[i, topk_indices] = NEG_INF_VALUE(masked_logits.dtype)

        t = t + (1 / self.num_inference_steps)
        return new_action_tokens, t

def get_schedule(schedule_name: str, num_inference_steps: int, pad_token_id: int, **kwargs) -> D3MSchedule:
    if schedule_name == "vanilla":
        return VanillaD3MSchedule(
            num_inference_steps,
            False
        )
    elif schedule_name == "vanilla-argmax":
        return VanillaD3MSchedule(
            num_inference_steps,
            True
        )
    elif schedule_name == "greedy":
        return GreedyD3MSchedule(
            num_inference_steps, 
            pad_token_id
        )
    elif schedule_name == "TwoBlockD3M":
        return TwoBlockD3MSchedule(
            num_inference_steps,
            False
        )
    else:
        raise KeyError(f"Schedule name '{schedule_name}' not found.")



def generate_discrete_noise(
        discrete_noise_type, 
        bsz, 
        action_chunk_size,
        max_action_dim,
        action_tokenizer,
        device,
    ):
    S, V = action_tokenizer.sequence_length, action_tokenizer.vocab_size
    if discrete_noise_type == "sequence":
        assert S <= V # NOTE (faraz): this is for embedder vocab size reasons
        return torch.arange(S, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
    elif discrete_noise_type == "zeros":
        return torch.zeros((bsz, S), device=device, dtype=torch.long)
    elif discrete_noise_type == "uniform":
        return torch.randint(0, V, (bsz, S), device=device, dtype=torch.long)
    elif discrete_noise_type == "tokenized-gaussian":
        gaussian            = torch.randn(bsz, action_chunk_size, max_action_dim, device=device) # Gaussian will not be normalized [-1, 1], but our tokenizer will yell at us if it's out of the range
        # We can use the regularize using this crude approx. of Expected minimum value
        n                   = action_chunk_size * max_action_dim
        approx_exp_min      = math.sqrt(2 * math.log(n))
        normalized          = (gaussian / approx_exp_min)
        clamped             = torch.clamp(normalized, min=-1.0, max=1.0)
        return action_tokenizer.tokenize(clamped)
    else:
        raise ValueError(f"Unexpected discrete_noise_type: {discrete_noise_type}")

