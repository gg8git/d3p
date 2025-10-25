# TODO (faraz): move action tokenization logic into here when ready
# unify tokenization API with GPT-2 for easier data distribution
# comparison.


from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
from torch import nn
from dataclasses import dataclass, field
import math
import logging
from torchaudio.functional import create_dct

logger = logging.getLogger(__name__)

"""
TODO: clean up the eos and pad token logic, it seems that we do not need them since
we are primarily focusing on fixed-length tokenizers.
"""

class ActionTokenizer(ABC):
    """Abstract base class for action tokenizers used in discrete flow matching."""
    
    def __init__(
        self, 
        max_action_dim: int, 
        action_chunk_size: int,
        pad_token_id: int,
        eos_token_id: int,
        num_reserved_tokens: int
    ):
        self.max_action_dim         = max_action_dim
        self.action_chunk_size      = action_chunk_size
        self.pad_token_id           = pad_token_id
        self.eos_token_id           = eos_token_id
        self.num_reserved_tokens    = num_reserved_tokens 

    @abstractmethod
    def tokenize(self, actions: torch.Tensor) -> torch.LongTensor:
        """Convert continuous actions to discrete tokens."""
        pass
    
    @abstractmethod
    def detokenize(self, tokens: torch.LongTensor) -> torch.Tensor:
        """Convert discrete tokens back to continuous actions."""
        pass

class BinningActionTokenizer(ActionTokenizer):

    def __init__(self, max_action_dim: int, action_chunk_size: int, vocab_size: int, contiguous_dimensions: bool = False):
        super().__init__(
            max_action_dim      = max_action_dim, 
            action_chunk_size   = action_chunk_size,
            pad_token_id        = 0,
            eos_token_id        = 1,
            num_reserved_tokens = 2
        )
        self.vocab_size = vocab_size
        self.n_bins = vocab_size - 2 # reserve 0, and 1 for pad and eos respectively
        self.sequence_length = action_chunk_size * max_action_dim
        self.contiguous_dimensions = contiguous_dimensions

    def tokenize(self, actions: torch.Tensor) -> torch.LongTensor:
        """
        Uniformly map continuous actions in [-1,1] → integer tokens in [0, vocab-1].
        """
        assert torch.all((actions >= -1.0) & (actions <= 1.0)), "Actions must be in range [-1, 1]"
        tokens = (((actions + 1.0) / 2.0) * (self.n_bins - 1)).round()
        tokens = tokens.long().clamp_(0, self.vocab_size - 1) 
        if self.contiguous_dimensions:
            B, T, D = tokens.shape
            tokens = tokens.permute(0, 2, 1).contiguous().view(B, D * T)
        else:
            tokens = tokens.flatten(start_dim=1)
        return tokens + self.num_reserved_tokens

    def detokenize(self, tokens: torch.LongTensor) -> torch.Tensor:
        """
        Convert integer tokens back to continuous actions in [-1,1].
        """
        tokens = tokens - self.num_reserved_tokens
        B, S = tokens.shape
        assert S == self.sequence_length, "tokens must be of shape (B, self.sequence_length)"
        if self.contiguous_dimensions:
            tokens = tokens.view(B, self.max_action_dim, self.action_chunk_size) \
                        .permute(0, 2, 1) \
                        .contiguous()
        else:
            tokens = tokens.view(B, self.action_chunk_size, self.max_action_dim)
        return (tokens.float() / (self.n_bins - 1)) * 2.0 - 1.0


class DCTWrapper(ActionTokenizer):
    """
    Wrap any ActionTokenizer by DCT-transforming along the time axis (T),
    normalizing coefficients to [-1, 1], delegating to the base tokenizer,
    then denormalizing and inverse-DCT'ing on detokenize.

    Assumptions (simplified):
      - Wrapper INPUT  : actions ∈ [-1, 1], shape (B, T, D)
      - Wrapper OUTPUT : actions ∈ [-1, 1], shape (B, T, D)
      - Base tokenizer expects/returns continuous tensors in [-1, 1]
      - DCT is orthonormal (create_dct(..., norm='ortho'))
    """

    def __init__(self, action_tokenizer: ActionTokenizer, eps: float = 1e-6):
        self.action_tokenizer = action_tokenizer
        super().__init__(
            max_action_dim      = action_tokenizer.max_action_dim, 
            action_chunk_size   = action_tokenizer.action_chunk_size,
            pad_token_id        = action_tokenizer.pad_token_id,
            eos_token_id        = action_tokenizer.eos_token_id,
            num_reserved_tokens = action_tokenizer.num_reserved_tokens,
        )
        self.eps = float(eps)
        self._sqrtT = math.sqrt(self.action_chunk_size)
        self._dct_cache = {}  # (T, device, dtype) -> (M, MT)

    # ------------------------ public API ------------------------

    @property
    def sequence_length(self):
        return self.action_tokenizer.sequence_length

    @property
    def vocab_size(self):
        return self.action_tokenizer.vocab_size

    def tokenize(self, actions: torch.Tensor) -> torch.LongTensor:
        """
        actions: (B, T, D) in [-1, 1]
        returns: tokens from the wrapped tokenizer
        """
        print("WARNING: DCTWrapper has not been through and through tested yet, but all shapes match. Moons performance is questionable")
        assert actions.ndim == 3, "Expected (B, T, D)"
        B, T, D = actions.shape
        assert T == self.action_chunk_size and D == self.max_action_dim, "Shape mismatch"
        if torch.is_floating_point(actions):
            assert torch.all((actions >= -1.0) & (actions <= 1.0)), "Inputs must be in [-1, 1]"

        M, _ = self._get_dct_mats(actions.device, actions.dtype)  # (T, T)

        # Apply DCT along time axis: (B, D, T) @ (T, T) -> (B, D, T)
        coeffs = (actions.permute(0, 2, 1) @ M).permute(0, 2, 1)

        # Normalize coefficients to [-1, 1] using the conservative bound sqrt(T)
        coeffs_norm = torch.clamp(coeffs / (self._sqrtT + self.eps), -1.0, 1.0)

        # Delegate to base tokenizer (expects [-1, 1])
        return self.action_tokenizer.tokenize(coeffs_norm)

    def detokenize(self, tokens: torch.LongTensor) -> torch.Tensor:
        """
        tokens: output of the wrapped tokenizer
        returns: (B, T, D) in [-1, 1]
        """
        print("WARNING: DCTWrapper has not been through and through tested yet, but all shapes match. Moons performance is questionable")
        # Recover normalized DCT coefficients in [-1, 1]
        coeffs_norm = self.action_tokenizer.detokenize(tokens)  # (B, T, D) in [-1, 1]

        # Denormalize: [-1, 1] -> original coefficient scale
        coeffs = coeffs_norm * self._sqrtT

        # Inverse DCT along time axis using M^T for orthonormal case
        _, MT = self._get_dct_mats(coeffs.device, coeffs.dtype)  # (T, T)
        actions = (coeffs.permute(0, 2, 1) @ MT).permute(0, 2, 1)

        return torch.clamp(actions, -1.0, 1.0)

    # ------------------------ helpers ------------------------

    def _get_dct_mats(self, device, dtype):
        """
        Returns (M, MT) where:
          - M  is the orthonormal DCT-II transform matrix (T, T)
          - MT is its transpose (the inverse for orthonormal DCT-II)
        """
        key = (self.action_chunk_size, device, dtype)
        if key in self._dct_cache:
            return self._dct_cache[key]

        T = self.action_chunk_size
        M = create_dct(n_mfcc=T, n_mels=T, norm="ortho").to(device=device, dtype=dtype).contiguous()
        MT = M.t().contiguous()
        self._dct_cache[key] = (M, MT)
        return M, MT

class KMeansDynaGripperWrapper(ActionTokenizer):
    def __init__(self, centroids: int, action_tokenizer: ActionTokenizer):
        self.action_tokenizer = action_tokenizer
        self.vocab_size = self.action_tokenizer.vocab_size
        self.sequence_length = self.action_tokenizer.sequence_length + 2 # for the additional gripper tokens

        super().__init__(
            max_action_dim      = self.action_tokenizer.max_action_dim + 2, 
            action_chunk_size   = self.action_tokenizer.action_chunk_size,
            pad_token_id        = 0,
            eos_token_id        = 1,
            num_reserved_tokens = 2
        )
        self.centroids = centroids
        assert self.centroids.shape == torch.Size((self.action_tokenizer.n_bins, self.action_chunk_size)), """
        Centroid must have same number as centroids as bins in BinningActionTokenizer. Recall that 
        n_bins is vocab_size-2 because of reserved pad and eos token.
        """

        assert self.action_tokenizer.max_action_dim == self.max_action_dim - 2, f"""
        KMeansDynaGripper will remove exactly two dimensions from the action dimension, 
        so the wrapped tokenizer must have an action dim of two less. Recieved 
        self.max_action_dim:                    {self.max_action_dim}
        self.action_tokenizer.max_action_dim:   {self.action_tokenizer.max_action_dim}
        """

        logging.info(f"""
        Wrapping action tokenizer: {action_tokenizer} w/ KMeansDynaGripperWrapper.
        Gripper dimensions will be tokenized using KMeans clustering
        """)

    def _tokenize_gripper(self, gripper_actions: torch.Tensor) -> torch.Tensor:
        B, T, D = gripper_actions.shape
        gripper_actions = gripper_actions.permute(0, 1, 2).contiguous() # B, D, T
        gripper_actions = gripper_actions.view(B * D, T)

        # Calculate the L2 norm between each gripper action and the centroids
        distances = torch.cdist(gripper_actions, self.centroids, p=2)
        nearest_centroid_indices = torch.argmin(distances, dim=1)
        nearest_centroid_indices = nearest_centroid_indices.view(B, D)
        return nearest_centroid_indices

    def _detokenize_gripper(self, gripper_tokens: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        self.centroids = self.centroids.to(device = gripper_tokens.device, dtype=dtype)
        gripper_actions = self.centroids[gripper_tokens]
        return gripper_actions.view(-1, 2, 32).permute(0, 2, 1).contiguous()

    def tokenize(self, actions: torch.Tensor) -> torch.LongTensor:
        gripper_actions = actions[:, :, [9, 19]].contiguous()
        no_gripper_actions = actions[:, :, [i for i in range(actions.shape[-1]) if i not in (9, 19)]].contiguous()

        no_gripper_tokens = self.action_tokenizer.tokenize(no_gripper_actions)
        gripper_token = self._tokenize_gripper(gripper_actions)
        tokens = torch.cat((no_gripper_tokens, gripper_token), dim=-1)
        return tokens

    def detokenize(self, tokens: torch.LongTensor) -> torch.LongTensor:
        gripper_tokens = tokens[:, -2:]
        no_gripper_tokens = tokens[:, :-2]

        no_gripper_actions = self.action_tokenizer.detokenize(no_gripper_tokens)
        gripper_actions = self._detokenize_gripper(gripper_tokens, dtype=no_gripper_actions.dtype)
        
        # Reconstruct the full action tensor by inserting gripper actions back into their original positions
        full_actions = torch.zeros((no_gripper_actions.shape[0], no_gripper_actions.shape[1], no_gripper_actions.shape[2] + 2), device=no_gripper_actions.device)
        full_actions[:, :, [i for i in range(full_actions.shape[-1]) if i not in (9, 19)]] = no_gripper_actions
        full_actions[:, :, 9] = gripper_actions[:, :, 0]
        full_actions[:, :, 19] = gripper_actions[:, :, 1]

        return full_actions

class RemovePaddingActionTokenizerWrapper(ActionTokenizer):
    def __init__(self, max_action_dim: int, action_tokenizer: ActionTokenizer):
        self.action_tokenizer = action_tokenizer
        self.vocab_size = self.action_tokenizer.vocab_size
        self.sequence_length = self.action_tokenizer.sequence_length

        super().__init__(
            max_action_dim      = max_action_dim, 
            action_chunk_size   = self.action_tokenizer.action_chunk_size,
            pad_token_id        = 0,
            eos_token_id        = 1,
            num_reserved_tokens = 2
        )

        assert self.action_tokenizer.max_action_dim < self.max_action_dim

        logging.info(f"""
        Wrapping action tokenizer: {action_tokenizer} w/ RemovePaddingActionTokenizerWrapper.
        Actions will be reduced to {self.action_tokenizer.max_action_dim} before being
        tokenized.
        """)
            
    def tokenize(self, actions: torch.Tensor) -> torch.LongTensor:
        real_actions = actions[:, :, :self.action_tokenizer.max_action_dim]
        tokens = self.action_tokenizer.tokenize(real_actions)
        return tokens

    def detokenize(self, tokens: torch.LongTensor) -> torch.LongTensor:
        unpadded_actions = self.action_tokenizer.detokenize(tokens)
        B, T, D = unpadded_actions.shape
        padded_actions = torch.zeros((B, T, self.max_action_dim), device=unpadded_actions.device)
        padded_actions[:, :, :D] = unpadded_actions
        return padded_actions

class NormalizedTokenizerWrapper(ActionTokenizer):

    def __init__(self, action_tokenizer: ActionTokenizer, action_stats: "ActionChunkStats"):
        super().__init__(
            max_action_dim      = action_tokenizer.max_action_dim, 
            action_chunk_size   = action_tokenizer.action_chunk_size,
            pad_token_id        = action_tokenizer.pad_token_id,
            eos_token_id        = action_tokenizer.eos_token_id,
            num_reserved_tokens = action_tokenizer.num_reserved_tokens
        )

        self.action_tokenizer = action_tokenizer
        self.vocab_size = action_tokenizer.vocab_size
        self.sequence_length = action_tokenizer.sequence_length
        self.mins = action_stats.mins
        self.maxs = action_stats.maxs

        # Assert that the shapes of the saved tensors match with the shapes of the passed in tokenizer
        assert self.mins.shape[-1] >= self.max_action_dim, "Mismatch in mins shape and max_action_dim"
        assert self.maxs.shape[-1] >= self.max_action_dim, "Mismatch in maxs shape and max_action_dim"
        assert self.mins.shape[-2] >= self.action_chunk_size, "Mismatch in mins shape and action_chunk_size"
        assert self.maxs.shape[-2] >= self.action_chunk_size, "Mismatch in mins shape and action_chunk_size"

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        B, T, D = actions.shape
        self.mins = self.mins.to(actions.device)
        self.maxs = self.maxs.to(actions.device)
        range_values = self.maxs[:T,:D] - self.mins[:T,:D]
        range_values[range_values == 0] = 1  # Avoid division by zero
        normalized_actions = 2 * (actions - self.mins[:T,:D]) / range_values - 1
        clamped_normalized_actions = normalized_actions.clamp(-1.05, 1.05)
        num_clamped = (clamped_normalized_actions != normalized_actions).sum().item()
        total_items = normalized_actions.numel()
        percent_clamped = (num_clamped / total_items) * 100
        if num_clamped > 0:
            print(f"Warning: Percent of clamped items in tokenizer.normalize() outside of -1.05, 1.05: {percent_clamped}%")
        return clamped_normalized_actions.clamp(-1.0, 1.0)
        
    def unnormalize(self, actions: torch.LongTensor) -> torch.Tensor:
        B, T, D = actions.shape
        self.mins = self.mins.to(actions.device)
        self.maxs = self.maxs.to(actions.device)
        range_values = self.maxs[:T,:D] - self.mins[:T,:D]
        range_values[range_values == 0] = 1  # Avoid division by zero
        actions = (actions + 1) / 2 * range_values + self.mins[:T,:D]
        clamped_actions = actions.clamp(-1.05, 1.05)
        num_clamped = (clamped_actions != actions).sum().item()
        total_items = actions.numel()
        percent_clamped = (num_clamped / total_items) * 100
        if num_clamped > 0:
            print(f"Warning: Percent of clamped items in tokenizer.unnormalize() outside of -1.05, 1.05: {percent_clamped}%")
        return clamped_actions.clamp(-1.0, 1.0)

    def tokenize(self, actions: torch.Tensor) -> torch.LongTensor:
        normalized_actions = self.normalize(actions)
        return self.action_tokenizer.tokenize(normalized_actions)

    def detokenize(self, tokens: torch.LongTensor) -> torch.Tensor:
        actions = self.action_tokenizer.detokenize(tokens)
        return self.unnormalize(actions)

class LIBERO90NormalizedBinningActionTokenizer(ActionTokenizer):

    def __init__(self, max_action_dim: int, action_chunk_size: int, vocab_size: int, contiguous_dimensions: bool = False):
        super().__init__(
            max_action_dim      = max_action_dim, 
            action_chunk_size   = action_chunk_size,
            pad_token_id        = 0,
            eos_token_id        = 1,
            num_reserved_tokens = 2
        )

        self.binning_action_tokenizer = BinningActionTokenizer(max_action_dim, action_chunk_size, vocab_size, contiguous_dimensions)
        self.vocab_size = self.binning_action_tokenizer.vocab_size
        self.sequence_length = self.binning_action_tokenizer.sequence_length
        self.maxs = torch.Tensor([
            0.3125, 
            0.3125,
            0.3125,
            0.125,
            0.125,
            0.125,
            1.0,
            *([0]*32)
        ])

        self.mins = -self.maxs

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        D = actions.shape[-1]
        self.mins = self.mins.to(actions.device)
        self.maxs = self.maxs.to(actions.device)
        normalized_actions = 2 * (actions - self.mins[:D]) / (self.maxs[:D] - self.mins[:D]) - 1
        return normalized_actions.clamp(-1, 1)
        
    def unnnormalize(self, tokens: torch.LongTensor) -> torch.Tensor:
        self.mins = self.mins.to(tokens.device)
        self.maxs = self.maxs.to(tokens.device)
        D = tokens.shape[-1]
        actions = (tokens + 1) / 2 * (self.maxs[:D] - self.mins[:D]) + self.mins[:D]
        return actions.clamp(-1, 1)

    def tokenize(self, actions: torch.Tensor) -> torch.LongTensor:
        normalized_actions = self.normalize(actions)
        return self.binning_action_tokenizer.tokenize(normalized_actions)

    def detokenize(self, tokens: torch.LongTensor) -> torch.Tensor:
        actions = self.binning_action_tokenizer.detokenize(tokens)
        return self.unnnormalize(actions)

@dataclass
class ActionChunkStats:
    mins: torch.Tensor
    maxs: torch.Tensor
    means: torch.Tensor
    std: torch.Tensor

# green_towel_stats: ActionChunkStats = torch.load("/mnt/dyna_users/faraz/stash/action_stats_cache_green-towel_n10000.pt", weights_only=False)


# libero90_statistics = ActionChunkStats(
#     means = torch.Tensor([[ 0.0132,  0.0115, -0.0296,  0.0018,  0.0002, -0.0020, -0.0185] + [0.0] * 25 ] * 32),
#     # std = torch.Tensor([[0.0981, 0.1206, 0.1376, 0.0160, 0.0186, 0.0294, 0.9815]] * 32)
#     std = torch.Tensor([[0.1800, 0.1800, 0.1800, 0.04, 0.040, 0.0550, 0.9815] + [1.0] * 25] * 32)
# )

class QuantileTransformedTokenizerWrapper(ActionTokenizer):

    def __init__(self, action_statistics: ActionChunkStats, action_tokenizer: ActionTokenizer):
        self.action_tokenizer = action_tokenizer
        self.vocab_size = self.action_tokenizer.vocab_size
        self.sequence_length = self.action_tokenizer.sequence_length

        super().__init__(
            max_action_dim      = self.action_tokenizer.max_action_dim, 
            action_chunk_size   = self.action_tokenizer.action_chunk_size,
            pad_token_id        = 0,
            eos_token_id        = 1,
            num_reserved_tokens = 2
        )

        self.action_statistics = action_statistics

        logging.info(f"""
        Wrapping action tokenizer: {action_tokenizer} w/ QuantileTransformedTokenizerWrapper.
        """)

    def quantile_transform(self, actions: torch.Tensor) -> torch.Tensor:
        clip = 3.0
        mean = self.action_statistics.means.to(actions.device)[None, :, :self.max_action_dim]
        std = self.action_statistics.std.to(actions.device)[None, :, :self.max_action_dim]
        z = (actions - mean) / std # broadcast batch dimension + standardise
        z = z.clamp(-clip, clip)                       # clip actions to be within .05 to .95

        # Φ(z) = 0.5·(1 + erf(z / √2))  ––> Uniform(0, 1)
        u = (torch.erf(z / math.sqrt(2.0)))
        return u 
    
    def inverse_quantile_transform(self, actions: torch.Tensor) -> torch.Tensor:
        # Φ⁻¹(u) = √2 · erfinv(2u − 1)
        mean = self.action_statistics.means.to(actions.device)[None, :, :self.max_action_dim]
        std = self.action_statistics.std.to(actions.device)[None, :, :self.max_action_dim]

        # Clamp to the open interval (−1, 1) before applying erfinv.
        # We keep the clamp extremely small (ε = 1e-6) to avoid altering valid
        # values, but we still detect if we had to clamp by more than 0.005 and
        # raise a warning so the caller is aware of potentially bad inputs.
        eps = 1e-6
        clamped_actions = actions.clamp(-1.0 + eps, 1.0 - eps)
        max_adjustment = (clamped_actions - actions).abs().max()
        if max_adjustment > 0.005:
            logging.warning(
                f"inverse_quantile_transform: clamped values by up to {max_adjustment.item():.6f}. "
                f"Inputs may be outside the valid (-1, 1) range."
            )

        z = math.sqrt(2.0) * torch.erfinv(clamped_actions)
        x = z * std + mean
        return x

    def tokenize(self, actions: torch.Tensor) -> torch.LongTensor:
        quantiled_actions = self.quantile_transform(actions)
        tokens = self.action_tokenizer.tokenize(quantiled_actions)
        return tokens

    def detokenize(self, tokens: torch.LongTensor) -> torch.Tensor:
        quantiled_actions = self.action_tokenizer.detokenize(tokens)
        actions = self.inverse_quantile_transform(quantiled_actions)
        return actions

class VQVAETokenizer(ActionTokenizer):

    def __init__(self, checkpoint_path, vqvae_head = None):
        from ml_models.dynamism.src.model.vla.head import VQVAEHead
        if vqvae_head is not None: 
            # allow direct passing of VQVAEHead
            self.vqvae_head = vqvae_head 
        else:
            # use checkpoint init
            self.vqvae_head = VQVAEHead.load_from_checkpoint(checkpoint_path)

        super().__init__(
            max_action_dim      = self.vqvae_head.action_dim, 
            action_chunk_size   = self.vqvae_head.action_chunk_size,
            pad_token_id        = 0,
            eos_token_id        = 1,
            num_reserved_tokens = 2
        )
        self.vocab_size = self.vqvae_head.codebook_size + self.num_reserved_tokens
        self.sequence_length = self.vqvae_head.sequence_length # HACK (faraz): hard code for now

    def tokenize(self, actions: torch.Tensor) -> torch.LongTensor:
        tokens = self.vqvae_head.tokenize(actions)
        return tokens
    
    def detokenize(self, tokens: torch.LongTensor) -> torch.Tensor:
        self.vqvae_head.to(tokens.device)
        actions = self.vqvae_head.detokenize(tokens)
        return actions
 

class TokenObserver:

    def __init__(self, sequence_length, vocab_size):

        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        
        self.reset_observations()

    def reset_observations(self):
        self._codebook_histogram_accumulator: torch.LongTensor = torch.zeros(
            self.sequence_length, 
            self.vocab_size
        )

    def observe(self, tokens: torch.LongTensor) -> torch.Tensor:
        B, S = tokens.shape
        one_hot_tokens = torch.nn.functional.one_hot(tokens, num_classes=self.vocab_size)
        position_histogram = one_hot_tokens.sum(dim=0)
        self._codebook_histogram_accumulator = self._codebook_histogram_accumulator.to(position_histogram.device)
        self._codebook_histogram_accumulator += position_histogram

    def plot_observations(self, output_path, title):
        from ml_models.dynamism.src.utils.visualization import plot_tensor_heatmap
        token_heatmap = self.codebook_histogram
        return plot_tensor_heatmap(token_heatmap.T, title, output_path, log_scale=True)

    @property
    def codebook_histogram(self) -> torch.Tensor:
        num_observed = self._codebook_histogram_accumulator[0].sum()
        print("number of token sequences observed: ", num_observed)
        return self._codebook_histogram_accumulator / num_observed

def get_tokenizer(tokenizer_name, vocab_size, max_action_dim, action_chunk_size, **kwargs) -> ActionTokenizer:
    if tokenizer_name == "binning":
        return BinningActionTokenizer(
            max_action_dim=max_action_dim,
            action_chunk_size=action_chunk_size,
            vocab_size=vocab_size,
        ) 
    if tokenizer_name == "ctgs-binning":
        return BinningActionTokenizer(
            max_action_dim=max_action_dim,
            action_chunk_size=action_chunk_size,
            vocab_size=vocab_size,
            contiguous_dimensions=True
        )
    if tokenizer_name == "dct-binning":
        bin_tok = BinningActionTokenizer(
            max_action_dim=max_action_dim,
            action_chunk_size=action_chunk_size,
            vocab_size=vocab_size,
            contiguous_dimensions=False
        )
        return DCTWrapper(
            action_tokenizer=bin_tok
        )
    elif tokenizer_name == "FAST":
        tok = FASTTokenizer(
            max_action_dim=max_action_dim,
            action_chunk_size=action_chunk_size,
        )
        assert vocab_size == tok.vocab_size, f"Passed DFM vocab size {vocab_size} does not match value required of the FAST tokenizer {tok.vocab_size}"
        return tok
    elif tokenizer_name == "KMeansDynaGripper":
        bin_tok = BinningActionTokenizer(
            max_action_dim = max_action_dim - 2,
            action_chunk_size = action_chunk_size,
            vocab_size = vocab_size
        )
        assert vocab_size == 1024
        kmeans_grip_tok = KMeansDynaGripperWrapper(
            centroids = torch.load("/mnt/dyna_users/faraz/dyna/research_tools/action_tokenization/centroids_n1022.pt"),
            action_tokenizer = bin_tok
        )
        return kmeans_grip_tok
    elif "GreenTowelNormalizedBinning" in tokenizer_name:
        bin_tok =  BinningActionTokenizer(
            max_action_dim=max_action_dim,
            action_chunk_size=action_chunk_size,
            vocab_size=vocab_size,
            contiguous_dimensions="ctgs" in tokenizer_name
        )
        return NormalizedTokenizerWrapper(
            bin_tok,
            green_towel_stats
        )
    elif tokenizer_name == "Libero90NormalizedBinning":
        return LIBERO90NormalizedBinningActionTokenizer(
            max_action_dim=max_action_dim,
            action_chunk_size=action_chunk_size,
            vocab_size=vocab_size,
        )
    elif tokenizer_name == "LiberoNoPadBinning":
        binning_tok = BinningActionTokenizer(
            max_action_dim=7, # Wrapper below will remove padding.
            action_chunk_size=action_chunk_size,
            vocab_size=vocab_size,
        ) 
        return RemovePaddingActionTokenizerWrapper(
            max_action_dim=max_action_dim,
            action_tokenizer=binning_tok
        )
    elif tokenizer_name == "LiberoQuantileBins":
        binning_tok = BinningActionTokenizer(
            max_action_dim=max_action_dim, # Wrapper below will remove padding.
            action_chunk_size=action_chunk_size,
            vocab_size=vocab_size,
        ) 
        quantile_tok = QuantileTransformedTokenizerWrapper(
            action_statistics=libero90_statistics,
            action_tokenizer=binning_tok
        )
        return quantile_tok
    elif tokenizer_name == "LiberoNoPadQuantileBins":
        binning_tok = BinningActionTokenizer(
            max_action_dim=7, # Wrapper below will remove padding.
            action_chunk_size=action_chunk_size,
            vocab_size=vocab_size,
        ) 
        quantiled_tok = QuantileTransformedTokenizerWrapper(
            action_statistics=libero90_statistics,
            action_tokenizer=binning_tok
        )
        return RemovePaddingActionTokenizerWrapper(
            max_action_dim=max_action_dim,
            action_tokenizer=binning_tok
        )
    elif tokenizer_name == "VQVAE":
        tok = VQVAETokenizer(
            kwargs["tokenizer_path"],
        )
        assert vocab_size == tok.vocab_size, f"Passed DFM vocab size {vocab_size} does not match value required of the VQVAE tokenizer {tok.vocab_size}"
        return tok
    else: 
        raise Exception(f"tokenizer_name: {tokenizer_name} not found.")
