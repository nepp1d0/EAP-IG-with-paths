import torch
from transformer_lens import HookedTransformer
from jaxtyping import Float
from typing import List
from torch import Tensor
import torch.nn.functional as F

def compare_token_probability(clean_resid: Float[Tensor, "batch seq d_model"],
								corrupted_resid: Float[Tensor, "batch seq d_model"],
								model: HookedTransformer,
								target_tokens: List[int]) -> Float:
	""" Compute the difference of predicting the target token in probability
		between the clean and corrupted model based on the final residuals.
		args:
			clean_resid: torch.Tensor, shape (batch, seq_len, d_model)
				The final residual stream of the clean model.
			corrupted_resid: torch.Tensor, shape (batch, seq_len, d_model)
				The final residual stream of the corrupted model.
			model: HookedTransformer
				The hooked transformer model.
			target_tokens: list of int
				The indexes of the target tokens.
		returns:
			float
				The difference in probability of predicting the target token.
	"""
	# Get logits for the last token
	clean_resid = model.ln_final(clean_resid)
	corrupted_resid = model.ln_final(corrupted_resid)
	clean_logits = model.unembed(clean_resid)[:, -1, :]
	corrupted_logits = model.unembed(corrupted_resid)[:, -1, :]

	# Get the probability of the target token
	prob_clean = torch.Tensor([clean_logits[i].softmax(dim=-1)[target_tokens[i]] for i in range(len(target_tokens))])
	prob_corrupted = torch.Tensor([corrupted_logits[i].softmax(dim=-1)[target_tokens[i]] for i in range(len(target_tokens))])

	return torch.mean(100*(prob_clean - prob_corrupted)/prob_clean).item()

def compare_token_logit(clean_resid: Float[Tensor, "batch seq d_model"],
						corrupted_resid: Float[Tensor, "batch seq d_model"],
						model: HookedTransformer,
						target_tokens: List[int]) -> Float:
	""" Compute the difference of logits for the target token as a percentage
		between the clean and corrupted model based on the final residuals.
		This implementation is optimized for transformerlens HookedTransformer.
		args:
			clean_resid: torch.Tensor, shape (batch, seq_len, d_model)
				The final residual stream of the clean model.
			corrupted_resid: torch.Tensor, shape (batch, seq_len, d_model)
				The final residual stream of the corrupted model.
			model: HookedTransformer
				The hooked transformer model.
			target_tokens: list of int
				The indexes of the target tokens.
		returns:
			float
				The percentage difference in logits for the target token.
	"""
	# Get the unembedding weights and bias
	W_U = model.W_U
	b_U = model.b_U

	# Get the final residual stream for the last token
	clean_final_resid = clean_resid[:, -1, :]
	corrupted_final_resid = corrupted_resid[:, -1, :]
	
	# Apply the layer norm to the final residuals
	clean_final_resid = model.ln_final(clean_final_resid)
	corrupted_final_resid = model.ln_final(corrupted_final_resid)
	
	# Get the logits associated with the residuals
	clean_logits = torch.einsum('b d, d b-> b', clean_final_resid, W_U[:, target_tokens]) + b_U[target_tokens]
	corrupted_logits = torch.einsum('b d, d b-> b', corrupted_final_resid, W_U[:, target_tokens]) + b_U[target_tokens]
	# Calculate the percentage difference
	percentage_diffs = 100 * (clean_logits - corrupted_logits) / (torch.abs(clean_logits))
	return torch.mean(percentage_diffs).item()

# TODO: check if this is correct
def kl_divergence(clean_resid: Float[Tensor, "batch seq d_model"],
					corrupted_resid: Float[Tensor, "batch seq d_model"],
					model: HookedTransformer,
					target_token: int) -> Float:
	""" Compute the Kullback-Leibler divergence between the probability distributions
		of the clean and corrupted model based on the final residuals.
		args:
			clean_resid: torch.Tensor, shape (batch, seq_len, d_model)
				The final residual stream of the clean model.
			corrupted_resid: torch.Tensor, shape (batch, seq_len, d_model)
				The final residual stream of the corrupted model.
			model: HookedTransformer
				The hooked transformer model.
		returns:
			float
				The Kullback-Leibler divergence.
	"""
	# Get logits for the last token	
	clean_resid = model.ln_final(clean_resid)
	corrupted_resid = model.ln_final(corrupted_resid)
	clean_logits = model.unembed(clean_resid)[:, -1, :]
	corrupted_logits = model.unembed(corrupted_resid)[:, -1, :]

	# Convert logits to probability distributions using softmax
	prob_clean = F.softmax(clean_logits, dim=-1)
	prob_corrupted = F.softmax(corrupted_logits, dim=-1)

	# Compute KL divergence
	kl_div = F.kl_div(torch.log(prob_corrupted), prob_clean, reduction='batchmean')
	return kl_div.item()