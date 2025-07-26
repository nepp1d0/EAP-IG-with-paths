import torch
import sys
import transformers
import torch
import circuitsvis as cv
import torch.nn as nn
import numpy as np
import einops
from copy import deepcopy
from fancy_einsum import einsum
import transformer_lens.utils as utils
from transformer_lens import HookedTransformer, FactoredMatrix, HookedTransformerConfig
from jaxtyping import Float, Int
from torch import Tensor
import huggingface_hub
from tqdm import tqdm
import torch.nn.functional as F
from transformer_lens.ActivationCache import ActivationCache
import re


from utils.metrics import compare_token_probability, kl_divergence, compare_token_logit
from utils.miscellanea import get_top_k_contributors, IOI_head_types
from utils.component_contributions import contribution_mlp, contribution_attn


transformers.logging.set_verbosity_error()
# torch.set_default_dtype(torch.bfloat16)

from utils.nodes import MLP_Node, EMBED_Node, FINAL_Node,Node, ATTN_Node
from utils.graph_search import path_message, evaluate_path, breadth_first_search
import dotenv
import os
dotenv.load_dotenv()

TOKEN = os.getenv("TOKEN")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_grad_enabled(False)

huggingface_hub.login(token=TOKEN)

def main():
    # Note: Eventually can set set fold_ln=False, center_unembed=False, center_writing_weights=False
    model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE, torch_dtype=torch.float32)
    find_subject_inibition = False
    if find_subject_inibition:
        target_idx = 1
    else:
        target_idx = 0

    prompts = ['When John and Mary went to the shops, John gave the bag to', 'When John and Mary went to the shops, Mary gave the bag to', 'When Tom and James went to the park, Tom gave the ball to', 'When Tom and James went to the park, James gave the ball to', 'When Dan and Sid went to the shops, Dan gave an apple to', 'When Dan and Sid went to the shops, Sid gave an apple to', 'After Martin and Amy went to the park, Martin gave a drink to', 'After Martin and Amy went to the park, Amy gave a drink to']
    answers = [(' Mary', ' John'), (' John', ' Mary'), (' James', ' Tom'), (' Tom', ' James'), (' Sid', ' Dan'), (' Dan', ' Sid'), (' Amy', ' Martin'), (' Martin', ' Amy')]

    # Keep only the prompts where the second token is the indirect object
    # This is required because the search requires fixed input positions
    prompts_fixed_pos = prompts[0::2]
    answers_fixed_pos = answers[0::2]

    example_idx = 2

    tokens = model.to_tokens(prompts[example_idx])
    logits, cache = model.run_with_cache(prompts_fixed_pos)

    model_token = logits[0][-1].argmax(dim=-1)
    correct_tokens = [model.to_tokens(str(answers_fixed_pos[i][target_idx]))[0][-1].item() for i in range(len(prompts_fixed_pos))]
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    n_heads = model.cfg.n_heads
    d_heads = model.cfg.d_head

    default_metric = compare_token_logit

    min_treshold = 0.025 #0.25, #0.25, 2, 0.025

    complete_paths, incomplete_paths = breadth_first_search(
        model,
        cache,
        default_metric,
        start_node = [FINAL_Node(layer=model.cfg.n_layers-1, position=14)],
        ground_truth_tokens = correct_tokens,
        max_depth = 100, # max number of components in the path (max number of nodes -2)
        max_branching_factor = 2048,
        min_contribution = min_treshold,
        min_contribution_percentage=0., #2, 5, 0.5
        inibition_task = find_subject_inibition
    )
    print(f"Found {len(complete_paths)} complete paths and {len(incomplete_paths)} incomplete paths.")

    # save circuit
    import json
    from datetime import datetime

    # Convert the complete_paths to a serializable format
    def convert_path_to_dict(path_tuple):
        score, path = path_tuple
        path_dict = {
            "score": float(score),
            "nodes": []
        }
        
        for node in path:
            node_dict = {
                "type": node.__class__.__name__,
                "layer": node.layer,
                "position": node.position
            }
            
            # Add attention-specific attributes
            if hasattr(node, 'head'):
                node_dict["head"] = node.head
            if hasattr(node, 'keyvalue_position'):
                node_dict["keyvalue_position"] = node.keyvalue_position
            if hasattr(node, 'patch_query'):
                node_dict["patch_query"] = node.patch_query
            if hasattr(node, 'patch_keyvalue'):
                node_dict["patch_keyvalue"] = node.patch_keyvalue
                
            path_dict["nodes"].append(node_dict)
        
        return path_dict

    # Convert all paths
    serializable_paths = [convert_path_to_dict(path) for path in complete_paths]

    # Create metadata
    metadata = {
        "model": "gpt2-small",
        "prompt": prompts[example_idx],
        "correct_answer": str(answers[example_idx][0]),
        "target_idx": target_idx,
        "find_subject_inhibition": find_subject_inibition,
        "timestamp": datetime.now().isoformat(),
        "total_paths": len(complete_paths),
        "min_treshold": min_treshold,
        "n_layers": model.cfg.n_layers,
        "d_model": model.cfg.d_model,
        "n_heads": model.cfg.n_heads,
        "metric": default_metric.__name__
    }

    # Combine data
    output_data = {
        "metadata": metadata,
        "paths": serializable_paths
    }

    # Save to JSON file
    filename = f"detected_paths/detected_circuit_gpt2_ioi_{default_metric.__name__}_{min_treshold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved {len(complete_paths)} paths to {filename}")
    print(f"Top 3 paths by score:")
    for i, path in enumerate(serializable_paths[:3]):
        print(f"  {i+1}. Score: {path['score']:.4f}, Nodes: {len(path['nodes'])}")

if __name__ == "__main__":
    main()