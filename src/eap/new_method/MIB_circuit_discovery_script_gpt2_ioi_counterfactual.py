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
from typing import List, Optional


from utils.metrics import logit_difference_counterfactual
from utils.miscellanea import get_top_k_contributors, IOI_head_types
from utils.component_contributions import contribution_mlp, contribution_attn


transformers.logging.set_verbosity_error()
# torch.set_default_dtype(torch.bfloat16)

from utils.nodes import MLP_Node, EMBED_Node, FINAL_Node,Node, ATTN_Node
from utils.graph_search import breadth_first_search_with_counterfactual
import dotenv
import os
dotenv.load_dotenv()

TOKEN = os.getenv("TOKEN")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_grad_enabled(False)

huggingface_hub.login(token=TOKEN)

def create_ioi_counterfactual_prompts():
    """
    Create IOI prompts with their counterfactual versions.
    Returns: (clean_prompts, counterfactual_prompts, clean_targets, counterfactual_targets)
    """
    # Original IOI prompts and answers
    prompts = [
        'When John and Mary went to the shops, John gave the bag to',
        'When John and Mary went to the shops, Mary gave the bag to', 
        'When Tom and James went to the park, Tom gave the ball to',
        'When Tom and James went to the park, James gave the ball to',
        'When Dan and Sid went to the shops, Dan gave an apple to',
        'When Dan and Sid went to the shops, Sid gave an apple to',
        'After Martin and Amy went to the park, Martin gave a drink to',
        'After Martin and Amy went to the park, Amy gave a drink to'
    ]
    
    # Answers: (indirect_object, subject) for each prompt
    answers = [
        (' Mary', ' John'), (' John', ' Mary'), 
        (' James', ' Tom'), (' Tom', ' James'), 
        (' Sid', ' Dan'), (' Dan', ' Sid'), 
        (' Amy', ' Martin'), (' Martin', ' Amy')
    ]
    
    # Keep only prompts where the second token is the indirect object (fixed positions)
    # This matches the original script's filtering
    prompts_fixed_pos = prompts[0::2]  # [0, 2, 4, 6]
    answers_fixed_pos = answers[0::2]   # [(Mary, John), (James, Tom), (Sid, Dan), (Amy, Martin)]
    
    # Create counterfactual prompts by swapping subject and indirect object
    counterfactual_prompts = []
    for i, (prompt, (indirect_obj, subject)) in enumerate(zip(prompts_fixed_pos, answers_fixed_pos)):
        # Create counterfactual by swapping the names in the prompt
        # Original: "When John and Mary went to the shops, John gave the bag to"
        # Counterfactual: "When Mary and John went to the shops, Mary gave the bag to"
        
        # Simple string replacement for the names
        counterfactual_prompt = prompt.replace(subject, "TEMP_SUBJECT").replace(indirect_obj, subject).replace("TEMP_SUBJECT", indirect_obj)
        counterfactual_prompts.append(counterfactual_prompt)
    
    return prompts_fixed_pos, counterfactual_prompts, answers_fixed_pos

def main():
    # Note: Eventually can set set fold_ln=False, center_unembed=False, center_writing_weights=False
    model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE, torch_dtype=torch.float32)
    
    # Configuration
    find_subject_inibition = False
    if find_subject_inibition:
        target_idx = 1  # Subject token
    else:
        target_idx = 0  # Indirect object token

    # Create prompts and counterfactuals
    clean_prompts, counterfactual_prompts, answers = create_ioi_counterfactual_prompts()
    
    # Select example (same as original script)
    example_idx = 2
    clean_prompt = clean_prompts[example_idx]
    counterfactual_prompt = counterfactual_prompts[example_idx]
    
    print(f"Clean prompt: {clean_prompt}")
    print(f"Counterfactual prompt: {counterfactual_prompt}")
    print(f"Clean answer: {answers[example_idx][target_idx]}")
    print(f"Counterfactual answer: {answers[example_idx][1-target_idx]}")  # Opposite target
    
    # Tokenize prompts
    clean_tokens = model.to_tokens(clean_prompt)
    counterfactual_tokens = model.to_tokens(counterfactual_prompt)
    
    # Run model with both prompts
    clean_logits, clean_cache = model.run_with_cache(clean_prompt)
    counterfactual_logits, counterfactual_cache = model.run_with_cache(counterfactual_prompt)
    
    # Get target tokens
    clean_target_token = model.to_tokens(str(answers[example_idx][target_idx]))[0][-1].item()
    counterfactual_target_token = model.to_tokens(str(answers[example_idx][1-target_idx]))[0][-1].item()
    
    print(f"Clean target token: {clean_target_token}")
    print(f"Counterfactual target token: {counterfactual_target_token}")
    
    # Model configuration
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    n_heads = model.cfg.n_heads
    d_heads = model.cfg.d_head

    # Use the new counterfactual metric
    default_metric = logit_difference_counterfactual

    min_treshold = 0.25

    # Get the correct position (last token position)
    clean_tokens = model.to_tokens(clean_prompt)
    position = clean_tokens.shape[1] - 1  # Last position
    
    print("Starting counterfactual breadth-first search...")
    
    # Use the new counterfactual BFS function
    complete_paths = breadth_first_search_with_counterfactual(
        model,
        clean_cache,
        counterfactual_cache,  # Explicit counterfactual cache
        default_metric,
        start_node = [FINAL_Node(layer=model.cfg.n_layers-1, position=position)],
        ground_truth_tokens = [clean_target_token],
        counterfactual_tokens = [counterfactual_target_token],  # Required for counterfactual mode
        max_depth = 100, # max number of components in the path (max number of nodes -2)
        max_branching_factor = 2048,
        min_contribution = min_treshold,
        min_contribution_percentage=0., #2, 5, 0.5
        inibition_task = find_subject_inibition,
        take_message_from_clean = True  # Take message from clean_cache, apply path in counterfactual_cache
    )
    
    print(f"Found {len(complete_paths)} complete paths.")

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
        "clean_prompt": clean_prompt,
        "counterfactual_prompt": counterfactual_prompt,
        "clean_answer": str(answers[example_idx][target_idx]),
        "counterfactual_answer": str(answers[example_idx][1-target_idx]),
        "target_idx": target_idx,
        "find_subject_inhibition": find_subject_inibition,
        "timestamp": datetime.now().isoformat(),
        "total_paths": len(complete_paths),
        "min_treshold": min_treshold,
        "n_layers": model.cfg.n_layers,
        "d_model": model.cfg.d_model,
        "n_heads": model.cfg.n_heads,
        "metric": default_metric.__name__,
        "search_mode": "counterfactual"  # Indicates this uses explicit counterfactual comparison
    }

    # Combine data
    output_data = {
        "metadata": metadata,
        "paths": serializable_paths
    }

    # Save to JSON file
    filename = f"detected_paths/detected_circuit_gpt2_ioi_counterfactual_{default_metric.__name__}_{min_treshold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved {len(complete_paths)} paths to {filename}")
    print(f"Top 3 paths by score:")
    for i, path in enumerate(serializable_paths[:3]):
        print(f"  {i+1}. Score: {path['score']:.4f}, Nodes: {len(path['nodes'])}")

if __name__ == "__main__":
    main() 