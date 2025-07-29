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
import argparse
import os
from datetime import datetime
import json

# Add the MIB circuit track to the path
sys.path.append('../../../../')
from MIB_circuit_track.dataset import HFEAPDataset

from utils.metrics import logit_difference_counterfactual
from utils.miscellanea import get_top_k_contributors, IOI_head_types
from utils.component_contributions import contribution_mlp, contribution_attn


transformers.logging.set_verbosity_error()
# torch.set_default_dtype(torch.bfloat16)

from utils.nodes import MLP_Node, EMBED_Node, FINAL_Node,Node, ATTN_Node
from utils.graph_search import breadth_first_search_with_counterfactual
import dotenv
dotenv.load_dotenv()

TOKEN = os.getenv("TOKEN")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_grad_enabled(False)

huggingface_hub.login(token=TOKEN)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2-small")
    parser.add_argument("--task", type=str, default="ioi")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num-examples", type=int, default=16)
    parser.add_argument("--example-idx", type=int, default=0)
    parser.add_argument("--max-depth", type=int, default=100)
    parser.add_argument("--max-branching-factor", type=int, default=2048)
    parser.add_argument("--min-contribution", type=float, default=15)
    parser.add_argument("--min-contribution-percentage", type=float, default=0.0)
    parser.add_argument("--find-subject-inhibition", action="store_true")
    parser.add_argument("--output-dir", type=str, default="detected_paths")
    args = parser.parse_args()
    
    # Load model
    model = HookedTransformer.from_pretrained(args.model, device=DEVICE, torch_dtype=torch.float32)
    
    # Configuration
    find_subject_inibition = args.find_subject_inhibition
    if find_subject_inibition:
        target_idx = 1  # Subject token
    else:
        target_idx = 0  # Indirect object token

    # Load dataset using MIB circuit track
    hf_task_name = f'mib-bench/ioi'
    dataset = HFEAPDataset(hf_task_name, model.tokenizer, split=args.split, task=args.task, 
                          num_examples=args.num_examples, counterfactual_type="s2_io_flip_counterfactual")
    
    # Use the new counterfactual metric
    default_metric = logit_difference_counterfactual

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect examples with fixed positions (same length)
    clean_prompts = []
    counterfactual_prompts = []
    clean_target_tokens = []
    counterfactual_target_tokens = []
    
    print("Collecting examples with fixed positions...")
    
    # First pass: collect all examples and find the most common length
    all_lengths = []
    for i in range(len(dataset)):
        clean_prompt, counterfactual_prompt, [correct_idx, incorrect_idx] = dataset[i]
        clean_tokens = model.to_tokens(clean_prompt)
        all_lengths.append(clean_tokens.shape[1])
    
    # Find the most common length
    from collections import Counter
    length_counts = Counter(all_lengths)
    most_common_length = length_counts.most_common(1)[0][0]
    print(f"Most common prompt length: {most_common_length} (appears {length_counts[most_common_length]} times)")
    
    # Second pass: collect examples with the most common length
    for i in range(len(dataset)):
        clean_prompt, counterfactual_prompt, [correct_idx, incorrect_idx] = dataset[i]
        clean_tokens = model.to_tokens(clean_prompt)
        
        if clean_tokens.shape[1] == most_common_length:
            clean_prompts.append(clean_prompt)
            counterfactual_prompts.append(counterfactual_prompt)
            clean_target_tokens.append(correct_idx)
            counterfactual_target_tokens.append(incorrect_idx)
            
            if len(clean_prompts) >= args.num_examples:
                break
    
    print(f"Collected {len(clean_prompts)} examples with length {most_common_length}")
    
    if len(clean_prompts) == 0:
        print("No examples found with consistent length. Exiting.")
        return
    
    # Process examples in batch (like the original script)
    print(f"Processing {len(clean_prompts)} examples in batch...")
    
    # Run model with both sets of prompts in batch
    clean_logits, clean_cache = model.run_with_cache(clean_prompts)
    counterfactual_logits, counterfactual_cache = model.run_with_cache(counterfactual_prompts)
    
    # Model configuration
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    n_heads = model.cfg.n_heads
    d_heads = model.cfg.d_head

    print("Starting counterfactual breadth-first search...")
    
    # Get the correct position (last token position) - same for all examples since they have same length
    clean_tokens = model.to_tokens(clean_prompts[0])  # Use first example to get position
    position = clean_tokens.shape[1] - 1  # Last position
    
    print(f"Prompt length: {clean_tokens.shape[1]}, Final position: {position}")
    print(f"Processing {len(clean_prompts)} examples with {len(clean_target_tokens)} target tokens")
    
    # Use the new counterfactual BFS function with batched processing
    complete_paths = breadth_first_search_with_counterfactual(
        model,
        clean_cache,
        counterfactual_cache,  # Explicit counterfactual cache
        default_metric,
        start_node = [FINAL_Node(layer=model.cfg.n_layers-1, position=position)],
        ground_truth_tokens = clean_target_tokens,  # List of tokens for all examples
        counterfactual_tokens = counterfactual_target_tokens,  # List of tokens for all examples
        max_depth = args.max_depth,
        max_branching_factor = args.max_branching_factor,
        min_contribution = args.min_contribution,
        min_contribution_percentage = args.min_contribution_percentage,
        inibition_task = find_subject_inibition,
        take_message_from_clean = True  # Take message from clean_cache, apply path in counterfactual_cache
    )
    
    print(f"Found {len(complete_paths)} complete paths.")

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
        "model": args.model,
        "task": args.task,
        "split": args.split,
        "num_examples_processed": len(clean_prompts),
        "prompt_length": clean_tokens.shape[1],
        "final_position": position,
        "clean_prompts": clean_prompts,
        "counterfactual_prompts": counterfactual_prompts,
        "clean_target_tokens": clean_target_tokens,
        "counterfactual_target_tokens": counterfactual_target_tokens,
        "find_subject_inhibition": find_subject_inibition,
        "timestamp": datetime.now().isoformat(),
        "total_paths": len(complete_paths),
        "min_contribution": args.min_contribution,
        "min_contribution_percentage": args.min_contribution_percentage,
        "max_depth": args.max_depth,
        "max_branching_factor": args.max_branching_factor,
        "n_layers": model.cfg.n_layers,
        "d_model": model.cfg.d_model,
        "n_heads": model.cfg.n_heads,
        "metric": default_metric.__name__,
        "search_mode": "counterfactual_batched",  # Indicates this uses batched counterfactual processing
        "processing_mode": "fixed_length_batch"  # Indicates fixed-length batch processing
    }

    # Combine data
    output_data = {
        "metadata": metadata,
        "paths": serializable_paths
    }

    # Save to JSON file
    filename = f"{args.output_dir}/detected_circuit_{args.model}_{args.task}_counterfactual_{default_metric.__name__}_{args.min_contribution}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved {len(complete_paths)} paths to {filename}")
    print(f"Top 3 paths by score:")
    for i, path in enumerate(serializable_paths[:3]):
        print(f"  {i+1}. Score: {path['score']:.4f}, Nodes: {len(path['nodes'])}")

if __name__ == "__main__":
    main() 