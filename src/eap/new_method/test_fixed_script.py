#!/usr/bin/env python3
"""
Test script to verify that the fixed circuit discovery script works correctly
with batched processing and fixed-length prompts.
"""

import torch
import sys
import transformers
from transformer_lens import HookedTransformer
from utils.metrics import logit_difference_counterfactual
from utils.nodes import FINAL_Node
from utils.graph_search import breadth_first_search_with_counterfactual
import dotenv
import os
dotenv.load_dotenv()

TOKEN = os.getenv("TOKEN")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_grad_enabled(False)

def test_batched_processing():
    """Test that the script works correctly with batched processing and fixed-length prompts."""
    
    # Load model
    model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE, torch_dtype=torch.float32)
    
    # Test prompts with FIXED length (all same length)
    test_prompts = [
        "When Dan and Sid went to the shops, Dan gave an apple to",
        "After Bob and Tom went to the park, Bob gave a ball to",
        "While Amy and John worked at home, Amy gave a book to",
        "As Mary and Paul left the store, Mary gave a pen to"
    ]
    
    counterfactual_prompts = [
        "When Sid and Dan went to the shops, Sid gave an apple to",
        "After Tom and Bob went to the park, Tom gave a ball to", 
        "While John and Amy worked at home, John gave a book to",
        "As Paul and Mary left the store, Paul gave a pen to"
    ]
    
    print("Testing batched circuit discovery with fixed-length prompts...")
    
    # Verify all prompts have the same length
    lengths = []
    for prompt in test_prompts:
        tokens = model.to_tokens(prompt)
        lengths.append(tokens.shape[1])
    
    print(f"Prompt lengths: {lengths}")
    if len(set(lengths)) != 1:
        print("ERROR: Prompts have different lengths!")
        return
    else:
        print(f"✓ All prompts have same length: {lengths[0]}")
    
    # Get target tokens for all examples
    clean_target_tokens = []
    counterfactual_target_tokens = []
    
    for i in range(len(test_prompts)):
        # Get target tokens (simplified for testing)
        clean_target_tokens.append(model.to_tokens(" Sid")[0][-1].item())
        counterfactual_target_tokens.append(model.to_tokens(" Dan")[0][-1].item())
    
    print(f"Clean target tokens: {clean_target_tokens}")
    print(f"Counterfactual target tokens: {counterfactual_target_tokens}")
    
    # Run model with batched processing
    print("Running model with batched processing...")
    clean_logits, clean_cache = model.run_with_cache(test_prompts)
    counterfactual_logits, counterfactual_cache = model.run_with_cache(counterfactual_prompts)
    
    print(f"Clean cache shape: {clean_cache['hook_embed'].shape}")
    print(f"Counterfactual cache shape: {counterfactual_cache['hook_embed'].shape}")
    
    # Get the correct position (same for all examples since they have same length)
    clean_tokens = model.to_tokens(test_prompts[0])
    position = clean_tokens.shape[1] - 1  # Last position
    
    print(f"Prompt length: {clean_tokens.shape[1]}, Final position: {position}")
    print(f"Processing {len(test_prompts)} examples in batch")
    
    # Test the BFS function with batched processing
    try:
        complete_paths = breadth_first_search_with_counterfactual(
            model,
            clean_cache,
            counterfactual_cache,
            logit_difference_counterfactual,
            start_node=[FINAL_Node(layer=model.cfg.n_layers-1, position=position)],
            ground_truth_tokens=clean_target_tokens,  # List of tokens for all examples
            counterfactual_tokens=counterfactual_target_tokens,  # List of tokens for all examples
            max_depth=3,  # Small depth for testing
            max_branching_factor=10,
            min_contribution=0.1,
            min_contribution_percentage=0.0,
            inibition_task=False,
            take_message_from_clean=True
        )
        
        print(f"✓ Success! Found {len(complete_paths)} paths")
        if len(complete_paths) > 0:
            print(f"  Top path score: {complete_paths[0][0]:.4f}")
            print(f"  Top path nodes: {len(complete_paths[0][1])}")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Batched test completed ===")

def test_single_example():
    """Test single example processing for comparison."""
    
    # Load model
    model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE, torch_dtype=torch.float32)
    
    # Single example
    clean_prompt = "When Dan and Sid went to the shops, Dan gave an apple to"
    counterfactual_prompt = "When Sid and Dan went to the shops, Sid gave an apple to"
    
    print("\nTesting single example processing...")
    print(f"Clean prompt: {clean_prompt}")
    print(f"Counterfactual prompt: {counterfactual_prompt}")
    
    # Get target tokens
    clean_target_token = model.to_tokens(" Sid")[0][-1].item()
    counterfactual_target_token = model.to_tokens(" Dan")[0][-1].item()
    
    print(f"Clean target token: {clean_target_token}")
    print(f"Counterfactual target token: {counterfactual_target_token}")
    
    # Run model
    clean_logits, clean_cache = model.run_with_cache(clean_prompt)
    counterfactual_logits, counterfactual_cache = model.run_with_cache(counterfactual_prompt)
    
    # Get the correct position for this specific prompt
    clean_tokens = model.to_tokens(clean_prompt)
    position = clean_tokens.shape[1] - 1  # Last position for this specific prompt
    
    print(f"Prompt length: {clean_tokens.shape[1]}, Final position: {position}")
    
    # Test the BFS function
    try:
        complete_paths = breadth_first_search_with_counterfactual(
            model,
            clean_cache,
            counterfactual_cache,
            logit_difference_counterfactual,
            start_node=[FINAL_Node(layer=model.cfg.n_layers-1, position=position)],
            ground_truth_tokens=[clean_target_token],
            counterfactual_tokens=[counterfactual_target_token],
            max_depth=3,  # Small depth for testing
            max_branching_factor=10,
            min_contribution=0.1,
            min_contribution_percentage=0.0,
            inibition_task=False,
            take_message_from_clean=True
        )
        
        print(f"✓ Success! Found {len(complete_paths)} paths")
        if len(complete_paths) > 0:
            print(f"  Top path score: {complete_paths[0][0]:.4f}")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Single example test completed ===")

if __name__ == "__main__":
    test_batched_processing()
    test_single_example() 