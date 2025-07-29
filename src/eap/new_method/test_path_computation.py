import torch
import sys
import transformers
from transformer_lens import HookedTransformer
from utils.metrics import logit_difference_counterfactual
from utils.nodes import FINAL_Node, ATTN_Node, MLP_Node
from utils.graph_search import evaluate_path_with_counterfactual, path_message
import dotenv
import os
dotenv.load_dotenv()

TOKEN = os.getenv("TOKEN")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_grad_enabled(False)

def test_path_computation():
    # Load model
    model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE, torch_dtype=torch.float32)
    
    # Test prompts
    clean_prompt = "When Dan and Sid went to the shops, Dan gave an apple to"
    counterfactual_prompt = "When Sid and Dan went to the shops, Sid gave an apple to"
    
    print(f"Clean prompt: {clean_prompt}")
    print(f"Counterfactual prompt: {counterfactual_prompt}")
    
    # Run model
    clean_logits, clean_cache = model.run_with_cache(clean_prompt)
    counterfactual_logits, counterfactual_cache = model.run_with_cache(counterfactual_prompt)
    
    # Get target tokens
    clean_target_token = model.to_tokens(" Sid")[0][-1].item()
    counterfactual_target_token = model.to_tokens(" Dan")[0][-1].item()
    
    print(f"Clean target token: {clean_target_token}")
    print(f"Counterfactual target token: {counterfactual_target_token}")
    
    # Test different path lengths
    print(f"\n=== Testing Different Path Lengths ===")
    
    # Test 1: Just final node
    final_node = [FINAL_Node(layer=model.cfg.n_layers-1, position=14)]
    
    score_final = evaluate_path_with_counterfactual(
        model, clean_cache, counterfactual_cache, final_node,
        logit_difference_counterfactual, [clean_target_token], [counterfactual_target_token],
        take_message_from_clean=True
    )
    print(f"Final node only: {score_final}")
    
    # Test 2: Final node + MLP
    mlp_node = MLP_Node(layer=model.cfg.n_layers-1, position=14)
    path_mlp = [mlp_node, final_node[0]]
    
    score_mlp = evaluate_path_with_counterfactual(
        model, clean_cache, counterfactual_cache, path_mlp,
        logit_difference_counterfactual, [clean_target_token], [counterfactual_target_token],
        take_message_from_clean=True
    )
    print(f"MLP + Final: {score_mlp}")
    
    # Test 3: Final node + Attention
    attn_node = ATTN_Node(layer=model.cfg.n_layers-1, head=0, position=14)
    path_attn = [attn_node, final_node[0]]
    
    score_attn = evaluate_path_with_counterfactual(
        model, clean_cache, counterfactual_cache, path_attn,
        logit_difference_counterfactual, [clean_target_token], [counterfactual_target_token],
        take_message_from_clean=True
    )
    print(f"Attention + Final: {score_attn}")
    
    # Test with lower threshold
    print(f"\n=== Testing with Lower Threshold ===")
    thresholds = [0.001, 0.01, 0.05, 0.1]
    for threshold in thresholds:
        meets_threshold = abs(score_final) >= threshold
        print(f"Threshold {threshold}: {'✓' if meets_threshold else '✗'} (value: {score_final:.6f})")
    
    # Test both directions
    print(f"\n=== Testing Both Directions ===")
    score_clean_to_counter = evaluate_path_with_counterfactual(
        model, clean_cache, counterfactual_cache, final_node,
        logit_difference_counterfactual, [clean_target_token], [counterfactual_target_token],
        take_message_from_clean=True
    )
    
    score_counter_to_clean = evaluate_path_with_counterfactual(
        model, clean_cache, counterfactual_cache, final_node,
        logit_difference_counterfactual, [clean_target_token], [counterfactual_target_token],
        take_message_from_clean=False
    )
    
    print(f"Clean→Counter: {score_clean_to_counter}")
    print(f"Counter→Clean: {score_counter_to_clean}")

if __name__ == "__main__":
    test_path_computation() 