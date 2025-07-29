import torch
import sys
import transformers
from transformer_lens import HookedTransformer
from utils.metrics import logit_difference_counterfactual
from utils.nodes import FINAL_Node
from utils.graph_search import evaluate_path_with_counterfactual
import dotenv
import os
dotenv.load_dotenv()

TOKEN = os.getenv("TOKEN")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_grad_enabled(False)

def debug_counterfactual():
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
    
    # Test the metric directly
    print("\n=== Testing Metric Directly ===")
    
    # Get final residuals
    clean_resid = clean_cache['blocks.11.hook_resid_post']  # Final layer
    counterfactual_resid = counterfactual_cache['blocks.11.hook_resid_post']
    
    print(f"Clean resid shape: {clean_resid.shape}")
    print(f"Counterfactual resid shape: {counterfactual_resid.shape}")
    
    # Test metric
    metric_value = logit_difference_counterfactual(
        clean_resid, counterfactual_resid, model, 
        [clean_target_token], [counterfactual_target_token], 
        use_ablation_mode=False
    )
    print(f"Direct metric value: {metric_value}")
    
    # Test with different thresholds
    thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0]
    print(f"\n=== Testing Different Thresholds ===")
    for threshold in thresholds:
        meets_threshold = abs(metric_value) >= threshold
        print(f"Threshold {threshold}: {'✓' if meets_threshold else '✗'} (value: {metric_value:.6f})")
    
    # Test path evaluation
    print(f"\n=== Testing Path Evaluation ===")
    start_node = [FINAL_Node(layer=model.cfg.n_layers-1, position=14)]
    
    path_score = evaluate_path_with_counterfactual(
        model, clean_cache, counterfactual_cache, start_node,
        logit_difference_counterfactual, [clean_target_token], [counterfactual_target_token],
        take_message_from_clean=True
    )
    print(f"Path evaluation score: {path_score}")
    
    # Test with original metric for comparison
    from utils.metrics import compare_token_logit
    original_score = compare_token_logit(
        clean_resid, counterfactual_resid, model, [clean_target_token]
    )
    print(f"Original metric score: {original_score}")
    
    # Test ablation mode
    print(f"\n=== Testing Ablation Mode ===")
    # Simulate ablation by subtracting a small amount
    ablated_resid = clean_resid - 0.1 * clean_resid
    ablation_score = logit_difference_counterfactual(
        clean_resid, ablated_resid, model, 
        [clean_target_token], use_ablation_mode=True
    )
    print(f"Ablation mode score: {ablation_score}")

if __name__ == "__main__":
    debug_counterfactual() 