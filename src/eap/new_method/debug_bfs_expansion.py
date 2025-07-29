import torch
import sys
import transformers
from transformer_lens import HookedTransformer
from utils.metrics import logit_difference_counterfactual
from utils.nodes import FINAL_Node, ATTN_Node, MLP_Node, EMBED_Node
from utils.graph_search import evaluate_path_with_counterfactual
import dotenv
import os
dotenv.load_dotenv()

TOKEN = os.getenv("TOKEN")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_grad_enabled(False)

def debug_bfs_expansion():
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
    
    # Test the initial node
    start_node = [FINAL_Node(layer=model.cfg.n_layers-1, position=14)]
    initial_score = evaluate_path_with_counterfactual(
        model, clean_cache, counterfactual_cache, start_node,
        logit_difference_counterfactual, [clean_target_token], [counterfactual_target_token],
        take_message_from_clean=True
    )
    print(f"\nInitial score: {initial_score}")
    
    # Test threshold calculation
    min_contribution = 0.001
    min_contribution_percentage = 5.0
    required_contribution = max(min_contribution_percentage * abs(initial_score) / 100.0, min_contribution)
    print(f"Required contribution: {required_contribution}")
    print(f"Initial score meets threshold: {abs(initial_score) >= required_contribution}")
    
    # Test candidate expansion
    print(f"\n=== Testing Candidate Expansion ===")
    
    # Get candidate components
    proxy_component = start_node[0].__class__(start_node[0].layer)
    candidate_components = proxy_component.get_prev_nodes(
        model.cfg, include_head=False, include_bos=True)
    
    print(f"Number of candidate components: {len(candidate_components)}")
    
    # Test each candidate
    for i, candidate in enumerate(candidate_components[:5]):  # Test first 5
        print(f"\nCandidate {i}: {candidate.__class__.__name__} at layer {candidate.layer}")
        
        # Test EMBED node
        if candidate.__class__.__name__ == 'EMBED_Node':
            candidate.position = 14  # Set position
            contribution = evaluate_path_with_counterfactual(
                model, clean_cache, counterfactual_cache, [candidate] + start_node,
                logit_difference_counterfactual, [clean_target_token], [counterfactual_target_token],
                take_message_from_clean=True
            )
            print(f"  EMBED contribution: {contribution}")
            print(f"  Meets threshold: {abs(contribution) >= required_contribution}")
        
        # Test MLP node
        elif candidate.__class__.__name__ == 'MLP_Node':
            mlp_node = MLP_Node(candidate.layer, position=14)
            contribution = evaluate_path_with_counterfactual(
                model, clean_cache, counterfactual_cache, [mlp_node] + start_node,
                logit_difference_counterfactual, [clean_target_token], [counterfactual_target_token],
                take_message_from_clean=True
            )
            print(f"  MLP contribution: {contribution}")
            print(f"  Meets threshold: {abs(contribution) >= required_contribution}")
        
        # Test ATTN node
        elif candidate.__class__.__name__ == 'ATTN_Node':
            # Test whole attention component
            whole_component_message = candidate.forward(model, clean_cache, patch=None)
            whole_component_contribution = evaluate_path_with_counterfactual(
                model, clean_cache, counterfactual_cache, start_node,
                logit_difference_counterfactual, [clean_target_token], [counterfactual_target_token],
                message=whole_component_message, take_message_from_clean=True
            )
            print(f"  ATTN whole component contribution: {whole_component_contribution}")
            print(f"  Meets threshold: {abs(whole_component_contribution) >= required_contribution}")
            
            if abs(whole_component_contribution) >= required_contribution:
                # Test individual heads
                for head in range(min(3, model.cfg.n_heads)):  # Test first 3 heads
                    attn_node = ATTN_Node(candidate.layer, head=head, position=14, patch_keyvalue=True, patch_query=False)
                    message = attn_node.forward(model, clean_cache, patch=None)
                    contribution = evaluate_path_with_counterfactual(
                        model, clean_cache, counterfactual_cache, start_node,
                        logit_difference_counterfactual, [clean_target_token], [counterfactual_target_token],
                        message=message, take_message_from_clean=True
                    )
                    print(f"    Head {head} contribution: {contribution}")
                    print(f"    Meets threshold: {abs(contribution) >= required_contribution}")

if __name__ == "__main__":
    debug_bfs_expansion() 