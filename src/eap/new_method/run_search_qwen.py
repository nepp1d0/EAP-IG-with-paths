import torch
import transformers
from transformer_lens import HookedTransformer
import huggingface_hub
from datasets import load_dataset
import dotenv
import os
import pickle as pkl
from utils.metrics import compare_token_probability, kl_divergence, compare_token_logit
from utils.nodes import MLP_Node, EMBED_Node, FINAL_Node, Node, ATTN_Node
from utils.graph_search import path_message, evaluate_path, breadth_first_search, breadth_first_search_cached, breadth_first_search_cached_no_pos
from datetime import datetime

transformers.logging.set_verbosity_error()
# torch.set_default_dtype(torch.bfloat16)

dotenv.load_dotenv()

TOKEN = os.getenv("TOKEN")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_grad_enabled(False)
TASK = "mcqa" # "ioi" or "mcqa"
TARGET_LENGTH = 32 # from 15 to 19 for ioi - from 32 to 37 for mcqa
BATCH_SIZE = 16 # Number of samples from the dataset to consider
DEFAULT_METRIC = compare_token_logit
CONTRIBUTION_THRESHOLD = 0.15 # 0.25 -> 2700~ path for depth=10 with ioi / 
NOSPACE = False

huggingface_hub.login(token=TOKEN)
# Note: Eventually can set set fold_ln=False, center_unembed=False, center_writing_weights=False
model = HookedTransformer.from_pretrained('Qwen/Qwen2.5-0.5B', 
                                          device=DEVICE, 
                                          torch_dtype=torch.float32, 
                                          center_unembed=True,
                                          )



samples = []
samples_prompt = []
sample_answers = []

if TASK == "ioi":
	train_dataset = load_dataset('mib-bench/ioi', split='train')
	validation_dataset = load_dataset('mib-bench/ioi', split='validation')
	test_dataset = load_dataset('mib-bench/ioi', split='test')
	for sample in train_dataset:
		if model.to_tokens(sample['prompt'], prepend_bos=True).shape[1] == TARGET_LENGTH:
			samples.append(sample)
			samples_prompt.append(sample['prompt'])
			if NOSPACE:
				sample_answers.append(model.to_tokens(f'{sample["metadata"]["indirect_object"]}', prepend_bos=False).item())
			else:
				sample_answers.append(model.to_tokens(f' {sample['metadata']['indirect_object']}', prepend_bos=False).item())
			if len(samples) >= BATCH_SIZE:
				break
elif TASK == "mcqa":
	train_dataset = load_dataset('mib-bench/copycolors_mcqa', '4_answer_choices', split='train')
	validation_dataset = load_dataset('mib-bench/copycolors_mcqa', '4_answer_choices', split='validation')
	test_dataset = load_dataset('mib-bench/copycolors_mcqa', '4_answer_choices', split='test')
	for sample in train_dataset:
		if model.to_tokens(sample['prompt'], prepend_bos=True).shape[1] == TARGET_LENGTH:
			samples.append(sample)
			samples_prompt.append(sample['prompt'])
			if NOSPACE:
				sample_answers.append(model.to_tokens(f'{sample['choices']['label'][sample['answerKey']]}', prepend_bos=False).item())
			else:
				sample_answers.append(model.to_tokens(f' {sample['choices']['label'][sample['answerKey']]}', prepend_bos=False).item())
			if len(samples) >= BATCH_SIZE:
				break
else:
	raise ValueError("Unsupported task. Please choose 'ioi' or 'MCQA'.")
print(f"Loaded {len(samples)} samples for the task {TASK} with target length {TARGET_LENGTH}.")
print(f"Sample prompt: \n''{samples_prompt[0]}''")
print(f"Sample answer: ''{model.to_string(sample_answers[0])}''")
print(f"Probability of the answer: {torch.softmax(model(samples_prompt[0], prepend_bos=True, return_type='logits')[0, -1], dim=-1)[sample_answers[0]].item()} ~ Logit: {model(samples_prompt[0], prepend_bos=True, return_type='logits')[0, -1, sample_answers[0]].item()}")


logits, cache = model.run_with_cache(samples_prompt, prepend_bos=True)

# complete_paths, incomplete_paths = breadth_first_search(
# 	model,
# 	cache,
# 	compare_token_logit,
# 	start_node = [FINAL_Node(layer=model.cfg.n_layers-1, position=TARGET_LENGTH-1)],
# 	ground_truth_tokens = sample_answers,
# 	max_depth = 15,
# 	max_branching_factor = 2048,
# 	min_contribution = CONTRIBUTION_THRESHOLD,
# 	min_contribution_percentage=0.,
# 	inibition_task = False
# )
complete_paths = breadth_first_search_cached_no_pos(
	model,
	cache,
	compare_token_logit,
	start_node = [FINAL_Node(layer=model.cfg.n_layers-1, position=None)],
	ground_truth_tokens = sample_answers,
	min_contribution = CONTRIBUTION_THRESHOLD,
	cached_path_lenght=0
)
incomplete_paths = []
print(f"Found {len(complete_paths)} complete paths and {len(incomplete_paths)} incomplete paths.")


# Save to JSON file
if NOSPACE:
	filename = f"paths_qwen2.5_{TASK}_{DEFAULT_METRIC.__name__}_{CONTRIBUTION_THRESHOLD}_bs{BATCH_SIZE}_l{TARGET_LENGTH}_nospace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
else:
	filename = f"paths_qwen2.5_{TASK}_{DEFAULT_METRIC.__name__}_{CONTRIBUTION_THRESHOLD}_bs{BATCH_SIZE}_l{TARGET_LENGTH}_nopos.pkl"
with open(filename, 'wb') as f:
    pkl.dump((complete_paths, incomplete_paths), f)