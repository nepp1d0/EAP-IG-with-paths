import numpy as np

def get_top_k_contributors(results: dict, k=5, only_heads=False, only_mlp=False, only_pos=False, inibition_task=False) -> list:
	"""
	get_top_k_contributors returns the top k contributors saved in results.

	Utility used to rank components. Calculate the mean of each element in the results dictionary and sort them in descending order. It assumes that the results dictionary is a dictionary of lists, where each
	

	Args:
		results (dict): a dictionary of lists, where each list contains the contributions of a component to the final output. The keys of the dictionary are the names of the components, and the values are lists of floats.
		k (int, optional): . Defaults to 5.
		only_heads (bool, optional): If True the function only consider components with HEAD in their name. Defaults to False.
		only_mlp (bool, optional): If True the function only consider components with MLP in their name. Defaults to False.
		only_pos (bool, optional): If True the function only consider components with POS in their name. Defaults to False.

	Returns:
		list: set of tuples, where each tuple contains the name of the component and its mean contribution to the final output. The list is sorted in descending order of contribution.
	"""
	components = results.copy()
	components = {k: np.mean(np.array(v)) for k, v in components.items()}
	components_sorted = sorted(components.items(), key=lambda x: x[1], reverse=not inibition_task)
	if only_heads:
		components_sorted = [(k, v) for k, v in components_sorted if "HEAD" in k]
	if only_mlp:
		components_sorted = [(k, v) for k, v in components_sorted if "MLP" in k]
	if only_pos:
		components_sorted = [(k, v) for k, v in components_sorted if "POS" in k]
	top_k = components_sorted[:k]
	return top_k



# IOI circuit head types, ground truth from the paper:
# Wang, Kevin & Variengien, Alexandre & Conmy, Arthur & Shlegeris, Buck & Steinhardt, 
# Jacob. (2022). Interpretability in the Wild: a Circuit for Indirect Object 
# Identification in GPT-2 small. 10.48550/arXiv.2211.00593. 
IOI_head_types = {
	"HEAD_0_1": "Duplicate Token Heads",
	"HEAD_3_0": "Duplicate Token Heads",
	"HEAD_0_10": "Duplicate Token Heads",

	"HEAD_2_2": "Previous Token Heads",
	"HEAD_4_11": "Previous Token Heads",

	"HEAD_5_5": "Induction Heads",
	"HEAD_6_9": "Induction Heads",
	"HEAD_5_8": "Induction Heads",
	"HEAD_5_9": "Induction Heads",

	"HEAD_7_3": "S-Inhibition Heads",
	"HEAD_7_9": "S-Inhibition Heads",
	"HEAD_8_6": "S-Inhibition Heads",
	"HEAD_8_10": "S-Inhibition Heads",

	"HEAD_10_7": "Negative Name Mover Heads",
	"HEAD_11_10": "Negative Name Mover Heads",

	"HEAD_9_9": "Name Mover Heads",
	"HEAD_9_6": "Name Mover Heads",
	"HEAD_10_0": "Name Mover Heads",

	"HEAD_9_0": "Backup Name Mover Heads",
	"HEAD_9_7": "Backup Name Mover Heads",
	"HEAD_10_1": "Backup Name Mover Heads",
	"HEAD_10_2": "Backup Name Mover Heads",
	"HEAD_10_6": "Backup Name Mover Heads",
	"HEAD_10_10": "Backup Name Mover Heads",
	"HEAD_11_2": "Backup Name Mover Heads",
	"HEAD_11_9": "Backup Name Mover Heads"
}