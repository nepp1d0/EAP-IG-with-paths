import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field
from math import ceil
from typing import List, Tuple, Set, Union
from utils.nodes import Node, ATTN_Node

class ImgNode:
	def __init__(self, cmpt: str, layer: int, head_idx: Union[int, None], position: Union[int, None], in_type: str = None):
		self.cmpt = cmpt
		self.layer = layer
		self.head_idx = head_idx
		self.position = position
		self.in_type = in_type
	
	def __repr__(self):
		return f"ImgNode(cmpt={self.cmpt}, layer={self.layer}, head_idx={self.head_idx}, position={self.position}, in_type={self.in_type})"
	
	def __str__(self):
		return f"ImgNode({self.cmpt}, {self.layer}, {self.head_idx}, {self.position}, {self.in_type})"

	def __lt__(self, other):
		if not isinstance(other, ImgNode):
			return NotImplemented
		return (self.layer, self.cmpt, self.position, self.head_idx) < (other.layer, other.cmpt, other.position, other.head_idx)

	def __eq__(self, other):
		if not isinstance(other, ImgNode):
			return False
		return (
			self.cmpt == other.cmpt and
			self.layer == other.layer and
			self.head_idx == other.head_idx and
			self.position == other.position
		)

	def __hash__(self):
		return hash((self.cmpt, self.layer, self.head_idx, self.position))


def make_graph_from_paths(paths: List[Tuple[float, List[ImgNode]]],
						  n_layers: int,
						  n_heads: int,
						  n_positions: int,
						  divide_heads: bool = True) -> nx.MultiDiGraph:
	"""
	Build a directed multigraph with nodes and weighted edges from pre-defined paths.
	Each path segment creates a distinct edge. Edges store the weight, index
	of the path, and styling attributes (like patch_query) from the dst node.

	Args:
		paths: A list of tuples, where each tuple contains (weight, [Node1, Node2, ...])
			   representing a path and its associated weight. Nodes can be ImgNode.
		n_layers: Total number of layers in the model.
		n_heads: Number of attention heads per layer (used if divide_heads=True).
		n_positions: Sequence length or number of positions.
		divide_heads: If True, creates 'sa' nodes for individual heads. 
					  If False, creates 'attn' nodes for full attention blocks.

	Returns:
		A networkx MultiDiGraph representing the paths.
	"""
	G = nx.MultiDiGraph()
	all_nodes: Set[ImgNode] = set() 
	all_edge_weights = [] 

	for path_idx, (path_weight, path_nodes) in enumerate(paths):
		if not path_nodes:
			continue
		for node in path_nodes:
			all_nodes.add(node)
		for i in range(len(path_nodes) - 1):
			src_node = path_nodes[i]
			dst_node = path_nodes[i+1]
			G.add_edge(src_node, dst_node, weight=path_weight, path_idx=path_idx, in_type=dst_node.in_type)
			all_edge_weights.append(path_weight)


	possible_nodes_context = {
		ImgNode('emb', 0, None, pos) for pos in range(n_positions)
	}
	possible_nodes_context |= {
		ImgNode('lmh', n_layers, None, pos) for pos in range(n_positions)
	}
	possible_nodes_context |= {
		ImgNode('mlp', layer, None, pos)
		for layer in range(n_layers)
		for pos in range(n_positions)
	}

	if divide_heads:
		possible_nodes_context |= {
			ImgNode('sa', layer, head, pos, in_type=t) 
			for layer in range(n_layers)
			for head in range(n_heads)
			for pos in range(n_positions)
			for t in ['query', 'key-value']
		}
	else: # Full attention blocks
		possible_nodes_context |= {
			ImgNode('attn', layer, None, pos)
			for layer in range(n_layers)
			for pos in range(n_positions)
		}

	G.add_nodes_from(all_nodes) 
	G.add_nodes_from(possible_nodes_context)
	G.graph['max_weight'] = max(all_edge_weights) if all_edge_weights else 1.0
	G.graph['min_weight'] = min(all_edge_weights) if all_edge_weights else 0.0
	G.graph['max_abs_weight'] = max(abs(w) for w in all_edge_weights) if all_edge_weights else 1.0
	G.graph['num_paths'] = len(paths)
	return G


def place_node(node: ImgNode, 
			   n_layers: int,
			   layer_spacing: float,
			   pos_spacing: float = 0.25,
			   divide_heads: bool = True,
			   n_heads: int = 0, # Only used if divide_heads is True
			   heads_per_row: int = 4 # Only used if divide_heads is True
			   ) -> tuple[float, float]:
	"""
	Compute (x, y) coordinates for each node type.
	Placement logic depends on `divide_heads`.

	Args:
		node: The node to place (ImgNode).
		n_layers: Total number of layers in the model.
		layer_spacing: Vertical spacing between layers.
		pos_spacing: Horizontal spacing between positions.
		divide_heads: If True, attention is shown per head ('sa' nodes).
					  If False, attention is shown as full blocks ('attn' nodes).
		n_heads: Number of attention heads per layer (used if divide_heads=True).
		heads_per_row: Number of sa heads per row (used if divide_heads=True).
	Returns:
		A tuple (x, y) representing the coordinates of the node.
	"""
	base_y = lambda lyr: layer_spacing * lyr

	if isinstance(node, ImgNode):
		if node.cmpt == 'emb':
			return (node.position or 0) * pos_spacing, base_y(-0.5)
		if node.cmpt == 'lmh':
			return (node.position or 0) * pos_spacing, base_y(n_layers + 0.5)

		rows = ceil(n_heads / heads_per_row) if n_heads > 0 else 2
		head_row_height = layer_spacing / (rows + 2) # +2 for MLP and spacing

		if node.cmpt == 'mlp':
			x = (node.position or 0) * pos_spacing
			y = base_y(node.layer) + (rows + 1) * head_row_height # Place MLP above SA heads
			return x, y
		if node.cmpt == 'sa':
			col = (node.head_idx or 0) % heads_per_row
			row_idx = (node.head_idx or 0) // heads_per_row
			x_offset = (col - (heads_per_row - 1) / 2) * (pos_spacing / (heads_per_row + 1))
			x = (node.position or 0) * pos_spacing + x_offset
			y = base_y(node.layer) + (row_idx + 0.75) * head_row_height
			return x, y
		if node.cmpt == 'attn':
			x = (node.position or 0) * pos_spacing
			# Place ATTN block below MLP in the lower part of the layer's vertical allocation
			y = base_y(node.layer) + head_row_height
			return x, y
	
	# Fallback: Should ideally not be reached if graph and nodes are consistent
	return (node.position or 0) * pos_spacing, base_y(node.layer)


def plot_transformer_paths(G: nx.MultiDiGraph,
							n_layers: int,
							n_heads: int,
							n_positions: int,
							example_input: list[str] = [],
							exaple_output: list[str] = [],
							cmap_name: str = 'viridis', 
							pos_spacing: float = 0.25,
							heads_per_row: int = 4,
							save_fig: bool = False,
							save_path: str = 'transformer_paths.png',
							max_w: float = None, # User-defined normalization cap for weight/contribution
							color_scheme: str = 'path_index',
							divide_heads: bool = True
						) -> None:
	"""
	Visualize the transformer multigraph G.
	The plot shows the nodes and edges of the graph over the transformer architecture.
	Components are divided by input position, layer and eventually head.

	Args:
		G: The graph to visualize (MultiDiGraph).
		n_layers: Total number of layers in the model.
		n_heads: Number of attention heads (used if divide_heads=True).
		n_positions: Sequence length or number of positions.
		example_input: List of input tokens for labeling the embeddings.
		exaple_output: List of output tokens for labeling the lmh outputs.
		cmap_name: Colormap name for edge coloring (especially for positive contributions in 'path_weight' scheme).
		pos_spacing: Horizontal spacing between attention heads positions.
		heads_per_row: Number of sa heads per row (used if divide_heads=True).
		save_fig: If True, save the figure to a file.
		save_path: Path to save the figure if save_fig is True.
		max_w: Maximum absolute contribution for color/width normalization. If None, derived from data.
		color_scheme: Color scheme for edges ('path_index', 'path_weight', or 'input_position').
		divide_heads: If True, attention is shown per head ('sa' nodes).
					  If False, attention is shown as full blocks ('attn' nodes).
	Returns:
		None. Displays the plot.
	"""
	if len(example_input) < n_positions:
		example_input = [''] * (n_positions - len(example_input)) + example_input
	if len(exaple_output) < n_positions:
		exaple_output = [''] * (n_positions - len(exaple_output)) + exaple_output

	if divide_heads:
		layer_spacing_multiplier = (ceil(n_heads / heads_per_row) if n_heads > 0 else 1) + 2
	else:
		layer_spacing_multiplier = 4.0
		n_heads = 0
		heads_per_row = 1

	layer_spacing = pos_spacing * layer_spacing_multiplier
	token_spacing = max(pos_spacing * (heads_per_row + 1), 1.)

	pos_dict = {
		node: place_node(node, n_layers, layer_spacing,
						 pos_spacing=pos_spacing,
						 divide_heads=divide_heads,
						 n_heads=n_heads, heads_per_row=heads_per_row)
		for node in G.nodes()
	}

	height = layer_spacing * (n_layers + 2) 
	width = token_spacing * (n_positions) 
	fig, ax = plt.subplots(figsize=(width, height))

	involved = {u for u, v, data in G.edges(data=True)} | {v for u, v, data in G.edges(data=True)}
	uninvolved = set(G.nodes()) - involved

	involved_emb = [n for n in involved if isinstance(n, ImgNode) and n.cmpt == 'emb']
	involved_lmh = [n for n in involved if isinstance(n, ImgNode) and n.cmpt == 'lmh']
	involved_mlp = [n for n in involved if isinstance(n, ImgNode) and n.cmpt == 'mlp']
	
	uninvolved_emb = [n for n in uninvolved if isinstance(n, ImgNode) and n.cmpt == 'emb']
	uninvolved_lmh = [n for n in uninvolved if isinstance(n, ImgNode) and n.cmpt == 'lmh']
	uninvolved_mlp = [n for n in uninvolved if isinstance(n, ImgNode) and n.cmpt == 'mlp']

	if divide_heads:
		involved_attn_sa = [n for n in involved if isinstance(n, ImgNode) and n.cmpt == 'sa']
		uninvolved_sa_bg = [n for n in uninvolved if isinstance(n, ImgNode) and n.cmpt == 'sa']
	else:
		involved_attn_full = [n for n in involved if isinstance(n, ImgNode) and n.cmpt == 'attn']
		uninvolved_attn_full_bg = [n for n in uninvolved if isinstance(n, ImgNode) and n.cmpt == 'attn']

	def draw_nodes(nodes, img_layer=0, **opts):
		if nodes:
			nx.draw_networkx_nodes(G, pos_dict, nodelist=nodes, ax=ax, **opts).set_zorder(img_layer)

	draw_nodes(involved_emb, node_shape='s', node_size=2400, node_color='white', edgecolors='black', linewidths=1.0, alpha=0.75)
	draw_nodes(involved_lmh, node_shape='s', node_size=2400, node_color='white', edgecolors='black', linewidths=1.0, alpha=0.75)
	draw_nodes(involved_mlp, node_shape='s', node_size=500, node_color='white', edgecolors='black', linewidths=1.0, alpha=0.75, img_layer=10)

	draw_nodes(uninvolved_mlp, node_shape='s', node_size=500, node_color='white', edgecolors='grey',  linewidths=0.5, alpha=0.55)
	draw_nodes(uninvolved_emb, node_shape='s', node_size=2400, node_color='white', edgecolors='grey',  linewidths=0.5, alpha=0.55)
	draw_nodes(uninvolved_lmh, node_shape='s', node_size=2400, node_color='white', edgecolors='grey',  linewidths=0.5, alpha=0.55)

	if divide_heads:
		draw_nodes(involved_attn_sa, node_shape='o', node_size=200, node_color='white', edgecolors='black', linewidths=1.0, alpha=0.75, img_layer=10)
		draw_nodes(uninvolved_sa_bg, node_shape='o', node_size=200, node_color='white', edgecolors='grey',  linewidths=0.5, alpha=0.55)
		labels_attn_inv = {n: str(n.head_idx) for n in involved_attn_sa}
		text_items = nx.draw_networkx_labels(G, pos_dict, labels=labels_attn_inv, font_size=8, font_weight='normal', ax=ax, font_color='black')
		for _, text_obj in text_items.items():
			text_obj.set_zorder(11)
	else: 
		draw_nodes(involved_attn_full, node_shape='o', node_size=500, node_color='white', edgecolors='black', linewidths=1.0, alpha=0.75, img_layer=10)
		draw_nodes(uninvolved_attn_full_bg, node_shape='o', node_size=500, node_color='white', edgecolors='grey',  linewidths=0.5, alpha=0.55)

	labels_emb_inv = {n: example_input[n.position].replace(' ', '_') for n in involved_emb}
	labels_emb_uninv = {n: example_input[n.position].replace(' ', '_') for n in uninvolved_emb}
	nx.draw_networkx_labels(G, pos_dict, labels=labels_emb_inv, font_size=12, font_weight='normal', ax=ax)
	nx.draw_networkx_labels(G, pos_dict, labels=labels_emb_uninv, font_size=12, font_weight='light', ax=ax)

	labels_lmh_inv = {n: exaple_output[n.position].replace(' ', '_') for n in involved_lmh}
	labels_lmh_uninv = {n: exaple_output[n.position].replace(' ', '_') for n in uninvolved_lmh}
	nx.draw_networkx_labels(G, pos_dict, labels=labels_lmh_inv, font_size=12, font_weight='normal', ax=ax)
	nx.draw_networkx_labels(G, pos_dict, labels=labels_lmh_uninv, font_size=12, font_weight='light', ax=ax)

	sorted_edges = sorted(G.edges(data=True, keys=True), key=lambda x: x[3]['weight'], reverse=True)

	num_paths = G.graph.get('num_paths', 1)
	
	# Determine normalization factor for contribution values (weights)
	# This will be used for both color intensity and edge width.
	graph_max_abs_weight = G.graph.get('max_abs_weight', 1.0)
	if graph_max_abs_weight == 0: graph_max_abs_weight = 1.0 # Avoid division by zero

	# Use user-provided max_w for normalization cap if available, otherwise use graph's max_abs_weight
	norm_cap = max_w if max_w is not None else graph_max_abs_weight
	if norm_cap <= 0: norm_cap = 1.0



	if color_scheme == 'path_index':
		cmap_obj = plt.get_cmap(cmap_name, num_paths if num_paths > 0 else 1)
		# This list is indexed by path_idx later
		edge_colors_by_path_idx = [cmap_obj(i) for i in np.arange(num_paths if num_paths > 0 else 1)]
	elif color_scheme == 'path_weight':
		cmap_obj = plt.get_cmap(cmap_name)
	elif color_scheme == 'input_position':
		cmap_obj_pos = plt.get_cmap(cmap_name, n_positions if n_positions > 0 else 1)
		path_idx_to_color = {}
		default_cmap_obj_pos = plt.get_cmap(cmap_name, num_paths if num_paths > 0 else 1)
		
		for in_node, _, _, data in sorted_edges:
			path_idx = data['path_idx']
			if in_node.cmpt == 'emb' and in_node.position is not None:
				path_idx_to_color[path_idx] = cmap_obj_pos(in_node.position % (n_positions if n_positions > 0 else 1))
	else:
		raise ValueError(f"Unknown color scheme: {color_scheme}. Use 'path_index', 'path_weight', or 'input_position'.")

	all_drawn_edges = []
	parallel_edge_drawn = {} 
	width_scale = 12 
	alpha = 0.6

	for i, (u, v, key, data) in enumerate(sorted_edges):
		path_idx = data['path_idx']
		contribution = data['weight'] # Renaming 'w' to 'contribution' for clarity
		
		# Edge Width: Proportional to the absolute value of the contribution
		current_width = (abs(contribution) / norm_cap) * width_scale
		current_width = max(0.1, current_width) # Ensure minimum width for visibility

		# Edge Color
		if color_scheme == 'path_weight':
			normalized_abs_contribution = abs(contribution) / norm_cap
			if normalized_abs_contribution > 1.0:
				normalized_abs_contribution = 1.0

			if contribution < 0:
				current_color = (0.6 + 0.4*normalized_abs_contribution, 0.2, 0.2) # (R, G, B) for negative
			else:
				current_color = cmap_obj(normalized_abs_contribution) # Use cmap for positive
		
		elif color_scheme == 'path_index':
			current_color = edge_colors_by_path_idx[path_idx % len(edge_colors_by_path_idx)]
		
		elif color_scheme == 'input_position':
			current_color = path_idx_to_color.get(path_idx, default_cmap_obj_pos(0.5)) # Default to middle color if not found


		edge_type = data.get('in_type', None)

		if (u,v) not in parallel_edge_drawn:
			parallel_edge_drawn[(u,v)] = 0
		rad_sign = 1 if parallel_edge_drawn[(u,v)] % 2 == 0 else -1
		rad_magnitude = 0.05 + 0.1 * (parallel_edge_drawn[(u,v)] // 2) 
		current_rad = rad_sign * rad_magnitude
		parallel_edge_drawn[(u,v)] += 1
		
		linestyle = 'solid' 
		if edge_type == 'query':
			linestyle = 'dotted'
		
		connectionstyle = f'arc3,rad={current_rad:.2f}'

		edge_patches = nx.draw_networkx_edges(
			G, pos_dict,
			edgelist=[(u, v, key)],
			width=current_width,
			edge_color=[current_color], # Pass as a list with one color
			alpha=alpha, 
			connectionstyle=connectionstyle,
			arrowstyle='-', 
			style=linestyle
		)
		if edge_patches:
			if hasattr(edge_patches, '__iter__'): 
				all_drawn_edges.extend(edge_patches)
			else:
				all_drawn_edges.append(edge_patches)

	edge_patch_map = {}
	for i, (u, v, key, data) in enumerate(sorted_edges):
		if i < len(all_drawn_edges):
			edge_patch_map[(u, v, key)] = all_drawn_edges[i]

	for u, v, key, data in sorted_edges:
		patch = edge_patch_map.get((u, v, key))
		if patch and hasattr(patch, 'set_zorder'):
			weight = data['weight']
			z_intensity = abs(weight) / (norm_cap + 1e-9) # Normalized 0-1
			z = 1 + z_intensity * 3 # Scale to 1-9
			patch.set_zorder(z)

	ax.set_yticks([layer_spacing * (i - 0.5) for i in range(0, n_layers + 3)], minor=False)
	ax.set_yticklabels(['EMB'] + list(range(n_layers)) + ['LMH'] + [''], fontsize=16)
	ax.tick_params(axis='y', left=True, labelleft=True)

	ax.set_xticks([(i - 0.5) * pos_spacing for i in range(n_positions + 1)], minor=True)
	ax.set_yticks([layer_spacing * i for i in range(0, n_layers + 1)], minor=True) 
	ax.grid(False)
	ax.grid(True, which='minor', linestyle='-', alpha=0.4)

	xs, ys = zip(*pos_dict.values()) if pos_dict else ([0],[0])
	
	max_x_offset_val = 0
	if divide_heads and n_heads > 0 and heads_per_row > 0:
		max_x_offset_val = (heads_per_row - 1) / 2 * (pos_spacing / (heads_per_row + 1))

	ax.set_xlim(min(xs) - pos_spacing - max_x_offset_val, max(xs) + pos_spacing + max_x_offset_val)
	ax.set_ylim(max(ys) + 2 * layer_spacing, min(ys) - 2 * layer_spacing) 
	ax.invert_yaxis()

	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.spines['bottom'].set_visible(False)

	fig.tight_layout()
	if save_fig:
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
	plt.show()

def get_image_paths(contrib_and_path: Tuple[float, List[Node]], divide_heads=True) -> List[ImgNode]:
	"""
	Convert a path of nodes to a list of ImgNode objects.
	"""
	contrib, path = contrib_and_path
	img_nodes = []
	for idx, node in enumerate(path):  # Access the list of nodes
		name = node.name.split('_')[0].lower()
		if 'final' in name:
			name = 'lmh'
		head_idx = None
		in_type = None
		
		position = node.position
		
		if divide_heads:
			if name == 'attn':
				name = name.replace('attn', 'sa')
				head_idx = node.head
			if isinstance(node, ATTN_Node):
				in_type = "query" if node.patch_query else "key-value"
		
		img_nodes.append(ImgNode(name, node.layer, head_idx, position, in_type=in_type))
	return (contrib, img_nodes)