"""
Path to Graph Translator

This script converts path-based circuit discovery results into the format expected by the MIB leaderboard.
It takes the output from breadth_first_search (list of (contribution_score, [nodes]) tuples) and converts
it into a JSON file with the structure expected by the evaluation framework.
"""

import json
import os
import torch
from typing import List, Tuple, Dict, Any
from transformer_lens import HookedTransformerConfig
from utils.nodes import Node, EMBED_Node, ATTN_Node, MLP_Node, FINAL_Node


def node_to_leaderboard_name(node: Node) -> str:
    """
    Convert a node from your method to the leaderboard naming convention.
    
    Leaderboard format:
    - input: "input"
    - embeddings: "input" (same as input)
    - attention heads: "a{layer}.h{head}"
    - MLPs: "m{layer}"
    - logits: "logits"
    
    Args:
        node: A node from your method
        
    Returns:
        String representation in leaderboard format
    """
    if isinstance(node, EMBED_Node):
        return "input"
    elif isinstance(node, ATTN_Node):
        if node.head is not None:
            return f"a{node.layer}.h{node.head}"
        else:
            # For attention nodes without specific heads, we'll use a generic format
            # This might need adjustment based on how you want to handle these
            return f"a{node.layer}"
    elif isinstance(node, MLP_Node):
        return f"m{node.layer}"
    elif isinstance(node, FINAL_Node):
        return "logits"
    else:
        raise ValueError(f"Unknown node type: {type(node)}")


def create_edge_name(from_node: Node, to_node: Node) -> str:
    """
    Create an edge name in the leaderboard format.
    
    Args:
        from_node: Source node
        to_node: Destination node
        
    Returns:
        Edge name in format "from->to" or "from->to<q/k/v>" for attention edges
    """
    from_name = node_to_leaderboard_name(from_node)
    to_name = node_to_leaderboard_name(to_node)
    
    # Handle attention-specific edge types
    if isinstance(to_node, ATTN_Node):
        if to_node.patch_query:
            return [f"{from_name}->{to_name}<q>"]
        elif to_node.patch_keyvalue:
            return [f"{from_name}->{to_name}<k>", f"{from_name}->{to_name}<v>"]
    else:
        return [f"{from_name}->{to_name}"]


def extract_edges_from_path(path: List[Node]) -> List[Tuple[str, float]]:
    """
    Extract edges from a path of nodes.
    
    Args:
        path: List of nodes representing a path
        
    Returns:
        List of (edge_name, position_weight) tuples
    """
    edges = []
    
    for i in range(len(path) - 1):
        from_node = path[i]
        to_node = path[i + 1]
        
        # Skip self-loops (edges where from_node and to_node are the same)
        if from_node == to_node:
            continue
        
        # Also skip if nodes have the same name (different objects but same node)
        from_name = node_to_leaderboard_name(from_node)
        to_name = node_to_leaderboard_name(to_node)
        if from_name == to_name:
            continue
        
        edge_name = create_edge_name(from_node, to_node)
        
        
        edges.extend(edge_name)
    
    return edges


def convert_paths_to_graph_format(
    complete_paths: List[Tuple[float, List[Node]]], 
    model_cfg: HookedTransformerConfig,
    min_score_threshold: float = 0.0,
    normalize_scores: bool = True
) -> Dict[str, Any]:
    """
    Convert path-based results to leaderboard graph format.
    
    Args:
        complete_paths: List of (contribution_score, [nodes]) tuples from breadth_first_search
        model_cfg: Model configuration
        min_score_threshold: Minimum score threshold for including edges
        normalize_scores: Whether to normalize edge scores
        
    Returns:
        Dictionary in the format expected by Graph.to_json()
    """
    # Initialize graph structure
    graph_data = {
        "cfg": {
            "n_layers": model_cfg.n_layers,
            "n_heads": model_cfg.n_heads,
            "d_model": model_cfg.d_model,
            "parallel_attn_mlp": False,  # Default for most models
        },
        "nodes": {},
        "edges": {}
    }
    
    # Track edge scores and counts
    edge_scores = {}
    edge_counts = {}
    
    # Process each path
    for contribution_score, path in complete_paths:
        # filter with a treshold 
        #if contribution_score < min_score_threshold:
        #    continue
            
        # Extract edges from this path
        path_edges = extract_edges_from_path(path)
        
        # Add scores to edges
        for edge_name in path_edges:
            if edge_name not in edge_scores:
                edge_scores[edge_name] = contribution_score
                edge_counts[edge_name] = 0
            
            # Sum the scores of edges in different paths
            edge_scores[edge_name] += contribution_score
            edge_counts[edge_name] += 1
    
    # Normalize scores 
    #if normalize_scores and edge_scores:
    #    max_score = max(edge_scores.values())
    #    if max_score > 0:
    #        for edge_name in edge_scores:
    #            edge_scores[edge_name] /= max_score
    
    # Convert to required format
    for edge_name, score in edge_scores.items():
        # Skip self-loops (edges where from and to nodes are the same)
        if "->" in edge_name:
            from_node, to_node = edge_name.split("->")
            # Remove QKV suffixes for comparison
            from_node = from_node.split("<")[0] if "<" in from_node else from_node
            to_node = to_node.split("<")[0] if "<" in to_node else to_node
        
        graph_data["edges"][edge_name] = {
            "score": float(score),
            "in_graph": True
        }
    
    # Add all possible nodes 
    # Add nodes based on the edges we found
    node_names = set()
    for edge_name in edge_scores.keys():
        if "->" in edge_name:
            from_node, to_node = edge_name.split("->")
            # Remove QKV suffixes from node names (e.g., "a2.h8<q>" -> "a2.h8")
            from_node = from_node.split("<")[0] if "<" in from_node else from_node
            to_node = to_node.split("<")[0] if "<" in to_node else to_node
            node_names.add(from_node)
            node_names.add(to_node)
    
    for node_name in node_names:
        graph_data["nodes"][node_name] = {
            "in_graph": True
        }
    
    return graph_data


def save_graph_to_json(graph_data: Dict[str, Any], filename: str):
    """
    Save graph data to JSON file.
    
    Args:
        graph_data: Graph data dictionary
        filename: Output filename
    """
    with open(filename, 'w') as f:
        json.dump(graph_data, f, indent=2)
    print(f"Graph saved to {filename}")


def save_graph_to_pt(graph_data: Dict[str, Any], filename: str):
    """
    Save graph data to a .pt file.
    
    Args:
        graph_data: Graph data dictionary
        filename: Output filename
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(graph_data, filename)
    print(f"Graph saved to {filename}")


def get_all_possible_nodes(model_cfg: HookedTransformerConfig) -> Tuple[List[str], List[str]]:
    """
    Generates all possible source and destination nodes for a given model config.
    
    Args:
        model_cfg: Model configuration object
        
    Returns:
        A tuple containing two lists: source nodes and destination nodes
    """
    n_layers = model_cfg.n_layers
    n_heads = model_cfg.n_heads
    
    # Source nodes
    src_nodes = ["input"]
    for l in range(n_layers):
        for h in range(n_heads):
            src_nodes.append(f"a{l}.h{h}")
        src_nodes.append(f"m{l}")
        
    # Destination nodes
    dst_nodes = []
    for l in range(n_layers):
        for h in range(n_heads):
            dst_nodes.append(f"a{l}.h{h}<q>")
            dst_nodes.append(f"a{l}.h{h}<k>")
            dst_nodes.append(f"a{l}.h{h}<v>")
        dst_nodes.append(f"m{l}")
    dst_nodes.append("logits")
    
    return src_nodes, dst_nodes


def convert_graph_to_pt_format(
    graph_data: Dict[str, Any],
    model_cfg: HookedTransformerConfig,
    percentage: float
) -> Dict[str, Any]:
    """
    Converts graph data with scores to a binary graph for a given percentage of top edges.
    
    Args:
        graph_data: Graph data with edge scores
        model_cfg: Model configuration
        percentage: Percentage of top edges to keep
        
    Returns:
        Dictionary in .pt format
    """
    src_nodes, dst_nodes = get_all_possible_nodes(model_cfg)
    src_map = {name: i for i, name in enumerate(src_nodes)}
    dst_map = {name: i for i, name in enumerate(dst_nodes)}

    # Get edges and scores
    edges_with_scores = sorted(
        graph_data["edges"].items(), 
        key=lambda item: item[1]["score"], 
        reverse=True
    )

    # Determine number of edges to keep
    total_possible_edges = len(src_nodes) * len(dst_nodes)
    num_edges_to_keep = int(total_possible_edges * (percentage / 100.0))
    top_edges = {edge[0] for edge in edges_with_scores[:num_edges_to_keep]}

    # Create tensors
    edges_in_graph = torch.zeros((len(src_nodes), len(dst_nodes)), dtype=torch.bool)
    nodes_in_graph = torch.zeros(len(src_nodes), dtype=torch.bool)

    # Populate tensors
    for edge_name in top_edges:
        if "->" in edge_name:
            from_node_full, to_node_full = edge_name.split("->")
            from_node = from_node_full.split("<")[0]
            
            if from_node in src_map and to_node_full in dst_map:
                src_idx = src_map[from_node]
                dst_idx = dst_map[to_node_full]
                edges_in_graph[src_idx, dst_idx] = True
                nodes_in_graph[src_idx] = True

    return {
        "cfg": graph_data["cfg"],
        "src_nodes": src_nodes,
        "dst_nodes": dst_nodes,
        "edges_in_graph": edges_in_graph,
        "nodes_in_graph": nodes_in_graph
    }


def convert_json_to_pt_submissions(
    input_filename: str,
    output_dir: str,
    percentages: List[float]
):
    """
    Converts a single JSON file into multiple .pt submissions for different percentages.
    
    Args:
        input_filename: Path to the input JSON file
        output_dir: Directory to save the .pt files
        percentages: List of percentages for edge thresholds
    """
    print(f"Loading data from {input_filename}...")
    paths = load_paths_from_json(input_filename)
    model_cfg = get_model_config_from_json(input_filename)
    
    print("Converting paths to graph format...")
    graph_with_scores = convert_paths_to_graph_format(
        complete_paths=paths,
        model_cfg=model_cfg,
        normalize_scores=False
    )
    
    print(f"Generating .pt files for {len(percentages)} percentages...")
    for p in percentages:
        pt_data = convert_graph_to_pt_format(graph_with_scores, model_cfg, p)
        
        # Format filename as fraction
        fraction = p / 100.0
        output_filename = os.path.join(output_dir, f"{fraction}_edges.pt")
        
        save_graph_to_pt(pt_data, output_filename)
        
    print("-" * 20)



def json_node_to_node(json_node: Dict[str, Any]) -> Node:
    """
    Convert a JSON node representation to a Node object.
    
    Args:
        json_node: Dictionary representation of a node
        
    Returns:
        Node object, or None if the node should be filtered out
    """
    node_type = json_node["type"]
    layer = json_node["layer"]
    position = json_node.get("position")
    
    # Filter out nodes with position: null
    if position is None:
        return None
    
    if node_type == "EMBED_Node":
        return EMBED_Node(layer=layer, position=position)
    elif node_type == "MLP_Node":
        return MLP_Node(layer=layer, position=position)
    elif node_type == "ATTN_Node":
        head = json_node.get("head")
        keyvalue_position = json_node.get("keyvalue_position")
        patch_query = json_node.get("patch_query", True)
        patch_keyvalue = json_node.get("patch_keyvalue", True)
        return ATTN_Node(
            layer=layer, 
            head=head, 
            position=position, 
            keyvalue_position=keyvalue_position,
            patch_query=patch_query,
            patch_keyvalue=patch_keyvalue
        )
    elif node_type == "FINAL_Node":
        return FINAL_Node(layer=layer, position=position)
    else:
        raise ValueError(f"Unknown node type: {node_type}")


def load_paths_from_json(filename: str) -> List[Tuple[float, List[Node]]]:
    """
    Load paths from a JSON file in the detected circuit format.
    
    Args:
        filename: Path to the JSON file
        
    Returns:
        List of (contribution_score, [nodes]) tuples
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    paths = []
    for path_data in data["paths"]:
        score = path_data["score"]
        # Convert nodes and filter out None values (nodes with position: null)
        nodes = [json_node_to_node(node_data) for node_data in path_data["nodes"]]
        nodes = [node for node in nodes if node is not None]
        
        # Only include paths that have at least 2 nodes after filtering
        if len(nodes) >= 2:
            paths.append((score, nodes))
    
    return paths


def get_model_config_from_json(filename: str):
    """
    Extract model configuration from the JSON metadata.
    
    Args:
        filename: Path to the JSON file
        
    Returns:
        Mock config object with model parameters
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Create a config object
    class ModelConfig:
        def __init__(self, metadata: str):
            self.model_name = metadata["model"]
            self.n_layers = metadata["n_layers"]
            self.n_heads = metadata["n_heads"]
            self.d_model = metadata["d_model"]
            
    metadata = data["metadata"]
    return ModelConfig(metadata)


def convert_json_file_to_leaderboard_format(
    input_filename: str,
    output_filename: str = None,
    min_score_threshold: float = 0.0,
    normalize_scores: bool = True
) -> Dict[str, Any]:
    """
    Convert a detected circuit JSON file to leaderboard format.
    
    Args:
        input_filename: Path to the input JSON file
        output_filename: Path for the output JSON file (optional)
        min_score_threshold: Minimum score threshold for including edges
        normalize_scores: Whether to normalize edge scores
        
    Returns:
        Graph data dictionary in leaderboard format
    """
    print(f"Loading paths from {input_filename}...")
    paths = load_paths_from_json(input_filename)
    
    print(f"Extracting model configuration...")
    model_cfg = get_model_config_from_json(input_filename)
    
    print(f"Converting {len(paths)} paths to leaderboard format...")
    graph_data = convert_paths_to_graph_format(
        complete_paths=paths,
        model_cfg=model_cfg,
        min_score_threshold=min_score_threshold,
        normalize_scores=normalize_scores
    )
    
    if output_filename:
        save_graph_to_json(graph_data, output_filename)
    
    return graph_data





def main():
    """
    Example usage of the translator.
    This function shows how to use the translator with your breadth_first_search results.
    """
    # Example usage (you would replace this with your actual results)
    print("Path to Graph Translator")
    print("=" * 50)
    print("This script converts your path-based circuit discovery results")
    print("into the format expected by the MIB leaderboard.")
    print()
    print("To use this script:")
    print("1. Import the functions in your main script")
    print("2. Call convert_paths_to_graph_format() with your complete_paths")
    print("3. Save the result using save_graph_to_json()")
    print()
    print("Example:")
    print("from path_to_graph_translator import convert_paths_to_graph_format, save_graph_to_json")
    print("graph_data = convert_paths_to_graph_format(complete_paths, model.cfg)")
    print("save_graph_to_json(graph_data, 'circuit_submission.json')")
    print()
    print("For JSON files:")
    print("from path_to_graph_translator import convert_json_file_to_leaderboard_format")
    print("graph_data = convert_json_file_to_leaderboard_format('detected_circuit.json', 'submission.json')")
    print()



if __name__ == "__main__":
    main() 