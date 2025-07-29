# Counterfactual BFS Implementation

This document describes the new breadth-first search functionality that supports both ablation-based and explicit counterfactual comparisons.

## Overview

We've implemented a new version of the breadth-first search that can handle two different approaches to counterfactual analysis:

1. **Ablation Mode** (Original behavior): Uses ablation-based counterfactuals by removing specific messages
2. **Explicit Counterfactual Mode** (New behavior): Uses actual counterfactual prompts with different target tokens

## New Functions

### 1. `logit_difference_counterfactual()` in `metrics.py`

**Signature:**
```python
def logit_difference_counterfactual(clean_resid: Float[Tensor, "batch seq d_model"],
                                   counterfactual_resid: Float[Tensor, "batch seq d_model"], 
                                   model: HookedTransformer,
                                   target_tokens: List[int],
                                   counterfactual_tokens: Optional[List[int]] = None,
                                   use_ablation_mode: bool = False) -> Float:
```

**Features:**
- Computes logit difference: `y' - y` between clean and counterfactual outputs
- Handles both ablation and explicit counterfactual modes internally
- Uses `use_ablation_mode` flag to switch between modes
- Validates that `counterfactual_tokens` is provided when needed

### 2. `evaluate_path_with_counterfactual()` in `graph_search.py`

**Signature:**
```python
def evaluate_path_with_counterfactual(model, cache, counterfactual_cache, path, metric, 
                                     correct_tokens, counterfactual_tokens=None, 
                                     message=None, invert_value=False):
```

**Features:**
- Evaluates path contributions using either ablation or explicit counterfactual approach
- Automatically detects mode based on whether `counterfactual_cache` is provided
- Raises validation errors for inconsistent parameter combinations
- Maintains backward compatibility with existing metrics

### 3. `breadth_first_search_with_counterfactual()` in `graph_search.py`

**Signature:**
```python
def breadth_first_search_with_counterfactual(
    model: HookedTransformer,
    cache: ActivationCache,
    counterfactual_cache: Optional[ActivationCache],
    metric: Callable,
    start_node: list[Node],
    ground_truth_tokens: list[int],
    counterfactual_tokens: Optional[list[int]] = None,
    max_depth: int = 5,
    max_branching_factor: int = 8,
    min_contribution: float = 0.5,
    min_contribution_percentage: float = 5.0,
    inibition_task: bool = False,
) -> List[Tuple[float, List[Node]]]:
```

**Features:**
- Supports both ablation and explicit counterfactual modes
- Input validation for parameter consistency
- On-demand computation of counterfactual residuals
- Same interface as original BFS with additional optional parameters

## Usage Examples

### Ablation Mode (Original Behavior)

```python
# Use like the original breadth_first_search
paths = breadth_first_search_with_counterfactual(
    model=model,
    cache=clean_cache,
    counterfactual_cache=None,  # No counterfactual cache = ablation mode
    metric=logit_difference_counterfactual,
    start_node=start_node,
    ground_truth_tokens=ground_truth_tokens,
    counterfactual_tokens=None,  # Not needed in ablation mode
    max_depth=3,
    min_contribution=0.1
)
```

### Explicit Counterfactual Mode (New Behavior)

```python
# Use with actual counterfactual prompts
paths = breadth_first_search_with_counterfactual(
    model=model,
    cache=clean_cache,
    counterfactual_cache=counterfactual_cache,  # Explicit counterfactual cache
    metric=logit_difference_counterfactual,
    start_node=start_node,
    ground_truth_tokens=ground_truth_tokens,
    counterfactual_tokens=counterfactual_tokens,  # Required in counterfactual mode
    max_depth=3,
    min_contribution=0.1
)
```

## Key Differences

### Ablation Mode
- `counterfactual_cache=None`
- `counterfactual_tokens=None`
- Uses ablation-based counterfactuals: `clean_resid - message`
- Same target tokens for both clean and ablated
- Equivalent to original `breadth_first_search` behavior

### Explicit Counterfactual Mode
- `counterfactual_cache=actual_counterfactual_activations`
- `counterfactual_tokens=actual_counterfactual_targets`
- Uses explicit counterfactual prompts
- Different target tokens for clean vs counterfactual
- New functionality for comparing actual counterfactual scenarios

## Error Handling

The implementation includes robust error handling:

1. **Parameter Validation**: Raises errors for inconsistent parameter combinations
2. **Required Parameters**: Ensures `counterfactual_tokens` is provided when `counterfactual_cache` is provided
3. **Mode Detection**: Automatically detects mode based on parameter presence

## Backward Compatibility

- All existing functions remain unchanged
- New functions have `_with_counterfactual` suffix
- Original `breadth_first_search` continues to work as before
- New functionality is additive, not replacing

## Memory Efficiency

- On-demand computation of counterfactual residuals
- No pre-computation to save memory
- Efficient caching strategy for path evaluation

## Integration with MIB Circuit Track

The implementation follows the same patterns as the MIB circuit track dataset handling:

- Same data structure: `(clean_prompt, counterfactual_prompt, [correct_idx, incorrect_idx])`
- Same token handling approach
- Same error handling patterns
- Compatible with existing evaluation pipelines 