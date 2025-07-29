# Bug Fix Summary: Batched Circuit Discovery with Fixed-Length Prompts

## Problem Description

The original circuit discovery script had a critical design misunderstanding. The issue was not with variable prompt lengths, but with trying to use variable-length prompts when the original design was based on **fixed-length prompts processed in batches**.

The main issues were:

1. **Design Mismatch**: The counterfactual script was trying to use variable-length prompts from the dataset, but the original `breadth_first_search` function was designed to work with **fixed-length prompts processed in batches**.

2. **Batched Processing Assumption**: The original design assumed that all examples in a batch have the same length, allowing the use of a fixed position for the final node.

3. **Cache Structure**: The original design expected activation caches with batch dimensions, where all tensors have shape `[batch_size, seq_len, ...]`.

## Root Cause Analysis

The original script design was:

```python
# Original design: Fixed-length prompts in batch
prompts_fixed_pos = prompts[0::2]  # Only prompts with same length
logits, cache = model.run_with_cache(prompts_fixed_pos)  # Batch processing
correct_tokens = [model.to_tokens(str(answers_fixed_pos[i][target_idx]))[0][-1].item() for i in range(len(prompts_fixed_pos))]

# Fixed position works because all prompts have same length
start_node = [FINAL_Node(layer=model.cfg.n_layers-1, position=14)]
```

But the counterfactual script was trying to use:

```python
# Problematic approach: Variable-length prompts
clean_prompt, counterfactual_prompt, [correct_idx, incorrect_idx] = dataset[args.example_idx]
# This gives single examples with potentially different lengths
```

## Solution Implemented

### 1. Fixed-Length Prompt Collection

The fix collects examples with the same length to maintain the original batched design:

```python
# First pass: find most common length
all_lengths = []
for i in range(len(dataset)):
    clean_prompt, counterfactual_prompt, [correct_idx, incorrect_idx] = dataset[i]
    clean_tokens = model.to_tokens(clean_prompt)
    all_lengths.append(clean_tokens.shape[1])

# Find the most common length
from collections import Counter
length_counts = Counter(all_lengths)
most_common_length = length_counts.most_common(1)[0][0]

# Second pass: collect examples with the most common length
for i in range(len(dataset)):
    clean_prompt, counterfactual_prompt, [correct_idx, incorrect_idx] = dataset[i]
    clean_tokens = model.to_tokens(clean_prompt)
    
    if clean_tokens.shape[1] == most_common_length:
        clean_prompts.append(clean_prompt)
        counterfactual_prompts.append(counterfactual_prompt)
        clean_target_tokens.append(correct_idx)
        counterfactual_target_tokens.append(incorrect_idx)
```

### 2. Batched Processing

Process multiple examples in a single batch, maintaining the original design:

```python
# Run model with both sets of prompts in batch
clean_logits, clean_cache = model.run_with_cache(clean_prompts)
counterfactual_logits, counterfactual_cache = model.run_with_cache(counterfactual_prompts)

# Fixed position works because all examples have same length
position = clean_tokens.shape[1] - 1  # Same for all examples

# Pass lists of tokens for all examples
complete_paths = breadth_first_search_with_counterfactual(
    model,
    clean_cache,
    counterfactual_cache,
    default_metric,
    start_node=[FINAL_Node(layer=model.cfg.n_layers-1, position=position)],
    ground_truth_tokens=clean_target_tokens,  # List for all examples
    counterfactual_tokens=counterfactual_target_tokens,  # List for all examples
    # ... other parameters
)
```

### 3. Maintained Original BFS Design

The `breadth_first_search_with_counterfactual` function was already designed to work with:
- **Batched caches**: All tensors have batch dimension `[batch_size, seq_len, ...]`
- **Multiple target tokens**: `ground_truth_tokens` and `counterfactual_tokens` are lists
- **Fixed positions**: Same position works for all examples in batch

## Key Insights

### Original Design Principles
1. **Fixed-Length Prompts**: All examples in a batch must have the same length
2. **Batched Processing**: Process multiple examples simultaneously in one cache
3. **Fixed Position**: Same final position works for all examples
4. **List of Tokens**: Pass lists of target tokens for all examples

### Why This Works
- **Efficiency**: Single forward pass processes multiple examples
- **Consistency**: Same position works for all examples
- **Scalability**: Can process many examples in one batch
- **Memory Efficiency**: Reuses cache structure across examples

## Usage Examples

### Process Examples with Fixed Lengths
```bash
python MIB_circuit_discovery_script_gpt2_ioi_counterfactual_dataset.py --num-examples 16
```

The script will:
1. Find the most common prompt length in the dataset
2. Collect examples with that length
3. Process them in a single batch
4. Use fixed position for all examples
5. Pass lists of target tokens for all examples

## Verification

The fix ensures that:

1. **Fixed Lengths**: All examples in a batch have the same length
2. **Batched Processing**: Multiple examples processed in single cache
3. **Fixed Position**: Same final position works for all examples
4. **List Tokens**: Target tokens passed as lists for all examples
5. **Original Design**: Maintains the original BFS function design

## Impact on BFS Functions

The `breadth_first_search_with_counterfactual` function was already designed correctly for batched processing:

- **Cache Structure**: Expects batched caches with `[batch_size, seq_len, ...]` tensors
- **Token Lists**: Expects lists of target tokens for all examples
- **Fixed Positions**: Works with same position for all examples
- **Batch Evaluation**: Evaluates paths across all examples simultaneously

This maintains the efficiency and design principles of the original implementation while enabling counterfactual analysis.

## Testing

A test script (`test_fixed_script.py`) was created to verify both approaches:

1. **Batched Processing**: Tests multiple examples with fixed lengths
2. **Single Example**: Tests single example processing for comparison

The test confirms that:
- Fixed-length prompts work correctly in batches
- The BFS function handles batched processing properly
- Both approaches produce valid results 