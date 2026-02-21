# Guide: Using Custom Beam Search in lmms-eval

## Overview
This guide explains how to use your custom beam search implementation from `/home/dalinw/DeployedProjects/transformers` in lmms-eval evaluations.

## Setup Complete ✅

I've already added a debug print statement to your `beam_search.py` that will print when the beam search is initialized:

```python
🔬 CUSTOM BEAM SEARCH INITIALIZED!
   Parameters: num_beams=X, batch_size=Y
   length_penalty=Z, do_early_stopping=...
```

## How Generation Parameters Flow in lmms-eval

### 1. Command Line → Parser
```bash
lmms-eval --model <model> --tasks <task> --gen_kwargs "num_beams=4,temperature=0.0"
```

### 2. Parser → Evaluator
The `--gen_kwargs` string is parsed by `simple_parse_args_string()` in `evaluator.py`:
```python
gen_kwargs = simple_parse_args_string(gen_kwargs)  # "num_beams=4" → {"num_beams": 4}
```

### 3. Evaluator → Model Implementation
The parsed kwargs are passed to the model's `generate_until()` method.

### 4. Model → Transformers .generate()
Each model in `lmms_eval/models/simple/` calls transformers' `.generate()` with these parameters.

**Example from `qwen2_vl.py` (line 360-380):**
```python
default_gen_kwargs = {
    "max_new_tokens": 128,
    "temperature": 0.0,
    "top_p": None,
    "num_beams": 1,  # ← Default is greedy (no beam search)
}
current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}  # CLI overrides defaults

cont = self.model.generate(
    **inputs,
    num_beams=current_gen_kwargs["num_beams"],  # ← Passed to transformers
    do_sample=True if current_gen_kwargs["temperature"] > 0 else False,
    temperature=current_gen_kwargs["temperature"],
    top_p=current_gen_kwargs["top_p"],
    max_new_tokens=current_gen_kwargs["max_new_tokens"],
)
```

## Step-by-Step: Test Your Custom Beam Search

### Step 1: Install lmms-eval with Your Local Transformers

```bash
cd /home/dalinw/DeployedProjects/lmms-eval
pip install -e ".[ltc]"
```

The `[ltc]` extra is already configured in `pyproject.toml` to use your local transformers:
```toml
ltc = [
    "transformers[torch] @ file:///home/dalinw/DeployedProjects/transformers",
    "qwen-vl-utils",
]
```

### Step 2: Verify Installation

```bash
python -c "import transformers; print(transformers.__file__)"
```
Should output: `/home/dalinw/DeployedProjects/transformers/src/transformers/__init__.py`

### Step 3: Run Evaluation with Beam Search

```bash
cd /home/dalinw/DeployedProjects/lmms-eval

# Example with a simple task (adjust model and task as needed)
lmms-eval \
    --model qwen2_vl \
    --model_args "pretrained=Qwen/Qwen2-VL-2B-Instruct" \
    --tasks mme \
    --batch_size 1 \
    --log_samples \
    --output_path ./results \
    --gen_kwargs "num_beams=4,temperature=0.0,max_new_tokens=50"
```

### Step 4: Check for Debug Output

When the evaluation starts and beam search is triggered, you should see:
```
================================================================================
🔬 CUSTOM BEAM SEARCH INITIALIZED!
   Parameters: num_beams=4, batch_size=1
   length_penalty=1.0, do_early_stopping=False
================================================================================
```

## Which Models Use Beam Search?

Based on the code search, most models in lmms-eval support `num_beams`:
- `qwen2_vl.py`
- `qwen2_5_vl.py`
- `llava_onevision.py`
- `llava_ov_qwen2.py`
- `vita.py`
- `vila.py`
- `gemma3.py`
- `phi3v.py`
- `phi4_multimodal.py`
- And many more...

## Make a More Obvious Test Change

If you want to see a concrete effect of your custom beam search, modify the scoring in `beam_search.py`:

**Location:** `/home/dalinw/DeployedProjects/transformers/src/transformers/generation/beam_search.py`

**In the `process()` method (around line 235):**

```python
def process(
    self,
    input_ids: torch.LongTensor,
    next_scores: torch.FloatTensor,
    next_tokens: torch.LongTensor,
    next_indices: torch.LongTensor,
    ...
) -> Dict[str, torch.Tensor]:
    # 🔬 CUSTOM MODIFICATION: Add small penalty to verify custom beam search
    print(f"🔬 Custom beam_search.process() called! Original scores shape: {next_scores.shape}")
    next_scores = next_scores - 0.01  # Slight penalty for testing
    
    # add up to the length which the next_scores is calculated on
    cur_len = input_ids.shape[-1] + 1
    # ... rest of method
```

This will:
1. Print every time the beam search processes candidates
2. Slightly modify scores, which should affect the output

## Key Generation Parameters

| Parameter | Description | Default | Effect on Beam Search |
|-----------|-------------|---------|----------------------|
| `num_beams` | Number of beams for beam search | 1 (greedy) | **Must be >1 to trigger beam search** |
| `temperature` | Sampling temperature | 0.0 | If >0, uses sampling; if 0, uses deterministic |
| `do_sample` | Whether to use sampling | False | Should be False for beam search |
| `top_p` | Nucleus sampling parameter | None | Not used with beam search |
| `max_new_tokens` | Max tokens to generate | 128 | Limits generation length |
| `length_penalty` | Penalty for longer sequences | 1.0 | >1 encourages longer, <1 shorter |

## Important Notes

1. **Beam search only activates when `num_beams > 1`**. Default is 1 (greedy decoding).

2. **Your custom beam search in `/home/dalinw/DeployedProjects/transformers` will be used** because:
   - lmms-eval is installed with `[ltc]` extra
   - This points to your local transformers installation
   - Models call `self.model.generate()` which uses transformers' generation

3. **The debug print will appear ONLY when beam search is actually used**, not for greedy decoding.

4. **To force beam search**, always include `num_beams=<N>` where N > 1 in your `--gen_kwargs`.

## Example Commands

### Test with Beam Search (num_beams=4)
```bash
lmms-eval \
    --model qwen2_vl \
    --model_args "pretrained=Qwen/Qwen2-VL-2B-Instruct" \
    --tasks mme \
    --batch_size 1 \
    --gen_kwargs "num_beams=4,max_new_tokens=30"
```

### Test with Greedy (no beam search, won't see debug print)
```bash
lmms-eval \
    --model qwen2_vl \
    --model_args "pretrained=Qwen/Qwen2-VL-2B-Instruct" \
    --tasks mme \
    --batch_size 1 \
    --gen_kwargs "num_beams=1,max_new_tokens=30"
```

## Troubleshooting

### Debug print doesn't appear?
1. Check if `num_beams > 1` in your gen_kwargs
2. Verify transformers installation: `pip show transformers` should show your local path
3. Check if the model/task actually generates text (some tasks are multiple-choice)

### Want to see which transformers is being used?
```bash
python -c "import transformers; print(transformers.__file__)"
```

### Want to force reinstall?
```bash
cd /home/dalinw/DeployedProjects/lmms-eval
pip uninstall transformers -y
pip install -e ".[ltc]"
```

## Research Workflow

1. **Make changes** to `/home/dalinw/DeployedProjects/transformers/src/transformers/generation/beam_search.py`
2. **No need to reinstall** - Python uses the source files directly in editable install
3. **Run evaluation** with `num_beams > 1`
4. **Check debug output** to confirm your changes are active
5. **Compare results** with vanilla beam search (use a different transformers install)

## Summary

✅ Your custom beam search is ready to use  
✅ Debug print added to verify it's active  
✅ Use `--gen_kwargs "num_beams=4"` to trigger beam search  
✅ Models in lmms-eval will automatically use your custom version  
✅ No reinstall needed after code changes  

Good luck with your beam search research! 🔬
