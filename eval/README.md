# DynToM Evaluation Module

Evaluate LLMs on the DynToM benchmark using the exact same data, prompts, and methodology as the paper, enabling direct comparison with Table 3.

## Setup

```bash
# Create conda environment
conda create -n dyntom-eval python=3.11 -y
conda activate dyntom-eval
pip install openai anthropic google-genai tqdm
```

Set API keys as environment variables:

```bash
export OPENAI_API_KEY="..."      # For GPT models
export ANTHROPIC_API_KEY="..."   # For Claude models
export GEMINI_API_KEY="..."      # For Gemini models
```

## Supported Models

| Name | Provider | Model ID |
|------|----------|----------|
| `gpt-4-turbo` | OpenAI (Chat Completions) | `gpt-4-turbo-2024-04-09` |
| `gpt-5.4` | OpenAI (Responses API) | `gpt-5.4-2026-03-05` |
| `gpt-5.4-mini` | OpenAI (Responses API) | `gpt-5.4-mini-2026-03-17` |
| `gpt-5.4-pro` | OpenAI (Responses API) | `gpt-5.4-pro` |
| `gemini-3.1-pro` | Google GenAI | `gemini-3.1-pro-preview` |
| `claude-opus-4.6` | Anthropic | `claude-opus-4-6` |

To add a new model, add an entry to the `MODELS` dict in `config.py`.

## Running Evaluation

```bash
cd eval/

# Run a single model (default: trials 50-1149 = 1,100 social contexts)
python run_eval.py --model gpt-5.4

# Run all models
python run_eval.py --model all

# Run a subset of trials
python run_eval.py --model gpt-5.4 --start 50 --end 100

# Adjust delay between API calls (seconds, default: 2)
python run_eval.py --model gpt-5.4 --delay 3
```

Answers are saved to `results/{model_name}/trial{id}_answers.json`. Existing answer files are skipped automatically, so runs can be safely resumed.

## Analyzing Results

```bash
# Analyze a single model
python analyze.py --model gpt-5.4

# Analyze all models and print Table 3 comparison
python analyze.py --model all

# Save results to JSON
python analyze.py --model all --output results/summary.json
```

## Output

The analyzer prints a Table 3 comparison with per-mental-state Understanding (U) and Transformation (T) accuracy, plus overall AVG, alongside the paper's reported values for Human, GPT-4o, and GPT-4-Turbo.

## File Structure

```
eval/
  config.py       # Model registry, API keys, paths
  run_eval.py     # Inference runner with prompt construction and response parsing
  analyze.py      # Accuracy computation and Table 3 breakdown
  results/        # Output directory (gitignored)
    {model_name}/
      trial{id}_answers.json
    eval.log
```
