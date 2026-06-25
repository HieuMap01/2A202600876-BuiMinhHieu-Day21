# Lab 21 Option A Run Checklist

Use the default/course dataset:

```text
5CD-AI/Vietnamese-alpaca-gpt4-gg-translated
```

## 1. Open Notebook

Open this notebook in Google Colab:

```text
notebooks/Lab21_LoRA_Finetuning_T4.ipynb
```

Then set:

```text
Runtime > Change runtime type > T4 GPU
```

## 2. Run Option A Dataset

Run the notebook from top to bottom. Keep Option A enabled:

```python
raw = load_dataset("5CD-AI/Vietnamese-alpaca-gpt4-gg-translated", split="train")
raw = raw.shuffle(seed=42).select(range(200))
```

Do not uncomment Option B unless you want custom data.

## 3. Training Must Produce

The notebook should train:

```text
r=16 baseline
r=8 experiment
r=64 experiment
```

Expected adapter folders:

```text
/content/lab21_lora_t4/r8
/content/lab21_lora_t4/r16
/content/lab21_lora_t4/r64
```

## 4. Required Result Files

Download these from Colab:

```text
rank_experiment_summary.csv
qualitative_comparison.csv
```

If generated, also download:

```text
loss_curve.png
```

## 5. Fill REPORT.md

Open `REPORT.md` in this repo and fill:

- `max_seq_length`
- training cost
- rank table values
- loss curve analysis
- 5 qualitative examples
- final rank trade-off conclusion

## 6. Recommended Submission Structure

```text
lab21_2A202600876_BuiMinhHieu/
├── REPORT.md
├── notebook.ipynb
├── adapters/
│   └── r16/
│       ├── adapter_model.safetensors
│       └── adapter_config.json
└── results/
    ├── rank_experiment_summary.csv
    ├── qualitative_comparison.csv
    └── loss_curve.png
```

Option A only requires submitting the best adapter folder, usually `r16`, while keeping metrics for all ranks in the CSV/report.

