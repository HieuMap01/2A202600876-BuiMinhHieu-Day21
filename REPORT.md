# Lab 21 — Evaluation Report

**Học viên**: Bui Minh Hieu — 2A202600876  
**Ngày nộp**: 2026-06-25  
**Submission option**: A — Lightweight ZIP  

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Fine-tuning method**: QLoRA 4-bit + LoRA adapters
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`
- **Dataset option**: Option A — course/default HuggingFace dataset
- **Samples used**: 200 total, shuffled with seed `42`
- **Split**: 180 train / 20 eval
- **Data format**: Alpaca instruction format using `instruction_vi`, optional `input_vi`, and `output_vi`
- **Token length distribution**: min `25`, p50 `227`, p95 `562`, p99 `704`, max `738`
- **max_seq_length**: `1024` because p95 was rounded up and capped by the T4 profile
- **GPU**: Google Colab Tesla T4, 16 GB class VRAM, runtime reported max memory `14.563 GB`
- **Training time**: `12.8` minutes total for the three rank runs
- **Training cost estimate**: `$0.07` at `$0.35/hour`

### LoRA Configuration

| Setting | Value |
|---|---|
| Target modules | `q_proj`, `v_proj` |
| Dropout | `0` |
| Bias | `none` |
| Gradient checkpointing | `unsloth` |
| Optimizer | `adamw_8bit` |
| Learning rate | `2e-4` |
| LR schedule | cosine |
| Warmup ratio | `0.10` |
| Epochs | `3` |
| Per-device train batch size | `1` |
| Gradient accumulation | `8` |
| Effective batch size | `8` |

## 2. Rank Experiment Results

| Rank | Alpha | Trainable Params | Train Time (min) | Peak VRAM (GB) | Eval Loss | Perplexity |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 1,843,200 | 4.09 | 7.22 | 1.5577 | 4.7479 |
| 16 | 32 | 3,686,400 | 4.63 | 6.62 | 1.5161 | 4.5544 |
| 64 | 128 | 14,745,600 | 4.10 | 8.00 | 1.4768 | 4.3790 |

The quantitative result shows a consistent perplexity improvement as rank increases. Rank 8 has the fewest trainable parameters but the worst perplexity. Rank 16 improves perplexity while still keeping trainable parameters modest. Rank 64 gives the best perplexity, but it uses about 4x more trainable parameters than r=16 and the highest peak VRAM.

## 3. Loss Curve Analysis

The notebook was run in T4 mode, so mid-training evaluation was disabled to avoid VRAM issues. The available loss curve is the training loss curve, and evaluation loss was computed after each adapter finished training.

- **Training loss trend**: decreasing overall during training, with normal small fluctuations.
- **Eval loss trend across ranks**: improves from r=8 (`1.5577`) to r=16 (`1.5161`) to r=64 (`1.4768`).
- **Overfitting check**: no clear overfitting signal from the available metrics, because post-training eval loss improves with rank and no mid-training eval curve was recorded.

The results suggest that the adapters successfully learned useful instruction-following patterns from the Vietnamese Alpaca dataset. However, because only 200 examples were used and the qualitative outputs still include factual mistakes, the lower perplexity should not be interpreted as a guarantee of factual correctness.

## 4. Qualitative Comparison

### Example 1

**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.  
**Base model**: Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động.  
**Fine-tuned r=16**: Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI.  
**Comment**: Improved. The fine-tuned answer is more concise and easier for beginners, although it can still be expanded with examples.

### Example 2

**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.  
**Base model**: Để tính số Fibonacci thứ n, bạn có thể sử dụng hàm đệ quy hoặc vòng lặp. Đây là một đoạn mã Python cho phép bạn tính số Fibonacci thứ n.  
**Fine-tuned r=16**: Để tính số Fibonacci thứ n, bạn có thể viết một đoạn code Python như sau: `def fibonacci(n): if n < 0: raise ValueError(...) elif n == 0: ...`  
**Comment**: Improved. The fine-tuned output starts with clearer input validation and code structure.

### Example 3

**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.  
**Base model**: 1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là giúp người dùng hoàn thành mục tiêu dễ dàng.  
**Fine-tuned r=16**: 1. Chuyển đổi: UI/UX thiết kế phải hướng tới việc chuyển đổi người dùng.  
**Comment**: Mixed. The base model gives a more standard UI/UX principle. The fine-tuned answer is less conventional and would need manual review.

### Example 4

**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.  
**Base model**: LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) đều là kỹ thuật fine-tuning hiệu quả cho mô hình lớn.  
**Fine-tuned r=16**: LoRA (Layer-wise Adaptive Regularization Optimizer) và QLoRA ...  
**Comment**: Degraded. The fine-tuned model appears to expand LoRA incorrectly, which is a factual reliability issue. This is a useful negative case and shows why qualitative evaluation is necessary.

### Example 5

**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.  
**Base model**: Prompt engineering, RAG (retrieval augmented generation), và fine-tuning là ba cách cải thiện hành vi LLM với mức độ can thiệp khác nhau.  
**Fine-tuned r=16**: Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau để tối ưu hóa và triển khai mô hình ngôn ngữ.  
**Comment**: Similar to slightly improved. The fine-tuned answer is more organized in phrasing, but it should explain the difference more explicitly.

## 5. Conclusion về Rank Trade-off

In this experiment, rank 64 achieved the best quantitative result with the lowest eval loss (`1.4768`) and perplexity (`4.3790`). This means the larger adapter capacity helped the model fit the evaluation set better. However, rank 64 also required `14,745,600` trainable parameters, which is about four times larger than rank 16 and eight times larger than rank 8. For a small 200-sample dataset, this extra capacity may not always be worth it in production, especially if storage, memory, or multi-adapter serving cost matters. Rank 8 is efficient, but it had the worst perplexity. Rank 16 is the best practical default for this lab because it gives a clear improvement over rank 8 while keeping trainable parameters and VRAM modest. If the goal is pure validation perplexity, r=64 wins; if the goal is ROI and deployability on limited hardware, r=16 is the most balanced choice.

## 6. What I Learned

- LoRA fine-tuning changes only a small number of adapter parameters while keeping the base model frozen, which makes training feasible on a Colab T4.
- QLoRA 4-bit loading is important because it reduces base model memory enough to run a 3B model in a free/low-cost GPU environment.
- Higher rank can improve perplexity, but qualitative examples still matter because a lower loss does not guarantee factual correctness.
- Rank selection should be treated as an engineering trade-off between quality, VRAM, training time, and deployment cost.

## Files Produced

The notebook output indicates that the following files/folders were produced in Colab:

```text
/content/lab21_lora_t4/
├── r8/
├── r16/
├── r64/
├── rank_experiment_summary.csv
└── qualitative_comparison.csv
```

For the lightweight submission, include the `r16` adapter folder plus the result CSV files and this report.
