# DA6401 Assignment 3 — Transformer NMT (German → English)

## Project Structure

```
da6401_assignment_3/
├── model.py          # Full Transformer implementation (MHA, PE, Noam, etc.)
├── dataset.py        # Data loading, vocabulary building, batching
├── train.py          # Training pipeline + all W&B experiments
├── inference.py      # Standalone inference / interactive mode
├── requirements.txt
├── checkpoints/
│   ├── vocab.pkl     # Saved vocabulary (auto-generated on first run)
│   └── best_model.pt # Best checkpoint (auto-saved during training)
└── README.md
```

## Setup

```bash
# 1. Clone the official skeleton (optional, for reference)
git clone https://github.com/MiRL-IITM/da6401_assignment_3

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download spaCy language models
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

## Training

### Quick start (main training only)
```bash
python train.py --epochs 30 --batch_size 128 --d_model 256 --num_heads 8 --num_layers 3 --d_ff 512
```

### Main training + all W&B ablation experiments
```bash
python train.py --epochs 30 --run_experiments --project da6401_assignment3
```

### Only experiments (if main model already trained)
```bash
python train.py --only_experiments --project da6401_assignment3
```

### Recommended hyperparameters (base model)
| Param         | Value  |
|---------------|--------|
| d_model       | 256    |
| num_heads     | 8      |
| num_layers    | 3      |
| d_ff          | 512    |
| dropout       | 0.1    |
| warmup_steps  | 4000   |
| label_smooth  | 0.1    |
| batch_size    | 128    |
| max_len       | 256    |

## After Training: Upload Weights to Google Drive

1. Upload `checkpoints/best_model.pt` to your Google Drive
2. Make the file **publicly accessible** (Anyone with the link → Viewer)
3. Copy the file ID from the share URL:
   `https://drive.google.com/file/d/FILE_ID_HERE/view`
4. Set `weights_gdrive_id="FILE_ID_HERE"` in `Transformer.__init__` in `model.py`
5. Also copy `checkpoints/vocab.pkl` to Drive and update `vocab_path` or include it alongside `model.py`

**Important**: Do NOT upload `best_model.pt` to Gradescope. The autograder calls
`gdown` inside `Transformer.__init__()` to download it from your Drive.

## Inference Test (autograder simulation)

```python
from model import Transformer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer().to(device)
model.eval()

german = "Ein Hund läuft durch den Park."
english = model.infer(german)
print(english)
```

## Interactive Inference

```bash
python inference.py --interactive
```

## W&B Experiments

| Exp | Description                              | W&B run name prefix      |
|-----|------------------------------------------|--------------------------|
| 1   | Noam Scheduler vs Fixed LR               | `exp1_noam / exp1_fixed` |
| 2   | Scaling factor 1/√dk ablation            | `exp2_with_scale / no_scale` |
| 3   | Attention head visualisation (rollout)   | `exp3_attention_rollout` |
| 4   | Sinusoidal vs Learned Positional Enc.    | `exp4_sinusoidal / learned_pe` |
| 5   | Label smoothing ε=0.1 vs ε=0.0          | `exp5_smooth_0.1 / 0.0`  |

## Architecture Notes

- **Positional Encoding**: Sinusoidal (as in original paper), registered as a buffer
- **LayerNorm**: Post-LN (same as original paper); can switch to Pre-LN
- **Masking**: Padding mask for encoder; padding + causal mask for decoder
- **Optimiser**: Adam β=(0.9, 0.98), ε=1e-9
- **Label smoothing**: ε=0.1 (as in paper)
- **Noam LR**: warmup_steps=4000, d_model^{-0.5}×min(step^{-0.5}, step×warmup^{-1.5})

## Submission Checklist

- [ ] `model.py` contains `Transformer` with `infer()` method
- [ ] `Transformer.__init__()` loads vocab, tokenisers, and weights
- [ ] Weights downloaded from Google Drive inside `__init__()`
- [ ] `model.infer(german_sentence)` returns English string
- [ ] W&B report is public and link is included in submission
- [ ] Code submitted via Gradescope (no trained weights file)
