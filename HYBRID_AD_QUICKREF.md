# Hybrid_AD Quick Reference

## Quick Start (Copy-Paste Ready)

### Test Installation

```bash
python test_hybrid_ad.py
```

### Train on MSL (Linux/Mac/WSL)

```bash
bash scripts/anomaly_detection/MSL/Hybrid_AD.sh
```

### Train on MSL (Windows PowerShell)

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
python -u run.py --task_name anomaly_detection --is_training 1 --root_path ./dataset/MSL --model_id MSL_Hybrid_AD --model Hybrid_AD --data MSL --features M --seq_len 100 --pred_len 0 --d_model 64 --d_ff 128 --e_layers 2 --enc_in 55 --c_out 55 --top_k 5 --anomaly_ratio 1 --batch_size 128 --learning_rate 0.0001 --train_epochs 5 --patience 3 --des 'Hybrid_AD_MSL'
```

## Architecture Summary

```
Input [B, 100, 55] → ┌─→ TimesNet → [B, 100, 55] ┐
                     │                            ├─→ g*TN + (1-g)*FED → [B, 100, 55]
                     └─→ FEDformer → [B, 100, 55]┘
                     ↓
                Gating MLP → g ∈ [0,1]
```

## Key Files

| File                                         | Purpose    | Lines |
| -------------------------------------------- | ---------- | ----- |
| `models/Hybrid_AD.py`                        | Main model | 155   |
| `utils/tools.py`                             | BA metric  | +45   |
| `exp/exp_anomaly_detection.py`               | Evaluation | +20   |
| `scripts/anomaly_detection/MSL/Hybrid_AD.sh` | Training   | 1     |

## Hyperparameters (MSL)

| Parameter  | Value  | Note            |
| ---------- | ------ | --------------- |
| seq_len    | 100    | Window size     |
| enc_in     | 55     | MSL variables   |
| d_model    | 64     | Embedding dim   |
| d_ff       | 128    | FF dim          |
| e_layers   | 2      | Encoder layers  |
| top_k      | 5      | TimesNet param  |
| batch_size | 128    | Batch size      |
| lr         | 0.0001 | Learning rate   |
| epochs     | 10     | Training epochs |

## Metrics Comparison

| Metric Type   | What it Measures    | Expected Range |
| ------------- | ------------------- | -------------- |
| PA (Standard) | Optimistic estimate | 0.90-0.95      |
| BA (Balanced) | Rigorous estimate   | 0.85-0.90      |

**BA is more scientifically sound - use it for paper reporting!**

## Common Commands

```bash
# Check model parameters
python -c "from test_hybrid_ad import test_hybrid_ad; test_hybrid_ad()"

# View results
cat result_anomaly_detection.txt | tail -20

# Compare with baseline
bash scripts/anomaly_detection/MSL/TimesNet.sh
bash scripts/anomaly_detection/MSL/FEDformer.sh
bash scripts/anomaly_detection/MSL/Hybrid_AD.sh

# Check checkpoint
ls -lh checkpoints/anomaly_detection_MSL_Hybrid_AD_*/

# GPU monitoring
nvidia-smi -l 1
```

## Troubleshooting

| Issue          | Solution                                      |
| -------------- | --------------------------------------------- |
| Import error   | Check `models/__init__.py` includes Hybrid_AD |
| OOM            | Reduce batch_size to 64 or 32                 |
| Shape mismatch | Verify enc_in=55, c_out=55                    |
| No dataset     | Download MSL to `./dataset/MSL/`              |

## Expected Output

```
Hybrid_AD initialized with:
  - TimesNet: XXX,XXX params
  - FEDformer: XXX,XXX params
  - Gating Network: X,XXX params

Epoch: 1 | Train Loss: 0.XXXX Vali Loss: 0.XXXX Test Loss: 0.XXXX
...
Best model saved!

=== Standard Point Adjustment ===
Accuracy : 0.XXXX, Precision : 0.XXXX, Recall : 0.XXXX, F-score : 0.XXXX

=== Balanced Point Adjustment (BA) Evaluation ===
BA - Accuracy : 0.XXXX, Precision : 0.XXXX, Recall : 0.XXXX, F-score : 0.XXXX
```

## Model Registry Check

```python
# Quick verification in Python
import sys
sys.path.insert(0, '.')
from exp.exp_basic import Exp_Basic
from argparse import Namespace

args = Namespace(model='Hybrid_AD', use_gpu=False)
exp = Exp_Basic(args)
print("✓ Hybrid_AD registered successfully!")
```

## Fusion Formula

```python
g = sigmoid(MLP(x_enc.flatten()))  # [B, 1, 1]
output = g * timesnet_out + (1 - g) * fedformer_out
```

Where:

- `g ∈ [0, 1]` (per sample)
- `g=1.0` → pure TimesNet
- `g=0.0` → pure FEDformer
- `g=0.5` → equal fusion

## Paper Reporting Template

```latex
We implemented a hybrid anomaly detection model (Hybrid_AD) that
combines TimesNet and FEDformer using a learnable gating mechanism.
The model achieves an F-score of X.XX (BA metric) on the MSL dataset,
outperforming TimesNet (Y.YY) and FEDformer (Z.ZZ) baselines.

The gating network learns sample-specific fusion weights g ∈ [0,1],
producing the final output as: X_hybrid = g·X_TimesNet + (1-g)·X_FEDformer.
```

## Citation

```bibtex
@misc{hybrid_ad_2025,
  title={Hybrid Anomaly Detection via Adaptive Fusion of TimesNet and FEDformer},
  note={Implementation in Time-Series-Library},
  year={2025}
}
```

---

**Quick Links:**

- Full docs: `HYBRID_AD_README.md`
- Summary: `HYBRID_AD_SUMMARY.md`
- Test script: `test_hybrid_ad.py`
- Model code: `models/Hybrid_AD.py`
