# Hybrid_AD: Hybrid Anomaly Detection Model

## Overview

Hybrid_AD is a novel anomaly detection model that combines **TimesNet** and **FEDformer** using a parallel fusion strategy with a learnable gating mechanism. This implementation provides scientifically rigorous evaluation using Balanced Point Adjustment (BA) metrics.

## Architecture

### Parallel Fusion Strategy

The model processes input through both TimesNet and FEDformer simultaneously:

```
Input (x_enc) → ┌─→ TimesNet    → X_TimesNet  ┐
                │                              ├─→ Weighted Fusion → Output
                └─→ FEDformer   → X_FEDformer ┘
                ↓
           Gating Network → g ∈ [0, 1]
```

**Fusion Formula:**

```
X_hybrid = g * X_TimesNet + (1 - g) * X_FEDformer
```

where `g` is a learnable scalar weight dynamically computed from the input.

### Gating Mechanism

The gating network is a small MLP that takes the flattened input window and produces a per-sample weight:

- **Input**: Flattened window `[B, seq_len * enc_in]`
- **Architecture**:
  - Linear(seq_len \* enc_in, 128) → ReLU → Dropout(0.1)
  - Linear(128, 64) → ReLU → Dropout(0.1)
  - Linear(64, 1) → Sigmoid
- **Output**: Scalar weight `g ∈ [0, 1]` for each sample

The sigmoid activation ensures weights remain in `[0, 1]`, providing interpretable fusion ratios.

## Key Features

1. **End-to-End Training**: The entire model (TimesNet + FEDformer + Gating Network) is trained jointly from scratch
2. **Dynamic Weighting**: Gating mechanism adapts per sample, choosing optimal balance between models
3. **Task-Specific Methods**: Supports anomaly_detection, forecast, imputation, and classification tasks
4. **Balanced Point Adjustment**: Implements scientifically rigorous BA metric alongside standard PA

## Implementation Details

### Files Created/Modified

1. **`models/Hybrid_AD.py`** (NEW)

   - Main model implementation
   - Gating network architecture
   - Parallel fusion logic
   - Task-specific forward methods

2. **`exp/exp_basic.py`** (MODIFIED)

   - Added `Hybrid_AD` import
   - Registered in `model_dict`

3. **`models/__init__.py`** (MODIFIED)

   - Added `Hybrid_AD` to exports

4. **`utils/tools.py`** (MODIFIED)

   - Added `balanced_adjustment()` function for BA metric

5. **`exp/exp_anomaly_detection.py`** (MODIFIED)

   - Added BA evaluation in test method
   - Outputs both PA and BA metrics

6. **`scripts/anomaly_detection/MSL/Hybrid_AD.sh`** (NEW)
   - Training script for MSL dataset

## Usage

### Training on MSL Dataset

On Linux/Mac/WSL:

```bash
cd Time-Series-Library
bash scripts/anomaly_detection/MSL/Hybrid_AD.sh
```

On Windows PowerShell:

```powershell
cd Time-Series-Library
$env:CUDA_VISIBLE_DEVICES="0"
python -u run.py `
  --task_name anomaly_detection `
  --is_training 1 `
  --root_path ./dataset/MSL `
  --model_id MSL_Hybrid_AD `
  --model Hybrid_AD `
  --data MSL `
  --features M `
  --seq_len 100 `
  --pred_len 0 `
  --d_model 64 `
  --d_ff 128 `
  --e_layers 2 `
  --enc_in 55 `
  --c_out 55 `
  --top_k 5 `
  --anomaly_ratio 1 `
  --batch_size 128 `
  --learning_rate 0.0001 `
  --train_epochs 10 `
  --patience 3 `
  --des 'Hybrid_AD_MSL'
```

### Testing Only

```bash
python -u run.py \
  --task_name anomaly_detection \
  --is_training 0 \
  --root_path ./dataset/MSL \
  --model_id MSL_Hybrid_AD \
  --model Hybrid_AD \
  --data MSL \
  --features M \
  --seq_len 100 \
  --enc_in 55 \
  --c_out 55 \
  --top_k 5
```

## Hyperparameters

### MSL Dataset Configuration

- **seq_len**: 100 (input window length)
- **enc_in/c_out**: 55 (MSL has 55 variables)
- **d_model**: 64 (embedding dimension)
- **d_ff**: 128 (feedforward dimension)
- **e_layers**: 2 (encoder layers)
- **top_k**: 5 (TimesNet parameter)
- **batch_size**: 128
- **learning_rate**: 0.0001
- **train_epochs**: 10

### Gating Network Configuration

- **Hidden dimension**: 128 → 64
- **Dropout**: 0.1
- **Activation**: ReLU + Sigmoid (output)

## Evaluation Metrics

The model outputs two sets of metrics:

### 1. Standard Point Adjustment (PA)

- Original TSLib default evaluation
- May overestimate performance

### 2. Balanced Point Adjustment (BA)

- More rigorous evaluation
- Only adjusts predictions within actual anomaly segments
- Does not adjust false positives in normal regions
- Provides scientifically sound comparison

### Example Output

```
=== Standard Point Adjustment ===
Accuracy : 0.9523, Precision : 0.8947, Recall : 0.9123, F-score : 0.9034

=== Balanced Point Adjustment (BA) Evaluation ===
BA - Accuracy : 0.9201, Precision : 0.8234, Recall : 0.9123, F-score : 0.8654
```

## Model Parameters

Total parameters breakdown:

- **TimesNet backbone**: ~X parameters
- **FEDformer backbone**: ~Y parameters
- **Gating Network**: ~Z parameters
- **Total**: X + Y + Z parameters

(Exact counts printed during model initialization)

## Comparison with Baselines

To compare with baseline models:

```bash
# Run TimesNet baseline
bash scripts/anomaly_detection/MSL/TimesNet.sh

# Run FEDformer baseline
bash scripts/anomaly_detection/MSL/FEDformer.sh

# Run Hybrid_AD
bash scripts/anomaly_detection/MSL/Hybrid_AD.sh
```

All results saved to `result_anomaly_detection.txt` with both PA and BA metrics.

## Technical Notes

### Why Balanced Point Adjustment?

Standard Point Adjustment (PA) can artificially inflate metrics because:

1. It adjusts entire anomaly segments even with single detection
2. May count consecutive predictions in normal regions favorably

Balanced Point Adjustment (BA) fixes this by:

1. Only adjusting within actual anomaly segments
2. Not modifying false positive behavior
3. Providing more realistic performance estimates

### Gating Network Design Choices

1. **Input**: Flattened window captures temporal dependencies and variable relationships
2. **Size**: Small MLP (seq_len \* enc_in → 128 → 64 → 1) prevents overfitting
3. **Dropout**: 0.1 dropout for regularization
4. **Sigmoid**: Ensures interpretable weights in [0, 1]

### Config Handling

Both TimesNet and FEDformer receive the same `configs` object, ensuring:

- Consistent input/output dimensions
- Same sequence length and features
- Unified training configuration

## Future Enhancements

Potential improvements:

1. **Attention-based gating**: Replace MLP with self-attention mechanism
2. **Multi-scale fusion**: Different fusion weights per time scale
3. **Channel-wise gating**: Per-variable fusion weights
4. **Ensemble techniques**: Stacking, boosting, or voting strategies

## Citation

If you use this model in your research, please cite:

```bibtex
@inproceedings{timesnet2023,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Wu, Haixu and Hu, Tengge and Liu, Yong and Zhou, Hang and Wang, Jianmin and Long, Mingsheng},
  booktitle={ICLR},
  year={2023}
}

@inproceedings{fedformer2022,
  title={FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting},
  author={Zhou, Tian and Ma, Ziqing and Wen, Qingsong and Wang, Xue and Sun, Liang and Jin, Rong},
  booktitle={ICML},
  year={2022}
}
```

## License

This implementation follows the Time-Series-Library license (MIT).

## Contact

For issues or questions, please open an issue in the Time-Series-Library repository.
