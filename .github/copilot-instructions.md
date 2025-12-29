# Time Series Library (TSLib) - AI Agent Instructions

## Architecture Overview

TSLib is a unified time-series deep learning library supporting 6 tasks: **long/short-term forecasting, imputation, anomaly detection, classification, and zero-shot forecasting**. The architecture follows a task-experiment-model pattern:

- **Entry point**: [run.py](run.py) - all experiments start here with CLI arguments
- **Task dispatching**: [run.py](run.py) instantiates one of 6 Exp classes based on `--task_name` (lines 196-208)
- **Experiment classes**: [exp/](exp/) - each task has a dedicated Exp class (e.g., `Exp_Long_Term_Forecast`)
- **Model registry**: [exp/exp_basic.py](exp/exp_basic.py) - `model_dict` (lines 11-59) maps model names to implementations
- **Models**: [models/](models/) - each model is a standalone file with a `Model` class
- **Data providers**: [data_provider/data_factory.py](data_provider/data_factory.py) - `data_provider()` function returns dataset/loader

### Critical Data Flow

1. [run.py](run.py) parses args → selects Exp class → creates experiment instance
2. Exp class calls `_build_model()` → looks up model in `model_dict` → instantiates from [models/](models/)
3. Exp class calls `_get_data()` → delegates to `data_provider()` → returns DataLoader
4. Training loop in Exp class: `train()` → `vali()` → `test()`

### Data Split Pattern

Fixed ratio splits in [data_provider/data_loader.py](data_provider/data_loader.py) (lines 64-66 for ETT datasets):

- Train: first 70% (12 months for ETT)
- Val: next 10% (4 months)
- Test: last 20% (8 months)

The border indices are computed as `[0, train_end - seq_len, val_end - seq_len]` to handle look-back windows.

## Adding New Models

**Required pattern** (see [models/TimesNet.py](models/TimesNet.py) as reference):

```python
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name  # Always check task type
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # ... model architecture ...

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, seq_len, enc_in] - historical data
        # x_mark_enc: [B, seq_len, time_features] - temporal encoding
        # x_dec: [B, label_len+pred_len, dec_in] - decoder input
        # x_mark_dec: [B, label_len+pred_len, time_features]

        # Return shapes by task:
        # Forecast: [B, pred_len, c_out]
        # Imputation/Anomaly: [B, seq_len, c_out]
        # Classification: [B, num_class]
```

**Task-specific methods** (optional, called automatically by Exp classes):

- `forecast(...)` for long/short-term forecasting
- `imputation(..., mask)` for imputation
- `anomaly_detection(x_enc)` for anomaly detection
- `classification(x_enc, x_mark_enc)` for classification

**Integration steps**:

1. Add model file to [models/](models/) directory
2. Add import in [models/**init**.py](models/__init__.py): `from .YourModel import Model as YourModel`
3. Register in [exp/exp_basic.py](exp/exp_basic.py) `model_dict`: `'YourModel': YourModel`
4. Create shell scripts in [scripts/{task}/{dataset}\_script/](scripts/) for testing
5. Follow existing script patterns (e.g., [scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh](scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh))

**Important**: Models like Mamba that require special dependencies should use conditional imports in [exp/exp_basic.py](exp/exp_basic.py) (see lines 60-63).

## Running Experiments

**Always use shell scripts** from [scripts/](scripts/) - they contain validated hyperparameters. Example:

```bash
bash scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh
```

**Never run `run.py` directly without args** - it requires 26+ parameters. Scripts handle this complexity.

### Critical CLI Arguments

From [run.py](run.py), most important args:

```bash
python -u run.py \
  --task_name long_term_forecast \      # Task: long_term_forecast, short_term_forecast,
                                        #   imputation, anomaly_detection, classification, zero_shot_forecast
  --is_training 1 \                     # 1=train+test, 0=test only
  --model TimesNet \                    # Model name from model_dict
  --data ETTh1 \                        # Dataset identifier (maps to Dataset class)
  --root_path ./dataset/ETT-small/ \    # Data directory
  --data_path ETTh1.csv \               # Specific file
  --features M \                        # M=multivariate, S=univariate, MS=multivariate→univariate
  --seq_len 96 \                        # Input lookback window
  --label_len 48 \                      # Decoder start token length (forecast tasks only)
  --pred_len 96 \                       # Forecast horizon
  --enc_in 7 --dec_in 7 --c_out 7 \     # Channel dimensions (must match data)
  --d_model 512 --d_ff 2048 \           # Model architecture dimensions
  --e_layers 2 --d_layers 1 \           # Encoder/decoder layer counts
  --batch_size 32 \                     # Training batch size
  --learning_rate 0.0001 \              # Optimizer learning rate
  --train_epochs 10                     # Number of training epochs
```

**Checkpoint naming convention** (auto-generated in [run.py](run.py) lines 210-230):

```
{task}_{model_id}_{model}_{data}_ft{features}_sl{seq_len}_ll{label_len}_pl{pred_len}_dm{d_model}_nh{n_heads}_el{e_layers}_dl{d_layers}_df{d_ff}_expand{expand}_dc{d_conv}_fc{factor}_eb{embed}_dt{distil}_{des}_{iteration}
```

Example: `long_term_forecast_ETTh1_96_96_TimesNet_ETTh1_ftM_sl96_ll48_pl96_dm16_nh8_el2_dl1_df32_...`

### Zero-Shot Forecasting (LTSMs)

Large Time Series Models use different workflow:

```bash
--task_name zero_shot_forecast \
--is_training 0 \                # Always 0 for zero-shot
--model Chronos \                # Or: Moirai, TimesFM, TimeMoE, Sundial, TiRex, Chronos2
```

See [exp/exp_zero_shot_forecasting.py](exp/exp_zero_shot_forecasting.py) for implementation details.

## Dataset Expectations

All datasets must be in [dataset/](dataset/) directory. If not found locally, some datasets auto-download from HuggingFace (see [data_provider/data_loader.py](data_provider/data_loader.py) lines 58-62).

**Time-series CSV format**:

- First column: timestamp (e.g., `date`)
- Remaining columns: feature values
- Example: ETT datasets have 7 features (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT)

**Anomaly detection**: Uses `.npy` files (e.g., [dataset/MSL/MSL_train.npy](dataset/MSL/MSL_train.npy))

**Classification**: UEA/UCR format handled by [data_provider/uea.py](data_provider/uea.py)

**Data loading flow**:

1. [data_provider/data_factory.py](data_provider/data_factory.py) maps `args.data` to Dataset class via `data_dict` (lines 5-19)
2. Dataset class in [data_provider/data_loader.py](data_provider/data_loader.py) handles reading and preprocessing
3. Standardization: `StandardScaler` fitted on train split, applied to all splits (lines 76-81)
4. Time features: encoded via [utils/timefeatures.py](utils/timefeatures.py) when `--embed timeF`

**Common datasets**:

- `ETTh1/h2`: hourly electricity transformer temperature (7 features)
- `ETTm1/m2`: minute-level ETT data
- `custom`: for user-provided CSVs (use `--data custom --data_path your_file.csv`)

## Model Implementation Patterns

### Normalization (Instance Normalization)

Most models use instance normalization from Non-stationary Transformer:

```python
# Normalize (in forward pass)
means = x_enc.mean(1, keepdim=True).detach()
x_enc = x_enc - means
stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
x_enc = x_enc / stdev

# ... model processing ...

# De-normalize (before returning)
dec_out = dec_out * stdev[:, -self.pred_len:, :] + means[:, -self.pred_len:, :]
```

### Embedding Layer

Use [layers/Embed.py](layers/Embed.py) `DataEmbedding` for standard time-series encoding:

```python
from layers.Embed import DataEmbedding

self.enc_embedding = DataEmbedding(
    configs.enc_in,      # Number of input channels
    configs.d_model,     # Embedding dimension
    configs.embed,       # 'timeF', 'fixed', or 'learned'
    configs.freq,        # Time frequency ('h', 'd', etc.)
    configs.dropout
)
# Returns: [B, seq_len, d_model]
enc_out = self.enc_embedding(x_enc, x_mark_enc)
```

### Decoder Input Construction (Forecasting)

Standard pattern from [exp/exp_long_term_forecasting.py](exp/exp_long_term_forecasting.py) (lines 49-51):

```python
# Create zero-filled future portion
dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
# Concatenate with label_len portion of ground truth
dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
# Shape: [B, label_len + pred_len, features]
```

This provides decoder with `label_len` real values as starting tokens, followed by zeros for prediction.

### Output Slicing Pattern

Models typically return full sequence but experiments extract forecast window:

```python
# In model forward
dec_out = ...  # Full decoder output: [B, label_len+pred_len, c_out]

# In experiment validation/testing (exp_long_term_forecasting.py line 63)
outputs = outputs[:, -self.args.pred_len:, :]  # Extract forecast portion
```

## Testing, Checkpointing & Debugging

**Metrics**: [utils/metrics.py](utils/metrics.py) - `metric()` function computes MSE, MAE, RMSE, MAPE, MSPE

**Early stopping**: [utils/tools.py](utils/tools.py) `EarlyStopping` class:

- Monitors validation loss with patience (default: 3 epochs)
- Saves best checkpoint to `./checkpoints/{setting}/checkpoint.pth`
- Load with: `torch.load(path)` then `model.load_state_dict(state_dict)`

**Device handling** (configurable in [run.py](run.py)):

- CUDA: `--use_gpu --gpu_type cuda --gpu 0` (default on Linux/Windows with NVIDIA GPU)
- MPS (Apple Silicon): `--use_gpu --gpu_type mps`
- Multi-GPU: `--use_multi_gpu --devices 0,1,2,3`
- CPU: `--use_gpu False`

**Windows-specific notes**:

- Use forward slashes `/` or escaped backslashes `\\` in paths
- PowerShell users: wrap path arguments with quotes if they contain spaces
- CUDA device selection: use `--gpu 0` instead of `export CUDA_VISIBLE_DEVICES=0`

**Debugging tips**:

1. Add `print(args)` to see all parsed arguments (done in [run.py](run.py) line 195)
2. Check tensor shapes with `print(x.shape)` at critical points
3. Use `--itr 1` to run single iteration during debugging
4. Set `--train_epochs 1` for quick validation of training loop
5. Test with small `--batch_size 8` to catch memory issues early

**Common error patterns**:

- Shape mismatch: Verify `enc_in`/`dec_in`/`c_out` match your dataset channels
- NaN losses: Check learning rate, may need to reduce from default 0.0001
- OOM errors: Reduce `batch_size`, `d_model`, or `d_ff`

## Common Pitfalls & Solutions

1. **Output shape mismatch in forecasting**:

   - Models must return `[B, pred_len, c_out]` after slicing
   - See [models/TimesNet.py](models/TimesNet.py) lines 137-145 for proper slicing pattern
   - Wrong: returning `[B, seq_len+pred_len, c_out]` directly
   - Correct: `return dec_out[:, -self.pred_len:, :]`

2. **Channel dimension errors**:

   - `enc_in`, `dec_in`, `c_out` must match dataset dimensions
   - ETT datasets: all three should be 7 (7 features)
   - Univariate (features='S'): all should be 1
   - Check dataset with: `df.shape[1] - 1` (subtract date column)

3. **Large Time Series Models (Chronos, Moirai, etc.)**:

   - Always use `--task_name zero_shot_forecast`
   - Always use `--is_training 0` (no training, inference only)
   - See [exp/exp_zero_shot_forecasting.py](exp/exp_zero_shot_forecasting.py)
   - Different models have different context length limits

4. **Mamba model dependencies**:

   - Requires separate install: `pip install mamba_ssm`
   - **CUDA-version specific** - must match your CUDA version
   - Model loaded conditionally in [exp/exp_basic.py](exp/exp_basic.py) lines 60-63
   - Won't work without proper installation

5. **Script execution errors**:

   - Scripts use relative paths: run from repository root
   - Ensure `./dataset/` exists with required data
   - Check CUDA device IDs in scripts: `export CUDA_VISIBLE_DEVICES=0`
   - **On Windows**: Scripts are bash-based. Either:
     - Use Git Bash or WSL to run `.sh` files directly
     - Extract individual `python -u run.py ...` commands from scripts and run in PowerShell/CMD
     - Remove `export CUDA_VISIBLE_DEVICES=X` lines (Windows uses different GPU selection)

6. **Data path confusion**:

   - `root_path`: directory containing data file
   - `data_path`: filename within root_path
   - Example: `--root_path ./dataset/ETT-small/ --data_path ETTh1.csv`
   - Not: `--root_path ./dataset/ --data_path ETT-small/ETTh1.csv`

7. **Label_len vs pred_len**:
   - `label_len`: decoder start tokens (typically seq_len/2)
   - `pred_len`: actual forecast horizon you want
   - Only matters for encoder-decoder models (Transformer, Autoformer, etc.)
   - Linear models often ignore label_len

## Repository Conventions & Best Practices

**Fixed random seed**: `fix_seed = 2021` in [run.py](run.py) line 15 for reproducibility across experiments

**Don't modify core experiment loops**: Training/validation/testing loops in [exp/](exp/) classes are standardized. To customize:

- Override methods in a new Exp class, or
- Add model-specific behavior in the model's `forward()` method, or
- Use task-specific methods (`forecast()`, `imputation()`, etc.)

**Script organization pattern**:

```
scripts/{task_name}/{dataset}_script/{ModelName}_{dataset}.sh
```

Examples:

- `scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh`
- `scripts/imputation/Weather_script/Autoformer_Weather.sh`

**Model file naming**: Use PascalCase matching paper name:

- Correct: `TimesNet.py`, `FEDformer.py`, `iTransformer.py`
- Wrong: `times_net.py`, `fedformer.py`

**Import pattern in [models/**init**.py](models/**init**.py)**:

```python
from .TimesNet import Model as TimesNet
from .FEDformer import Model as FEDformer
```

Then register in [exp/exp_basic.py](exp/exp_basic.py) `model_dict`.

**Arguments pattern**: All CLI args are lowercase with underscores:

- `--task_name`, `--seq_len`, `--pred_len`
- Accessed as: `args.task_name`, `args.seq_len`, etc.

**Dependencies**: Core requirements in [requirements.txt](requirements.txt):

- PyTorch 2.5.1
- Standard scientific stack (numpy, pandas, scikit-learn)
- Transformers library for LTSMs
- Optional: mamba_ssm (for Mamba model)

**Docker support**: See [docker-compose.yml](docker-compose.yml) for containerized setup
