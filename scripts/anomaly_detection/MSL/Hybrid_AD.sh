export CUDA_VISIBLE_DEVICES=0

# Hybrid_AD on MSL - Anomaly Detection
# This script runs the Hybrid Anomaly Detection model that fuses TimesNet and FEDformer
# Using learnable gating mechanism for adaptive weighting

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/MSL \
  --model_id MSL_Hybrid_AD \
  --model Hybrid_AD \
  --data MSL \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 128 \
  --e_layers 2 \
  --enc_in 55 \
  --c_out 55 \
  --top_k 5 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --train_epochs 5 \
  --patience 3 \
  --des 'Hybrid_AD_MSL'
