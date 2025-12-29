"""
Hybrid Anomaly Detection Model
Combines TimesNet and FEDformer using a learnable gating mechanism for parallel fusion.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import TimesNet, FEDformer


class GatingNetwork(nn.Module):
    """
    Learnable gating mechanism that outputs a scalar weight g ∈ [0, 1]
    Takes flattened input window and produces gating weight via small MLP + Sigmoid
    """
    def __init__(self, input_dim, hidden_dim=128):
        super(GatingNetwork, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Ensures output in [0, 1]
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, seq_len, enc_in] - input window
        Returns:
            g: [B, 1, 1] - gating weight per sample
        """
        B, T, C = x.shape
        x_flat = x.reshape(B, -1)  # [B, seq_len * enc_in]
        g = self.mlp(x_flat)  # [B, 1]
        g = g.unsqueeze(-1)  # [B, 1, 1] for broadcasting
        return g


class Model(nn.Module):
    """
    Hybrid Anomaly Detection Model
    Paper-inspired fusion: X_hybrid = g * X_TimesNet + (1-g) * X_FEDformer
    where g is a learnable gating weight from a small MLP
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Initialize TimesNet backbone
        self.timesnet = TimesNet.Model(configs)
        
        # Initialize FEDformer backbone  
        self.fedformer = FEDformer.Model(configs)
        
        # Learnable gating mechanism
        # Input dimension is seq_len * enc_in (flattened window)
        gate_input_dim = configs.seq_len * configs.enc_in
        self.gating_network = GatingNetwork(gate_input_dim, hidden_dim=128)
        
        print(f"Hybrid_AD initialized with:")
        print(f"  - TimesNet: {sum(p.numel() for p in self.timesnet.parameters())} params")
        print(f"  - FEDformer: {sum(p.numel() for p in self.fedformer.parameters())} params")
        print(f"  - Gating Network: {sum(p.numel() for p in self.gating_network.parameters())} params")
    
    def anomaly_detection(self, x_enc):
        """
        Anomaly detection via parallel fusion
        Args:
            x_enc: [B, seq_len, enc_in] - input time series
        Returns:
            dec_out: [B, seq_len, c_out] - reconstructed output
        """
        # Get reconstructions from both models
        timesnet_out = self.timesnet.anomaly_detection(x_enc)  # [B, seq_len, c_out]
        fedformer_out = self.fedformer.anomaly_detection(x_enc)  # [B, seq_len, c_out]
        
        # Compute gating weight from input
        g = self.gating_network(x_enc)  # [B, 1, 1]
        
        # Weighted fusion: X_hybrid = g * X_TimesNet + (1-g) * X_FEDformer
        dec_out = g * timesnet_out + (1 - g) * fedformer_out
        
        return dec_out
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forecasting via parallel fusion (if needed for other tasks)
        """
        timesnet_out = self.timesnet.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        fedformer_out = self.fedformer.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        g = self.gating_network(x_enc)
        dec_out = g * timesnet_out + (1 - g) * fedformer_out
        
        return dec_out
    
    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """
        Imputation via parallel fusion (if needed for other tasks)
        """
        timesnet_out = self.timesnet.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        fedformer_out = self.fedformer.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        
        g = self.gating_network(x_enc)
        dec_out = g * timesnet_out + (1 - g) * fedformer_out
        
        return dec_out
    
    def classification(self, x_enc, x_mark_enc):
        """
        Classification via parallel fusion (if needed for other tasks)
        """
        timesnet_out = self.timesnet.classification(x_enc, x_mark_enc)
        fedformer_out = self.fedformer.classification(x_enc, x_mark_enc)
        
        g = self.gating_network(x_enc)
        g = g.squeeze(-1)  # [B, 1] for classification output
        dec_out = g * timesnet_out + (1 - g) * fedformer_out
        
        return dec_out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Main forward pass - delegates to task-specific methods
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
