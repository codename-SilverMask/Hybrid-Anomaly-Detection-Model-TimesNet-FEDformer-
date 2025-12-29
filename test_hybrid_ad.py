"""
Quick test script to verify Hybrid_AD model implementation
Tests model instantiation, forward pass, and parameter counting
"""
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Hybrid_AD import Model as Hybrid_AD
from argparse import Namespace


def test_hybrid_ad():
    """Test Hybrid_AD model instantiation and forward pass"""
    
    print("=" * 80)
    print("Testing Hybrid_AD Model Implementation")
    print("=" * 80)
    
    # Create mock configs for MSL dataset
    configs = Namespace(
        task_name='anomaly_detection',
        seq_len=100,
        pred_len=0,
        label_len=48,
        enc_in=55,
        dec_in=55,
        c_out=55,
        d_model=64,
        d_ff=128,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        top_k=5,
        num_kernels=6,
        embed='timeF',
        freq='h',
        dropout=0.1,
        moving_avg=25,
        factor=1,
        distil=True,
        activation='gelu'
    )
    
    print("\n1. Instantiating Hybrid_AD model...")
    try:
        model = Hybrid_AD(configs)
        print("✓ Model instantiated successfully")
    except Exception as e:
        print(f"✗ Failed to instantiate model: {e}")
        return False
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    timesnet_params = sum(p.numel() for p in model.timesnet.parameters())
    fedformer_params = sum(p.numel() for p in model.fedformer.parameters())
    gating_params = sum(p.numel() for p in model.gating_network.parameters())
    
    print(f"\n2. Parameter counts:")
    print(f"   - TimesNet:      {timesnet_params:,}")
    print(f"   - FEDformer:     {fedformer_params:,}")
    print(f"   - Gating Net:    {gating_params:,}")
    print(f"   - Total:         {total_params:,}")
    print(f"   - Trainable:     {trainable_params:,}")
    
    # Test forward pass with dummy data
    print("\n3. Testing forward pass...")
    batch_size = 32
    x_enc = torch.randn(batch_size, configs.seq_len, configs.enc_in)
    
    try:
        model.eval()
        with torch.no_grad():
            output = model(x_enc, None, None, None)
        
        expected_shape = (batch_size, configs.seq_len, configs.c_out)
        actual_shape = tuple(output.shape)
        
        if actual_shape == expected_shape:
            print(f"✓ Forward pass successful")
            print(f"   Input shape:  {tuple(x_enc.shape)}")
            print(f"   Output shape: {actual_shape}")
        else:
            print(f"✗ Output shape mismatch!")
            print(f"   Expected: {expected_shape}")
            print(f"   Got:      {actual_shape}")
            return False
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test gating mechanism
    print("\n4. Testing gating mechanism...")
    try:
        with torch.no_grad():
            g = model.gating_network(x_enc)
        
        print(f"   Gating weights shape: {tuple(g.shape)}")
        print(f"   Gating weights range: [{g.min().item():.4f}, {g.max().item():.4f}]")
        print(f"   Gating weights mean:  {g.mean().item():.4f}")
        
        # Check if weights are in [0, 1]
        if g.min() >= 0 and g.max() <= 1:
            print("✓ Gating weights correctly bounded in [0, 1]")
        else:
            print("✗ Gating weights outside [0, 1] range!")
            return False
            
    except Exception as e:
        print(f"✗ Gating mechanism test failed: {e}")
        return False
    
    # Test anomaly_detection method directly
    print("\n5. Testing anomaly_detection method...")
    try:
        with torch.no_grad():
            output = model.anomaly_detection(x_enc)
        print(f"✓ anomaly_detection method works")
        print(f"   Output shape: {tuple(output.shape)}")
    except Exception as e:
        print(f"✗ anomaly_detection method failed: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run training: bash scripts/anomaly_detection/MSL/Hybrid_AD.sh")
    print("2. Check results in: ./checkpoints/ and result_anomaly_detection.txt")
    print("3. Compare with baseline TimesNet and FEDformer results")
    
    return True


if __name__ == '__main__':
    success = test_hybrid_ad()
    sys.exit(0 if success else 1)
