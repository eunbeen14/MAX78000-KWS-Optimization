"""
=============================================================================
[MCU Weights Extraction Utility: Key Inspector]
Description: 
A utility script to inspect and extract the layer names (keys) from a 
Quantization-Aware Training (QAT) checkpoint (.pth.tar).
Ensures the PyTorch layer names match the expected C-array generation targets.
=============================================================================
"""
import argparse
import torch

def main(checkpoint_path):
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']

    print("\n=== Model Checkpoint Layer Keys ===")
    for key in state_dict.keys():
        print(key)
    print("===================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inspect checkpoint keys.")
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        default='qat_best-q.pth.tar',
        help='Path to the quantized PyTorch checkpoint file'
    )
    args = parser.parse_args()
    main(args.checkpoint)