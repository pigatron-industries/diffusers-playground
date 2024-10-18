import argparse
from safetensors import safe_open
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Print safe tensors')
    parser.add_argument('--model', type=str, help='Path to the model')
    args = parser.parse_args()

    with safe_open(args.model, 'pt') as f:
        metadata = f.metadata()
        for k in f.keys():
            print(f"{k}: {f.get_tensor(k).shape}")