"""
Pretrained Model Weights Utility for TopoGrapher.

Verifies local model checkpoint or guides downloading the official
pretrained TopoGrapherNet weights checkpoint.
"""

import os
import sys
import hashlib

CHECKPOINT_DIR = "d:/Workspace/checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "topographer_weights.pt")


def verify_checkpoint(path: str = CHECKPOINT_PATH):
    if not os.path.exists(path):
        print(f"[-] Checkpoint not found at: {path}")
        print("[*] TopoGrapher will initialize with native procedural architectural weights,")
        print("    or you can train your own weights via: python scripts/train.py")
        return False

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"[+] Found TopoGrapher checkpoint: {path} ({size_mb:.2f} MB)")
    return True


if __name__ == "__main__":
    verify_checkpoint()
