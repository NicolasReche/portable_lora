#!/bin/bash
set -e

# 1. Activate .venv if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d ".venv" ]; then
        echo "[*] Activating .venv..."
        source .venv/bin/activate
    else
        echo "[!] Error: .venv directory not found."
        exit 1
    fi
fi

python3 - << 'PYCHECK'
import torch, transformers, peft, trl, bitsandbytes
print(f"    - PyTorch:      {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
print(f"    - Transformers: {transformers.__version__}")
print(f"    - PEFT:         {peft.__version__}")
print(f"    - TRL:          {trl.__version__}")
print(f"    - BitsAndBytes: {bitsandbytes.__version__}")
PYCHECK

mkdir -p logs outputs predictions results models data config jobs scripts

python3 -m py_compile scripts/rl_peft/reward_function.py scripts/rl_peft/train_rl_model.py 2>/dev/null && echo "    [✓] Core RL scripts: PASS" || echo "    [!] Warning: Syntax check failed on RL scripts."


echo " Environment verified "
