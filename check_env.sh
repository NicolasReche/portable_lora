#!/bin/bash
set -e

echo "=========================================================="
echo "          RUNNING PRE-FLIGHT ENVIRONMENT CHECK            "
echo "=========================================================="

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
echo "[✓] Python Binary: $(which python3)"

# 2. Check Essential Deep Learning Packages
echo "[*] Verifying core framework versions..."
python3 - << 'PYCHECK'
import torch, transformers, peft, trl, bitsandbytes
print(f"    - PyTorch:      {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
print(f"    - Transformers: {transformers.__version__}")
print(f"    - PEFT:         {peft.__version__}")
print(f"    - TRL:          {trl.__version__}")
print(f"    - BitsAndBytes: {bitsandbytes.__version__}")
PYCHECK

# 3. Create Project Directory Tree
echo "[*] Ensuring required directory structure exists..."
mkdir -p logs outputs predictions results models data config jobs scripts

# 4. Check Python Syntax of Critical Scripts
echo "[*] Verifying syntax of training and reward scripts..."
python3 -m py_compile scripts/rl_peft/reward_function.py scripts/rl_peft/train_rl_model.py 2>/dev/null && echo "    [✓] Core RL scripts: PASS" || echo "    [!] Warning: Syntax check failed on RL scripts."

echo "=========================================================="
echo " Environment verified! Ready to launch pipelines:         "
echo "   1. Reproduction baselines : ./run_repro.sh             "
echo "   2. RL exploration (V1-V4) : ./run_rl.sh                "
echo "   3. Benchmark analysis     : ./analyse.sh               "
echo "=========================================================="
