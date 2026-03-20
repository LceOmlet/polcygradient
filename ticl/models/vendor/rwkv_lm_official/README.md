Vendored upstream source from `BlinkDL/RWKV-LM` for RWKV-7 integration work.

Upstream snapshot:
- repo: `https://github.com/BlinkDL/RWKV-LM`
- local clone source used for vendoring: `/tmp/RWKV-LM-official`
- date vendored: `2026-03-20`

Files copied verbatim for source-of-truth reference:
- `RWKV-v7/rwkv_v7_demo_fast.py`
- `RWKV-v7/rwkv_v7_demo_rnn.py`
- `RWKV-v7/cuda/wkv7.cu`
- `RWKV-v7/cuda/wkv7_op.cpp`
- `RWKV-v7/cuda/wkv7s.cu`
- `RWKV-v7/cuda/wkv7s_op.cpp`
- `RWKV-v7/train_temp/src/model.py`
- `RWKV-v7/train_temp/cuda/wkv7_cuda.cu`
- `RWKV-v7/train_temp/cuda/wkv7_cuda_fp32.cu`
- `RWKV-v7/train_temp/cuda/wkv7_op.cpp`
- `RWKV-v7/train_temp/cuda/wkv7_op_fp32.cpp`
- `RWKV-v7/train_temp/cuda/rwkv7_clampw.cu`
- `RWKV-v7/train_temp/cuda/rwkv7_clampw.cpp`

Intent:
- keep official RWKV-7 training / inference code in-tree as the canonical reference
- build thin PFN adapters around these files instead of growing another handwritten RWKV core

Important limitation from upstream:
- upstream RWKV-7 splits efficient training and efficient recurrent inference across different files / kernels
- `RWKV-v7/train_temp/*` contains the training CUDA autograd path
- `RWKV-v7/rwkv_v7_demo_fast.py` and `RWKV-v7/rwkv_v7_demo_rnn.py` contain the recurrent stateful inference path
- there is no single upstream drop-in module that already matches the maintained `policy_step rollout` training API in this repository

