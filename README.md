# vLLM-ERNIE

vLLM plugin for ERNIE models, one package per model under
`src/vllm_ernie/models/`. It currently serves ERNIE 5.5 (`Ernie5_5_MoeForCausalLM`:
MLA + HCA + DSA, mHC, MoE), with MTP and DSpark speculative decoding and the
`ernie5` reasoning / tool-call parsers. It runs on the released vLLM
**0.32.0** wheel, unmodified.

## Install

```bash
uv venv --python 3.12 ernie5-venv && source ernie5-venv/bin/activate
uv pip install vllm==0.32.0
git clone https://github.com/baidu/vLLM-ERNIE.git
uv pip install -e vLLM-ERNIE --no-deps
```

The plugin is loaded through the `vllm.general_plugins` entry point in every
vLLM process. `VLLM_PLUGINS=""` disables it (the architectures then no longer
resolve).

## Serve

One node with 4 GPUs: attention DP4 (VHA needs attention TP=1) + EP4.

### FP8 + MTP

32x32 block FP8 with ue8m0 scales (loaded as MXFP8 on sm100), 82 GiB of
weights per GPU:

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=3600
vllm serve baidu/ERNIE-5.5-Flash-Base \
    --host 0.0.0.0 --port 8000 \
    --data-parallel-size 4 --enable-expert-parallel \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

### BF16 + MTP

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=3600
vllm serve baidu/ERNIE-5.5-Flash-Base-BF16 \
    --host 0.0.0.0 --port 8000 \
    --data-parallel-size 4 --enable-expert-parallel \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --max-model-len 65536 --max-num-seqs 16 \
    --max-num-batched-tokens 8192 --max-cudagraph-capture-size 512
```

### DSpark instead of MTP

> **Note:** the DSpark draft checkpoint is coming soon.

Replace the MTP `--speculative-config` with the DSpark draft

FP8 + DSpark (256K context; for 1M see below):

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=3600
vllm serve baidu/ERNIE-5.5-Flash-Base \
    --host 0.0.0.0 --port 8000 \
    --data-parallel-size 4 --enable-expert-parallel \
    --speculative-config '{"method":"dspark","model":"/path/to/dspark_checkpoint","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}' \
    --max-model-len 262144
```

BF16 + DSpark (64K context, like BF16 + MTP):

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=3600
vllm serve baidu/ERNIE-5.5-Flash-Base-BF16 \
    --host 0.0.0.0 --port 8000 \
    --data-parallel-size 4 --enable-expert-parallel \
    --speculative-config '{"method":"dspark","model":"/path/to/dspark_checkpoint","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}' \
    --gpu-memory-utilization 0.95 --max-model-len 65536 --max-num-seqs 16 \
    --max-num-batched-tokens 8192 --max-cudagraph-capture-size 512
```

### 1M context (FP8)

A 1M-token prefill needs extra temporary memory, so lower
`--gpu-memory-utilization` from the default 0.92 to 0.85. The values are for
GPUs with 184 GiB of memory.

FP8 + MTP:

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=3600
vllm serve baidu/ERNIE-5.5-Flash-Base \
    --host 0.0.0.0 --port 8000 \
    --data-parallel-size 4 --enable-expert-parallel \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --gpu-memory-utilization 0.85
```

FP8 + DSpark (draft checkpoint coming soon):

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=3600
vllm serve baidu/ERNIE-5.5-Flash-Base \
    --host 0.0.0.0 --port 8000 \
    --data-parallel-size 4 --enable-expert-parallel \
    --speculative-config '{"method":"dspark","model":"/path/to/dspark_checkpoint","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}' \
    --gpu-memory-utilization 0.85 --max-num-seqs 128
```

### Reasoning / tool-call parsers

> **Note:** the chat checkpoint is coming soon.

To enable the `ernie5` reasoning and tool-call parsers, append these flags to any `vllm serve` command above:

```bash
    --reasoning-parser ernie5 \
    --tool-call-parser ernie5 --enable-auto-tool-choice
```

### Request

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"1+1=?"}]}'
```

## Test

Needs a GPU.

```bash
python -m pytest tests -q
```
