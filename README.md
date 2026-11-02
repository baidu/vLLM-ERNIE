# vLLM-ERNIE

vLLM plugin for ERNIE models, one package per model under
`src/vllm_ernie/models/`. It currently serves ERNIE 5.5 (`Ernie5_5_MoeForCausalLM`:
MLA + HCA + DSA, mHC, MoE), with MTP and DSpark speculative decoding and the
`ernie5` reasoning / tool-call parsers. It runs on the released vLLM
**0.32.0** wheel, unmodified.

## Supported models

| Model | Weights | Hugging Face | ModelScope |
|---|---|---|---|
| ERNIE-5.5-Flash-Base | FP8 | [baidu/ERNIE-5.5-Flash-Base](https://huggingface.co/baidu/ERNIE-5.5-Flash-Base) | [baidu/ERNIE-5.5-Flash-Base](https://www.modelscope.cn/models/baidu/ERNIE-5.5-Flash-Base) |
| ERNIE-5.5-Flash-Base-BF16 | BF16 | [baidu/ERNIE-5.5-Flash-Base-BF16](https://huggingface.co/baidu/ERNIE-5.5-Flash-Base-BF16) | [baidu/ERNIE-5.5-Flash-Base-BF16](https://www.modelscope.cn/models/baidu/ERNIE-5.5-Flash-Base-BF16) |
| ERNIE-5.5-Flash | BF16 / FP8 / FP4 | Coming soon | Coming soon |

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

128x128 block FP8 with ue8m0 scales, about 86 GiB per GPU after loading:

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=3600
vllm serve baidu/ERNIE-5.5-Flash-Base \
    --host 0.0.0.0 --port 8000 \
    --data-parallel-size 4 --enable-expert-parallel \
    --moe-backend humming --kv-cache-dtype fp8 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

### BF16 + MTP

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=3600
vllm serve baidu/ERNIE-5.5-Flash-Base-BF16 \
    --host 0.0.0.0 --port 8000 \
    --data-parallel-size 4 --enable-expert-parallel \
    --kv-cache-dtype fp8 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --gpu-memory-utilization 0.95 --max-model-len 65536 --max-num-seqs 16 \
    --max-num-batched-tokens 8192 --max-cudagraph-capture-size 512
```

### MXFP4 indexer

On SM100-family datacenter GPUs, select MXFP4 for the indexer's runtime Q and
K cache with the existing attention option:

```bash
vllm serve baidu/ERNIE-5.5-Flash-Base-BF16 \
    --data-parallel-size 4 --enable-expert-parallel \
    --kv-cache-dtype fp8 \
    --attention-config '{"indexer_kv_dtype":"mxfp4"}'
```

The MXFP4 indexer currently requires `--prefill-context-parallel-size 1` and
`--decode-context-parallel-size 1` (the defaults). Other PCP/DCP sizes are rejected
during indexer initialization.

### DSpark instead of MTP

> **Note:** the DSpark draft checkpoint is coming soon.

Replace the MTP `--speculative-config` with the DSpark draft

FP8 + DSpark (256K context; for 1M see below):

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=3600
vllm serve baidu/ERNIE-5.5-Flash-Base \
    --host 0.0.0.0 --port 8000 \
    --data-parallel-size 4 --enable-expert-parallel \
    --moe-backend humming --kv-cache-dtype fp8 \
    --speculative-config '{"method":"dspark","model":"/path/to/dspark_checkpoint","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}' \
    --max-model-len 262144
```

BF16 + DSpark (64K context, like BF16 + MTP):

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=3600
vllm serve baidu/ERNIE-5.5-Flash-Base-BF16 \
    --host 0.0.0.0 --port 8000 \
    --data-parallel-size 4 --enable-expert-parallel \
    --kv-cache-dtype fp8 \
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
    --moe-backend humming --kv-cache-dtype fp8 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --gpu-memory-utilization 0.85
```

FP8 + DSpark (draft checkpoint coming soon):

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=3600
vllm serve baidu/ERNIE-5.5-Flash-Base \
    --host 0.0.0.0 --port 8000 \
    --data-parallel-size 4 --enable-expert-parallel \
    --moe-backend humming --kv-cache-dtype fp8 \
    --speculative-config '{"method":"dspark","model":"/path/to/dspark_checkpoint","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}' \
    --gpu-memory-utilization 0.85 --max-num-seqs 128
```

### Reasoning / tool-call parsers

> **Note:** the chat checkpoint is coming soon.

The `ernie5` reasoning and tool-call parsers are for the chat checkpoint
(ERNIE-5.5-Flash); the Base checkpoints have no chat training. To enable them,
add these flags to its `vllm serve` command:

```bash
    --reasoning-parser ernie5 \
    --tool-call-parser ernie5 --enable-auto-tool-choice
```

### Request

The Base checkpoints continue text, so send prompts to `/v1/completions`:

```bash
curl http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model": "baidu/ERNIE-5.5-Flash-Base",
         "prompt": "Question: What is 15% of 240?\nAnswer:",
         "max_tokens": 64, "temperature": 0, "stop": ["\n"]}'
```

A correct answer starts with `36`. For the BF16 checkpoint, set `"model"` to
`baidu/ERNIE-5.5-Flash-Base-BF16`.

## Test

Needs a GPU.

```bash
python -m pytest tests -q
```
