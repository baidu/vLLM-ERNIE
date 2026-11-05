<p align="center">
  <img src="assets/logo.png" alt="Baidu ERNIE" width="600">
</p>

# vLLM-ERNIE

vLLM plugin for ERNIE models, one package per model under
`src/vllm_ernie/models/`. It currently serves ERNIE 5.5
(`Ernie5_5_MoeForCausalLM`: HCA + MLA-DSA, mHC, MoE), with ECHO (MTP) and DSpark
speculative decoding and the `ernie5` reasoning / tool-call parsers. It runs on
the released vLLM **0.32.0** wheel, unmodified, on NVIDIA GPUs only.

## Supported Models

| Model | Weights | Use |
|---|---|---|
| ERNIE-5.5-Flash-Base | [FP8](https://huggingface.co/baidu/ERNIE-5.5-Flash-Base) / [BF16](https://huggingface.co/baidu/ERNIE-5.5-Flash-Base-BF16) | Base model, completions. Released |
| ERNIE-5.5-Flash | FP8 / FP4 / BF16 | Chat, reasoning, tool calling, DSpark. Coming soon |

Set `VLLM_USE_MODELSCOPE=True` to download from
[ModelScope](https://www.modelscope.cn/models/baidu/ERNIE-5.5-Flash-Base); the
repo names are the same on both hubs.

## Model Introduction

ERNIE-5.5-Flash is a sparse Mixture-of-Experts language model from Baidu, with
280B total and 14B activated parameters per token (excluding the 0.36B ECHO
layer), pre-trained from scratch on 33.5T tokens with a context of up to 128K.
Inference supports up to 1,048,576 tokens.



- **Hybrid HCA + MLA-DSA Attention.** 37 of the 43 layers use Heavily
  Compressed Attention (one cached entry per 128 tokens plus a 128-token
  sliding window); 6 layers keep full-context access through MLA with DSA
  sparse attention (top-2048 tokens).
- **Gated Attention and Virtual Head Attention (VHA).** Every attention layer
  adds a low-rank output gate, a low-rank post-mix across heads and QK-norm.
- **Multi-Router Latent MoE.** 42 MoE layers, each with 512 routed experts
  (top-10) and 1 shared expert in a 2048-d latent space, with SiTU-GLU
  activations.
- **mHC.** Manifold-constrained hyper-connections (expansion rate 4) replace
  the residual stream.
- **MultiMax Output Head.** with a learnable piece-wise temperature.
- **ECHO.** The final layer, trained to predict future tokens, doubles as a
  built-in draft; vLLM runs it through its MTP path.

## Install

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install vllm==0.32.0
uv pip install git+https://github.com/baidu/vLLM-ERNIE.git
```

The plugin is loaded through the `vllm.general_plugins` entry point in every
vLLM process. `VLLM_PLUGINS=""` disables it. If `vllm serve` reports
`Model architectures ['Ernie5_5_MoeForCausalLM'] are not supported`, the plugin
is not installed in the environment that runs `vllm`, or `VLLM_PLUGINS` is set
and does not include `ernie`.

## Serve

Attention runs data parallel and the MoE expert parallel
(`--data-parallel-size N --enable-expert-parallel`); attention tensor
parallelism (`--tensor-parallel-size` 2, 4 or 8) also works, see Configuration
Tips. The commands below are for one 4-GPU node and use the 128K training
context (`--max-model-len 131072`), with ECHO (MTP) speculative decoding on.
The smallest GPU counts that hold the weights:

| GPU memory | FP8 (~280 GB weights) | BF16 (~560 GB weights) |
|---|---|---|
| 288 GB | 2 GPUs | 4 GPUs |
| 192 GB | 4 GPUs | 8 GPUs (or 2 nodes × 4 GPUs) |
| 80 GB | 8 GPUs | 2 nodes × 8 GPUs |

The FP8 commands pass `--moe-backend humming`, which supports SiTU with FP8
weights on Blackwell-architecture GPUs. On Hopper-architecture GPUs, drop it
and keep vLLM's default MoE kernel. On 80 GB GPUs, serve FP8 on 8 GPUs with
`--data-parallel-size 8` and `--gpu-memory-utilization 0.93`; up to
`--max-model-len 262144` fits, 1M does not.

The commands bind `0.0.0.0` without authentication; add `--api-key <key>` or
bind `127.0.0.1` unless the network is trusted.

### FP8

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=3600
vllm serve baidu/ERNIE-5.5-Flash-Base \
    --host 0.0.0.0 \
    --port 8000 \
    --data-parallel-size 4 \
    --enable-expert-parallel \
    --moe-backend humming \
    --kv-cache-dtype fp8 \
    --max-model-len 131072 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

Drop `--speculative-config` to serve without a draft. ECHO drafts 2 steps
here; 1 or 3 also work.

### BF16

BF16 runs on two 4-GPU nodes (DP8 + EP8). Set `HEAD_IP` to the first node's
address and `IFACE_NAME` to the network interface between the nodes, then run
on the first node:

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export GLOO_SOCKET_IFNAME=${IFACE_NAME}
export NCCL_SOCKET_IFNAME=${IFACE_NAME}
vllm serve baidu/ERNIE-5.5-Flash-Base-BF16 \
    --host 0.0.0.0 \
    --port 8000 \
    --data-parallel-size 8 \
    --data-parallel-size-local 4 \
    --data-parallel-address ${HEAD_IP} \
    --data-parallel-hybrid-lb \
    --enable-expert-parallel \
    --kv-cache-dtype fp8 \
    --max-model-len 131072 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

The second node runs the same command with `--data-parallel-start-rank 4`.
One 4-GPU node also holds the BF16 weights, but leaves little room for the KV
cache: there, pass `--gpu-memory-utilization 0.95`, `--max-model-len 65536`,
`--max-num-seqs 16`, `--max-num-batched-tokens 8192` and
`--max-cudagraph-capture-size 512` instead.

### DSpark (ERNIE-5.5-Flash, Coming Soon)

The ERNIE-5.5-Flash model bundles a DSpark draft; the Base models
carry none. Replace the ECHO `--speculative-config` with:

```bash
    --speculative-config '{"method":"dspark","draft_sample_method":"probabilistic"}'
```

The draft length comes from the model's block size, so the config leaves
out `num_speculative_tokens`. vLLM's default `draft_sample_method` is `greedy`;
the command uses `probabilistic`.

### 1M Context

The model allows 1,048,576 tokens. Pass `--max-model-len 1048576`; a 1M-token
prefill needs tens of GB of temporary memory on top, so lower
`--gpu-memory-utilization` from the default 0.92 to 0.85, and lower
`--max-num-seqs` so each rank's KV cache holds whole 1M-token requests. The
values are for GPUs with 184 GiB of memory; 80 GB GPUs do not fit 1M:

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=3600
vllm serve baidu/ERNIE-5.5-Flash-Base \
    --host 0.0.0.0 \
    --port 8000 \
    --data-parallel-size 4 \
    --enable-expert-parallel \
    --moe-backend humming \
    --kv-cache-dtype fp8 \
    --max-model-len 1048576 \
    --gpu-memory-utilization 0.85 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

### Prefill-Decode (PD) Disaggregation

One prefill (P) and one decode (D) 4-GPU node, each DP4 + EP4, move the KV
cache over RDMA with `MooncakeConnector`, behind vllm-router 0.1.16. Install
the Mooncake transfer engine on both nodes and the router on any host that
reaches both:

```bash
uv pip install mooncake-transfer-engine-cuda13   # P and D
uv pip install vllm-router==0.1.16               # router host
```

Set these on both nodes. `IB_DEVICES` lists the InfiniBand HCAs only
(`ibdev2netdev` shows them):

```bash
export P_IP=10.0.0.1
export D_IP=10.0.0.2
export IB_DEVICES=mlx5_0,mlx5_1,mlx5_4,mlx5_5
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=1800        # keep idle router connections
export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=600  # fail a stuck KV transfer
```

P node (D is the same with `D_IP` and `kv_consumer`):

```bash
VLLM_HOST_IP=${P_IP} vllm serve baidu/ERNIE-5.5-Flash-Base \
    --host ${P_IP} \
    --port 8000 \
    --data-parallel-size 4 \
    --enable-expert-parallel \
    --moe-backend humming \
    --kv-cache-dtype fp8 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 128 \
    --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_connector_extra_config":{"device_name":"'${IB_DEVICES}'"}}'
```

Router:

```bash
vllm-router \
    --host 0.0.0.0 \
    --port 8080 \
    --policy round_robin \
    --vllm-pd-disaggregation \
    --kv-connector mooncake \
    --prefill http://${P_IP}:8000 8998 \
    --decode http://${D_IP}:8000 \
    --intra-node-data-parallel-size 4
```

P serves the Mooncake bootstrap on port 8998. With speculative decoding, pass
the same `--speculative-config` to both nodes. The `model` field of each request must
match the served model name; otherwise the request fails mid-stream instead
of returning a 404.

### Reasoning and Tool Calling (ERNIE-5.5-Flash, Coming Soon)

For ERNIE-5.5-Flash (the Base models have no chat training), add to the
`vllm serve` command:

```bash
    --reasoning-parser ernie5 \
    --tool-call-parser ernie5 \
    --enable-auto-tool-choice
```

Thinking is returned in `reasoning_content`, the answer in `content` and tool
calls in `message.tool_calls`. Thinking is on by default; pass
`"chat_template_kwargs": {"enable_thinking": false}` in a request to turn it
off.

### KV Cache Offload

Long shared prefixes can be kept outside GPU memory and loaded back instead of
recomputed. With `SimpleCPUOffloadConnector`, `cpu_bytes_to_use_per_rank` is
pinned per DP rank, so the rank count times this value must fit in host RAM;
for example, 128 GiB per rank with 4 ranks:

```bash
    --kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use_per_rank":137438953472}}'
```

## Request

The Base models continue text, so send prompts to `/v1/completions`; for
few-shot use, put the examples in the prompt and stop at the end of one answer.
The model's `generation_config.json` sets `do_sample=false`,
`temperature=1.0`, `top_p=0.95`. vLLM applies `temperature` and `top_p` from it
but not `do_sample`, so pass `"temperature": 0` for greedy decoding.

```bash
curl http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "baidu/ERNIE-5.5-Flash-Base",
        "prompt": "Question: What is 15% of 240?\nAnswer:",
        "max_tokens": 64,
        "temperature": 0,
        "stop": ["\n"]
    }'
```

The completion text in the response is ` 36`. For the BF16 weights, set
`"model"` to `baidu/ERNIE-5.5-Flash-Base-BF16`. The tokenizer prepends `<|ERNIE_MASK|>`
(id 2) to completions prompts itself; do not add it to the prompt.

## Configuration Tips

- **Parallelism.** Data parallel attention with `--data-parallel-size` and
  `--enable-expert-parallel` holds the most KV cache and gives the highest
  throughput. Attention tensor parallelism (`--tensor-parallel-size`, which
  must divide 8) lowers latency at low concurrency; the single KV head is
  replicated on every rank of the TP group, so it holds less KV cache.
- **MoE kernel.** The FP8 commands pass `--moe-backend humming`, which
  supports SiTU with FP8 weights. On Hopper-architecture GPUs, keep vLLM's
  default.
- **Long context.** The commands pass `--max-model-len 131072`, the 128K
  training length. The model allows 1,048,576: pass
  `--max-model-len 1048576`, lower `--max-num-seqs` so each rank's KV cache
  holds whole 1M-token requests, and pass `--gpu-memory-utilization 0.85`.
  On 80 GB GPUs 1M does not fit; they reach 262,144 at
  `--gpu-memory-utilization 0.93`.
- **KV cache.** Every command passes `--kv-cache-dtype fp8`, which stores the
  KV cache in FP8; `auto` keeps it in BF16.

## License

Apache License 2.0, see [LICENSE](LICENSE).
