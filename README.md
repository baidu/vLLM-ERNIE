# vLLM-ERNIE

vLLM plugin for ERNIE models, one package per model under
`src/vllm_ernie/models/`. It currently serves ERNIE 5.5 (`Ernie5_5_MoeForCausalLM`:
HCA + MLA-DSA, mHC, MoE), with ECHO (MTP) and DSpark speculative decoding and
the `ernie5` reasoning / tool-call parsers. It runs on the released vLLM
**0.32.0** wheel, unmodified, on NVIDIA GPUs only.

## Supported Models

| Model | Weights | Use |
|---|---|---|
| ERNIE-5.5-Flash-Base | [FP8](https://huggingface.co/baidu/ERNIE-5.5-Flash-Base) / [BF16](https://huggingface.co/baidu/ERNIE-5.5-Flash-Base-BF16) | Base model, completions. Released |
| ERNIE-5.5-Flash | FP8 / FP4 / BF16 | Chat, reasoning, tool calling, DSpark. Coming soon |

Set `VLLM_USE_MODELSCOPE=True` to download from
[ModelScope](https://www.modelscope.cn/models/baidu/ERNIE-5.5-Flash-Base); the
repo names are the same on both hubs.

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
(`--data-parallel-size N --enable-expert-parallel`); the plugin does not
implement attention tensor parallelism yet. The commands below are for one
4-GPU node and use the 128K training context (`--max-model-len 131072`), with
ECHO (MTP) speculative decoding on. The smallest GPU counts that hold the
weights:

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
    --host 0.0.0.0 --port 8000 \
    --data-parallel-size 4 --enable-expert-parallel \
    --moe-backend humming --kv-cache-dtype fp8 --max-model-len 131072 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

Drop `--speculative-config` to serve without a draft. The commands draft 2 steps,
the same default as SGLang; ECHO was trained for one future token and drafts
more steps by running again, so 1 or 3 also work.

### BF16

BF16 runs on two 4-GPU nodes (DP8 + EP8). Set `HEAD_IP` to the first node's
address and `IFACE_NAME` to the network interface between the nodes, then run
on the first node:

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export GLOO_SOCKET_IFNAME=${IFACE_NAME} NCCL_SOCKET_IFNAME=${IFACE_NAME}
vllm serve baidu/ERNIE-5.5-Flash-Base-BF16 \
    --host 0.0.0.0 --port 8000 \
    --data-parallel-size 8 --data-parallel-size-local 4 \
    --data-parallel-address ${HEAD_IP} --data-parallel-hybrid-lb \
    --enable-expert-parallel --kv-cache-dtype fp8 --max-model-len 131072 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

The second node runs the same command with `--data-parallel-start-rank 4`.
One 4-GPU node also holds the BF16 weights, but leaves little room for the KV
cache: there, pass `--gpu-memory-utilization 0.95 --max-model-len 65536
--max-num-seqs 16 --max-num-batched-tokens 8192 --max-cudagraph-capture-size 512`
instead.

### DSpark (ERNIE-5.5-Flash, Coming Soon)

The ERNIE-5.5-Flash checkpoint bundles a DSpark draft; the Base checkpoints
carry none. Replace the ECHO `--speculative-config` with:

```bash
    --speculative-config '{"method":"dspark","draft_sample_method":"probabilistic"}'
```

The draft length comes from the checkpoint's block size, so the config leaves
out `num_speculative_tokens`. vLLM's default `draft_sample_method` is `greedy`;
the command uses `probabilistic`.

### 1M Context (FP8)

The checkpoint allows 1,048,576 tokens. Pass `--max-model-len 1048576`; a
1M-token prefill needs tens of GB of temporary memory on top, so lower
`--gpu-memory-utilization` from the default 0.92 to 0.85, and lower
`--max-num-seqs` so each rank's KV cache holds whole 1M-token requests. The
values are for GPUs with 184 GiB of memory; 80 GB GPUs do not fit 1M:

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=3600
vllm serve baidu/ERNIE-5.5-Flash-Base \
    --host 0.0.0.0 --port 8000 \
    --data-parallel-size 4 --enable-expert-parallel \
    --moe-backend humming --kv-cache-dtype fp8 --max-model-len 1048576 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --gpu-memory-utilization 0.85
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
export P_IP=10.0.0.1 D_IP=10.0.0.2 IB_DEVICES=mlx5_0,mlx5_1,mlx5_4,mlx5_5
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=1800        # keep idle router connections
export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=600  # fail a stuck KV transfer
```

P node (D is the same with `D_IP` and `kv_consumer`):

```bash
VLLM_HOST_IP=${P_IP} vllm serve baidu/ERNIE-5.5-Flash-Base \
    --host ${P_IP} --port 8000 \
    --data-parallel-size 4 --enable-expert-parallel \
    --moe-backend humming --kv-cache-dtype fp8 --max-model-len 131072 \
    --gpu-memory-utilization 0.85 --max-num-seqs 128 \
    --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer",
        "kv_connector_extra_config":{"device_name":"'${IB_DEVICES}'"}}'
```

Router:

```bash
vllm-router --host 0.0.0.0 --port 8080 --policy round_robin \
    --vllm-pd-disaggregation --kv-connector mooncake \
    --prefill http://${P_IP}:8000 8998 --decode http://${D_IP}:8000 \
    --intra-node-data-parallel-size 4
```

P serves the Mooncake bootstrap on port 8998. With speculative decoding, pass the same
`--speculative-config` to both nodes. The `model` field of each request must
match the served model name; otherwise the request fails mid-stream instead
of returning a 404.

### Reasoning and Tool Calling (ERNIE-5.5-Flash, Coming Soon)

The `ernie5` reasoning and tool-call parsers are for ERNIE-5.5-Flash; the Base
checkpoints have no chat, reasoning or tool-call training. Add to its
`vllm serve` command:

```bash
    --reasoning-parser ernie5 \
    --tool-call-parser ernie5 --enable-auto-tool-choice
```

Thinking lands in `reasoning_content` and the answer in `content`. Thinking is
on by default; turn it off for one request with
`"chat_template_kwargs": {"enable_thinking": false}` (the chat template reads
`enable_thinking`, not `thinking`). The tool-call parser reads the model's
`<ernie:tool_call>` blocks, types each argument from the tool's JSON schema,
and returns `message.tool_calls`; use it together with the reasoning parser,
since the model reasons before calling a tool.

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model": "baidu/ERNIE-5.5-Flash",
         "messages": [{"role": "user", "content": "What is the weather in Beijing?"}],
         "tools": [{"type": "function", "function": {
           "name": "get_weather",
           "description": "Get the current weather for a city",
           "parameters": {"type": "object",
                          "properties": {"city": {"type": "string"}},
                          "required": ["city"]}}}]}'
```

### KV Cache Offload

Long shared prefixes can be kept outside GPU memory and loaded back instead of
recomputed. With `SimpleCPUOffloadConnector`, `cpu_bytes_to_use_per_rank` is
pinned per DP rank, so the rank count times this value must fit in host RAM;
for example, 128 GiB per rank with 4 ranks:

```bash
    --kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both",
        "kv_connector_extra_config":{"cpu_bytes_to_use_per_rank":137438953472}}'
```

For a filesystem tier, point it at fast local NVMe: the KV blocks take about
280 KB per token on disk.

## Request

The Base checkpoints continue text, so send prompts to `/v1/completions`; for
few-shot use, put the examples in the prompt and stop at the end of one answer.
The checkpoint's `generation_config.json` sets `do_sample=false`,
`temperature=1.0`, `top_p=0.95`. vLLM applies `temperature` and `top_p` from it
but not `do_sample`, so pass `"temperature": 0` for greedy decoding.

```bash
curl http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model": "baidu/ERNIE-5.5-Flash-Base",
         "prompt": "Question: What is 15% of 240?\nAnswer:",
         "max_tokens": 64, "temperature": 0, "stop": ["\n"]}'
```

Example output (`choices[0].text`): ` 36`. For the BF16 checkpoint, set `"model"` to
`baidu/ERNIE-5.5-Flash-Base-BF16`. The tokenizer prepends `<|ERNIE_MASK|>`
(id 2) to completions prompts itself; do not add it to the prompt.

## Model: ERNIE 5.5 Flash

ERNIE-5.5-Flash is the efficient member of Baidu's ERNIE 5.5 family: a sparse
Mixture-of-Experts language model with 280B total and 14B activated parameters
per token, pre-trained from scratch on 33.5T tokens with a training context of
up to 128K. Fine-tuning and inference support a 1M-token context (1,048,576),
which is vLLM's default `--max-model-len` for this model; the commands above
pass the 128K training length instead.

- **ERNIE-5.5-Flash-Base** (FP8 and BF16) is taken at the end of pre-training,
  before any mid- or post-training. It is a starting point for domain
  adaptation, instruction tuning and reinforcement-learning research, and is
  served through `/v1/completions`.
- **ERNIE-5.5-Flash** (FP8 / FP4 / BF16, coming soon) adds reasoning, tool
  calling and a bundled DSpark draft.

Architecture, shared by all checkpoints:

- **Hybrid HCA + MLA-DSA attention.** 37 of the 43 layers use Heavily
  Compressed Attention (HCA: one cached entry per 128 tokens plus a 128-token
  sliding window, 512-dim heads). One layer in seven, 6 in total, keeps
  full-context access through MLA with DSA sparse attention (top-2048 tokens,
  256-dim heads).
- **Low-rank gated attention and Virtual Head Attention (VHA).** Every
  attention layer adds a low-rank output gate, a low-rank post-mix across heads
  and QK-norm.
- **Multi-Router Latent MoE.** 1 dense FFN layer plus 42 MoE layers, each with
  512 routed experts (top-10) and 1 shared expert in a 2048-d latent space,
  selected by several sigmoid routers, with SiTU-GLU expert activations.
- **mHC.** Manifold-constrained hyper-connections (expansion rate 4) replace
  the residual stream.
- **MultiMax output head** with a learnable piece-wise temperature.
- **ECHO.** The backbone's own final layer, trained to predict future tokens,
  doubles as a built-in draft; vLLM runs it through its MTP path.

Total parameters include the input embedding and the output head; activated
parameters count the output head but not the embedding lookup. Both exclude
the 0.36B ECHO parameters.

| Checkpoint | Size |
|---|---|
| FP8: 128x128 blocks, UE8M0 scales | ~280 GB |
| BF16 | ~560 GB |

The FP8 checkpoint keeps the embedding, the output head, the MoE routers, the
HCA compressor, the indexer weight projection, the VHA post-mix and the ECHO
projections in BF16, and the mHC parameters in FP32; every other linear layer
and the routed experts are FP8.

The checkpoints are released under the ERNIE 5.5 Flash Base Model License
Agreement v1.0 (the `LICENSE` file in each checkpoint); this plugin is Apache
2.0.

## Configuration Tips

- **Parallelism.** The plugin does not implement attention tensor
  parallelism yet; `--tensor-parallel-size` above 1 fails at startup. Scale
  with `--data-parallel-size` and `--enable-expert-parallel`.
- **MoE kernel.** The FP8 commands pass `--moe-backend humming`, which
  supports SiTU with FP8 weights. On Hopper-architecture GPUs, keep vLLM's
  default.
- **Long context.** The commands pass `--max-model-len 131072`, the 128K
  training length. The checkpoint allows 1,048,576: pass
  `--max-model-len 1048576`, lower `--max-num-seqs` so each rank's KV cache
  holds whole 1M-token requests, and pass `--gpu-memory-utilization 0.85`.
  On 80 GB GPUs 1M does not fit; they reach 262,144 at
  `--gpu-memory-utilization 0.93`.
- **KV cache.** Every command passes `--kv-cache-dtype fp8`, which stores the
  KV cache in FP8; `auto` keeps it in BF16.

## Development

Install from a clone in editable mode and run the unit tests (needs a GPU):

```bash
git clone https://github.com/baidu/vLLM-ERNIE.git && cd vLLM-ERNIE
uv pip install -e . --no-deps
python -m pytest tests -q
```

## License

Apache License 2.0, see [LICENSE](LICENSE).
