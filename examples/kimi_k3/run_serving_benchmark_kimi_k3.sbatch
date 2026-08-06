#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Kimi K3 serving benchmark on 16 NVIDIA Blackwell GPUs.
#
# Brings up trtllm-serve with attention TP16 + expert parallelism 16 and the
# breakable prefill CUDA graph (prefill_cuda_graph_backend: breakable) with the
# overlap scheduler on, then drives the InferenceX bench_serving client
# (InferenceX/utils/bench_serving/benchmark_serving.py) co-located on the head
# node with synthetic random 8K-input/1K-output requests at concurrency 1.
# One warmup request is discarded before 10 measured requests.
#
# Submit from the repository root of a built TensorRT-LLM checkout (see the
# README Prerequisites section):
#
#   sbatch examples/kimi_k3/run_serving_benchmark_kimi_k3.sbatch \
#       --image /path/to/tensorrt-llm-container.sqsh
#
# Optional arguments: --model PATH (default the shared kimi-k3 checkpoint),
# --isl N (default 8192), --osl N (default 1024), --concurrencies "1". REPO
# defaults to the submit directory (override by exporting REPO) and must
# contain the built TensorRT-LLM checkout. The InferenceX repo is expected as a
# sibling of REPO (override with BENCH_SERVING_DIR). Results and server/client
# logs go to kimi-k3-serving-benchmark-<job-id>/ in the submission directory.
#
# The --comment below requests an idle-GPU-reaper exemption on clusters that
# cancel jobs whose GPUs sit idle: server initialization and weight loading
# take 10-15 minutes before timed benchmark traffic starts. A command-line
# --comment overrides this directive; repeat it verbatim if you pass one.
#
#SBATCH --job-name=kimi-k3-serving-benchmark
#SBATCH --partition=batch
#SBATCH --account=${account}
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=05:00:00
#SBATCH --output=kimi-k3-serving-benchmark-%j.log
#SBATCH --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"benchmarking","description":"Kimi K3 16-GPU server initialization and weight loading takes 10-15 minutes; the server is kept alive after the sweep for iterative testing"}}'

set -euo pipefail

usage() {
    echo "Usage: sbatch $0 --image PATH [--model PATH] [--isl N] [--osl N] [--concurrencies \"1\"]"
}

MODEL="/lustre/share/coreai_comparch_inferencex/models/kimi-k3"
CONTAINER_IMAGE=""
ISL=8192
OSL=1024
CONCURRENCIES="1"
PORT=8000
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|--image|--isl|--osl|--concurrencies)
            [[ $# -ge 2 ]] || { echo "error: $1 requires a value" >&2; usage >&2; exit 2; }
            VALUE=$2
            OPT=$1
            shift 2
            ;;
        --model=*|--image=*|--isl=*|--osl=*|--concurrencies=*)
            VALUE=${1#*=}
            OPT=${1%%=*}
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    case "$OPT" in
        --model) MODEL=$VALUE ;;
        --image) CONTAINER_IMAGE=$VALUE ;;
        --isl) ISL=$VALUE ;;
        --osl) OSL=$VALUE ;;
        --concurrencies) CONCURRENCIES=$VALUE ;;
    esac
done

[[ -n "$CONTAINER_IMAGE" ]] || { echo "error: --image is required" >&2; usage >&2; exit 2; }
[[ -e "$MODEL" ]] || { echo "error: model path does not exist: $MODEL" >&2; exit 2; }
[[ -e "$CONTAINER_IMAGE" ]] || { echo "error: image path does not exist: $CONTAINER_IMAGE" >&2; exit 2; }
[[ "$ISL" =~ ^[0-9]+$ && "$OSL" =~ ^[0-9]+$ ]] || { echo "error: --isl/--osl must be integers" >&2; exit 2; }
for concurrency in $CONCURRENCIES; do
    [[ "$concurrency" =~ ^[0-9]+$ ]] || { echo "error: bad concurrency: $concurrency" >&2; exit 2; }
done

# 9344 at the default 8192/1024, matching the verified runs exactly.
MAX_SEQ_LEN=$((ISL + OSL + 128))

REPO=${REPO:-$SLURM_SUBMIT_DIR}
# InferenceX bench_serving client (co-located on the head node). Expected as a
# sibling of the TensorRT-LLM checkout unless BENCH_SERVING_DIR overrides it.
BENCH_SERVING_DIR=${BENCH_SERVING_DIR:-$(dirname "$REPO")/InferenceX/utils/bench_serving}
[[ -e "$BENCH_SERVING_DIR/benchmark_serving.py" ]] || {
    echo "error: InferenceX bench_serving not found at $BENCH_SERVING_DIR (set BENCH_SERVING_DIR)" >&2
    exit 2
}
WORK_DIR=$SLURM_SUBMIT_DIR/kimi-k3-serving-benchmark-$SLURM_JOB_ID
mkdir -p "$WORK_DIR"

# Serving config. tep16 (attention TP16, no attention DP) + EP16 MoE, the
# breakable prefill CUDA graph, and the overlap scheduler. free_gpu_memory
# _fraction 0.25 leaves headroom for the KDA/SSM state cache on the weights.
CONFIG_FILE=$WORK_DIR/serve_config.yaml
cat > "$CONFIG_FILE" <<'EOF'
enable_attention_dp: false
moe_expert_parallel_size: 16
disable_overlap_scheduler: false
cuda_graph_config:
  max_batch_size: 16
prefill_cuda_graph_backend: breakable
prefill_capture_num_tokens: [8192]
stream_interval: 10
enable_chunked_prefill: true
kv_cache_config:
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.25
EOF

# Shared warm caches, persistent across runs: without a pre-populated
# FLASHINFER_CUBIN_DIR, ranks on egress-less compute nodes can hang fetching
# cubins during the TRTLLM-Gen FMHA warmup.
CACHE_DIR=${CACHE_DIR:-$SLURM_SUBMIT_DIR/.kimi_k3_cache}
mkdir -p "$CACHE_DIR"

MOUNTS="$REPO:$REPO:rw,$MODEL:$MODEL:ro,$WORK_DIR:$WORK_DIR:rw,$CACHE_DIR:$CACHE_DIR:rw,$BENCH_SERVING_DIR:$BENCH_SERVING_DIR:ro"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | awk 'NR == 1')

SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

# Server: all 16 ranks; rank 0 (this node) binds the HTTP endpoint.
# --kill-on-bad-exit=1: any failed rank kills the whole step immediately.
srun --mpi=pmix --kill-on-bad-exit=1 \
    --container-image="$CONTAINER_IMAGE" \
    --container-mount-home \
    --container-mounts="$MOUNTS" \
    --output="$WORK_DIR/serve.log" \
    bash -c "
        set -x
        ulimit -n 65536

        # Node-local Triton cache: ~/.triton on shared NFS races across
        # ranks during autotune JIT (stale file handles).
        export TRITON_CACHE_DIR=/tmp/triton-cache-rank\${SLURM_PROCID:-0}
        export TRTLLM_DG_CACHE_DIR=/tmp/deep-gemm-${SLURM_JOB_ID}-rank\${SLURM_PROCID:-0}
        mkdir -p \"\$TRITON_CACHE_DIR\"
        mkdir -p \"\$TRTLLM_DG_CACHE_DIR\"
        # Shared warm caches (see the CACHE_DIR comment above).
        export FLASHINFER_WORKSPACE_BASE=$CACHE_DIR/fi_workspace
        export FLASHINFER_CUBIN_DIR=$CACHE_DIR/fi_cubins
        export HF_HOME=\${HF_HOME:-$CACHE_DIR/hf_home}
        mkdir -p \"\$FLASHINFER_WORKSPACE_BASE\" \"\$FLASHINFER_CUBIN_DIR\" \"\$HF_HOME\"

        # Exercise CommunicationFactory AUTO selection rather than inheriting
        # a caller-side forced communication strategy.
        unset TRTLLM_FORCE_COMM_METHOD

        # FP8 weight-read: fused gate_up on, KDA glue kept BF16.
        export KIMI_K3_FP8_WEIGHT_READ=1
        export KIMI_K3_FP8_WEIGHT_READ_KDA=0
        export KIMI_K3_FP8_WEIGHT_READ_GATE_UP=1
        # In-place venv of the built checkout (build_wheel.py creates .venv-3.12).
        export PATH=\"$REPO/.venv-3.12/bin:\$PATH\"

        exec '$REPO/tensorrt_llm/llmapi/trtllm-llmapi-launch' trtllm-serve \
            \"$MODEL\" \
            --backend pytorch \
            --host 0.0.0.0 \
            --port $PORT \
            --tp_size 16 \
            --max_batch_size 16 \
            --max_num_tokens 8192 \
            --max_seq_len $MAX_SEQ_LEN \
            --trust_remote_code \
            --config '$CONFIG_FILE'
    " &
SERVER_PID=$!

# Wait up to 25 min for health; on timeout or server death, cancel the job
# so the allocation never sits on idle GPUs.
deadline=$((SECONDS + 1500))
until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null || (( SECONDS >= deadline )); then
        echo "error: server not healthy within 25 min (see $WORK_DIR/serve.log); cancelling job" >&2
        scancel "$SLURM_JOB_ID"
        sleep 120
        exit 1
    fi
    sleep 15
done
echo "Server is healthy; starting benchmark sweep (see $WORK_DIR/client.log)."

# Client on the head node, alongside the server (no separate allocation). The
# InferenceX bench_serving random dataset is generated client-side so the
# server sees exactly ISL input tokens; --ignore-eos fixes output length at OSL.
# --mpi=none so the single-rank client never joins the server step's PMIx job.
srun --overlap --mpi=none --kill-on-bad-exit=1 --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    --container-image="$CONTAINER_IMAGE" \
    --container-mount-home \
    --container-mounts="$MOUNTS" \
    --output="$WORK_DIR/client.log" \
    bash -c "
        set -x
        # Prevent the client-only process from inheriting the server step's
        # Slurm/PMIx environment and spuriously calling MPI_Init_thread.
        for mpi_env in \$(env | awk -F= '/^(PMI|PMIX|MPI|OMPI|SLURM)_/{print \$1}'); do
            unset \"\$mpi_env\"
        done
        export PATH=\"$REPO/.venv-3.12/bin:\$PATH\"

        run_client() {
            python3 '$BENCH_SERVING_DIR/benchmark_serving.py' \
                --backend openai \
                --model \"$MODEL\" \
                --host 127.0.0.1 \
                --port $PORT \
                --endpoint /v1/completions \
                --dataset-name random \
                --random-input-len $ISL \
                --random-output-len $OSL \
                --ignore-eos \
                --trust-remote-code \
                \"\$@\"
        }

        for concurrency in $CONCURRENCIES; do
            echo \"=== Warmup: concurrency \$concurrency ===\"
            run_client --num-prompts \$concurrency --max-concurrency \$concurrency

            echo \"=== Measured: concurrency \$concurrency ===\"
            run_client \
                --num-prompts \$((concurrency * 10)) \
                --max-concurrency \$concurrency \
                --save-result \
                --result-dir '$WORK_DIR' \
                --result-filename \"concurrency_\${concurrency}.json\"
        done
    "

echo "Benchmark complete; results in $WORK_DIR"

# Keep the server + allocation alive for iterative testing instead of
# releasing after a single sweep. Export KEEP_ALIVE=<seconds> before sbatch to
# hold the live server (drive more client sweeps with
# 'srun --overlap --mpi=none --nodes=1 --ntasks=1 -w <head-node> ...' into this
# job, or 'scancel $SLURM_JOB_ID' to release early). Default 0 = tear down now.
KEEP_ALIVE=${KEEP_ALIVE:-0}
if (( KEEP_ALIVE > 0 )); then
    echo "Keeping server + allocation alive for ${KEEP_ALIVE}s on $HEAD_NODE:$PORT (KEEP_ALIVE); scancel $SLURM_JOB_ID to release early."
    sleep "$KEEP_ALIVE" || true
fi

kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
