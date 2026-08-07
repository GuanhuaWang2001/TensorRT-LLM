#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Kimi K3 BCG-vs-baseline prefill nsys capture on 8 Blackwell GPUs (tep8), c=1.
#
# For each of {baseline (BCG off), BCG on}: bring up trtllm-serve, then send a
# single ISL=2048 request and a single ISL=512 request (OSL=4, so each is one
# prefill + a few decode iters). Rank 0 is wrapped with nsys -c cudaProfilerApi;
# the PyExecutor's TLLM_PROFILE_START_STOP brackets the first real iterations
# with cudaProfilerStart/Stop, so the trace is a tiny window around the two
# prefills. BCG prefill buckets are captured during server startup warmup, so
# the profiled requests replay (no capture cost in the trace).
#
#   sbatch examples/kimi_k3/run_bcg_nsys_kimi_k3.sbatch \
#       --image /path/to/tensorrt-llm-container.sqsh
#
# Provide partition/account/job-name on the sbatch line.
#
#SBATCH --job-name=kimi-k3-bcg-nsys
#SBATCH --partition=batch
#SBATCH --account=${account}
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=03:00:00
#SBATCH --output=kimi-k3-bcg-nsys-%j.log

set -euo pipefail

usage() { echo "Usage: sbatch $0 --image PATH [--model PATH]"; }

MODEL="/lustre/share/coreai_comparch_inferencex/models/kimi-k3"
CONTAINER_IMAGE=""
OSL=4
PORT=8000
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|--image) [[ $# -ge 2 ]] || { echo "error: $1 requires a value" >&2; exit 2; }; VALUE=$2; OPT=$1; shift 2 ;;
        --model=*|--image=*) VALUE=${1#*=}; OPT=${1%%=*}; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    case "$OPT" in --model) MODEL=$VALUE ;; --image) CONTAINER_IMAGE=$VALUE ;; esac
done

[[ -n "$CONTAINER_IMAGE" ]] || { echo "error: --image is required" >&2; usage >&2; exit 2; }
[[ -e "$MODEL" ]] || { echo "error: model path does not exist: $MODEL" >&2; exit 2; }
[[ -e "$CONTAINER_IMAGE" ]] || { echo "error: image path does not exist: $CONTAINER_IMAGE" >&2; exit 2; }

REPO=${REPO:-$SLURM_SUBMIT_DIR}
BENCH_SERVING_DIR=${BENCH_SERVING_DIR:-$(dirname "$REPO")/InferenceX/utils/bench_serving}
[[ -e "$BENCH_SERVING_DIR/benchmark_serving.py" ]] || { echo "error: InferenceX bench_serving not found at $BENCH_SERVING_DIR" >&2; exit 2; }
WORK_DIR=$SLURM_SUBMIT_DIR/kimi-k3-bcg-nsys-$SLURM_JOB_ID
mkdir -p "$WORK_DIR"

COMMON_YAML='enable_attention_dp: false
moe_expert_parallel_size: 8
disable_overlap_scheduler: false
enable_chunked_prefill: true
cuda_graph_config:
  enable_padding: true
  max_batch_size: 16
moe_config:
  max_num_tokens: 33024
  use_low_precision_moe_combine: true
kv_cache_config:
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.25
  tokens_per_block: 64'

CONFIG_BASE=$WORK_DIR/serve_config_nobcg.yaml
CONFIG_BCG=$WORK_DIR/serve_config_bcg.yaml
{ echo "$COMMON_YAML"; echo "prefill_cuda_graph_backend: disabled"; } > "$CONFIG_BASE"
{ echo "$COMMON_YAML"
  echo "prefill_cuda_graph_backend: breakable"
  echo "prefill_capture_num_tokens: [512, 1024, 2048]"; } > "$CONFIG_BCG"

MOUNTS="$REPO:$REPO:rw,$MODEL:$MODEL:ro,$WORK_DIR:$WORK_DIR:rw,$BENCH_SERVING_DIR:$BENCH_SERVING_DIR:ro"
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | awk 'NR == 1')

SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

start_server() {
    local tag=$1 config=$2
    echo "===== starting nsys server [$tag] ====="
    srun --mpi=pmix --kill-on-bad-exit=1 \
        --container-image="$CONTAINER_IMAGE" \
        --container-mount-home \
        --container-mounts="$MOUNTS" \
        --output="$WORK_DIR/serve_${tag}.log" \
        bash -c "
            set -x
            ulimit -n 65536
            export TRITON_CACHE_DIR=/tmp/triton-cache-rank\${SLURM_PROCID:-0}
            export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer-rank\${SLURM_PROCID:-0}
            export HF_MODULES_CACHE=/tmp/hf-modules-rank\${SLURM_PROCID:-0}
            mkdir -p \"\$TRITON_CACHE_DIR\"
            unset TRTLLM_FORCE_COMM_METHOD
            export PATH=\"$REPO/.venv-3.12/bin:\$PATH\"
            export PYTHONPATH=\"$REPO\${PYTHONPATH:+:\$PYTHONPATH}\"
            # Bracket the first real iterations (both profiled prefills + a few
            # decode steps) with cudaProfilerStart/Stop on every rank.
            export TLLM_PROFILE_START_STOP=1-12
            # nsys-wrap rank 0 only (keeps the capture small; its head-shard
            # prefill kernels are representative).
            PROFILER=
            if [ \"\${SLURM_PROCID:-0}\" = \"0\" ]; then
                PROFILER=\"nsys profile -c cudaProfilerApi --capture-range-end=stop --force-overwrite=true --trace=cuda,nvtx -o $WORK_DIR/nsys_${tag}\"
            fi
            exec '$REPO/tensorrt_llm/llmapi/trtllm-llmapi-launch' \$PROFILER python3 \
                '$REPO/.venv-3.12/bin/trtllm-serve' \
                \"$MODEL\" \
                --backend pytorch \
                --host 0.0.0.0 --port $PORT \
                --tp_size 8 \
                --max_batch_size 16 \
                --max_num_tokens 8192 \
                --max_seq_len 3200 \
                --trust_remote_code \
                --config '$config'
        " &
    SERVER_PID=$!
    local deadline=$((SECONDS + 1500))
    until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null || (( SECONDS >= deadline )); then
            echo "error: server [$tag] not healthy (see $WORK_DIR/serve_${tag}.log)" >&2
            scancel "$SLURM_JOB_ID"; sleep 60; exit 1
        fi
        sleep 15
    done
    echo "===== server [$tag] healthy ====="
}

stop_server() {
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    sleep 20
}

send_one() {
    local tag=$1 isl=$2
    srun --overlap --mpi=none --kill-on-bad-exit=1 --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
        --container-image="$CONTAINER_IMAGE" \
        --container-mount-home \
        --container-mounts="$MOUNTS" \
        --output="$WORK_DIR/client_${tag}_isl${isl}.log" \
        bash -c "
            set -x
            for mpi_env in \$(env | awk -F= '/^(PMI|PMIX|MPI|OMPI|SLURM)_/{print \$1}'); do unset \"\$mpi_env\"; done
            export PATH=\"$REPO/.venv-3.12/bin:\$PATH\"
            python3 '$BENCH_SERVING_DIR/benchmark_serving.py' \
                --backend openai --model \"$MODEL\" \
                --host 127.0.0.1 --port $PORT --endpoint /v1/completions \
                --dataset-name random --random-input-len $isl --random-output-len $OSL \
                --ignore-eos --trust-remote-code \
                --num-prompts 1 --max-concurrency 1
        "
}

for pair in "nobcg:$CONFIG_BASE" "bcg:$CONFIG_BCG"; do
    tag=${pair%%:*}; config=${pair#*:}
    start_server "$tag" "$config"
    # Requests are concurrency-1 and sequential, so ~ (1 prefill + OSL decode)
    # iterations each. The two targets land at iters 1 and ~1+(1+OSL).
    echo "=== [$tag] profiled request ISL=2048 (prefill ~iter 1) ==="
    send_one "$tag" 2048
    echo "=== [$tag] profiled request ISL=512 (prefill ~iter $((2 + OSL))) ==="
    send_one "$tag" 512
    # Flush: tiny requests to advance the iteration counter past the profiler
    # stop (TLLM_PROFILE_START_STOP=1-12) so cudaProfilerStop fires and nsys
    # finalizes the report. Their (tiny) prefills are harmless in the trace.
    echo "=== [$tag] flush requests to close the profiler window ==="
    for i in 1 2 3 4; do send_one "$tag" 64; done
    sleep 10
    stop_server
done

echo "===== nsys capture complete; reports in $WORK_DIR ====="
ls -l "$WORK_DIR"/nsys_*.nsys-rep 2>/dev/null || echo "(no .nsys-rep produced — check serve logs for the profiler start/stop iteration lines)"
