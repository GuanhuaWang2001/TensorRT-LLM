#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Kimi K3 BCG-vs-baseline serving perf on 8 NVIDIA Blackwell GPUs (tep8).
#
# Brings up trtllm-serve (tep8, overlap scheduler ON) twice in one allocation —
# once with the breakable prefill CUDA graph OFF (baseline) and once ON — and
# drives the InferenceX bench_serving client (co-located on the head node) at
# concurrency 1 and 16 with ISL=2048 / OSL=1024, saving TTFT + throughput JSON.
#
# Launch/config mirror the proven run_gsm8k_kimi_k3.sbatch: python3 <script>
# (bypasses the /workspace/tekit console-script shebang) and NO
# KIMI_K3_FP8_WEIGHT_READ (that forces the DeepGEMM fp8_gemm_nt JIT path). So
# these are BF16-weight-read numbers; the BCG-vs-baseline delta is still valid.
#
#   sbatch examples/kimi_k3/run_bcg_perf_kimi_k3.sbatch \
#       --image /path/to/tensorrt-llm-container.sqsh
#
# Optional: --model PATH, --isl N (2048), --osl N (1024),
# --concurrencies "1 16". Provide partition/account/job-name on the sbatch line.
#
#SBATCH --job-name=kimi-k3-bcg-perf
#SBATCH --partition=batch
#SBATCH --account=${account}
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=05:00:00
#SBATCH --output=kimi-k3-bcg-perf-%j.log

set -euo pipefail

usage() {
    echo "Usage: sbatch $0 --image PATH [--model PATH] [--isl N] [--osl N] [--concurrencies \"1 16\"]"
}

MODEL="/lustre/share/coreai_comparch_inferencex/models/kimi-k3"
CONTAINER_IMAGE=""
ISL=2048
OSL=1024
CONCURRENCIES="1 16"
PORT=8000
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|--image|--isl|--osl|--concurrencies)
            [[ $# -ge 2 ]] || { echo "error: $1 requires a value" >&2; usage >&2; exit 2; }
            VALUE=$2; OPT=$1; shift 2 ;;
        --model=*|--image=*|--isl=*|--osl=*|--concurrencies=*)
            VALUE=${1#*=}; OPT=${1%%=*}; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
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

MAX_SEQ_LEN=$((ISL + OSL + 128))
REPO=${REPO:-$SLURM_SUBMIT_DIR}
BENCH_SERVING_DIR=${BENCH_SERVING_DIR:-$(dirname "$REPO")/InferenceX/utils/bench_serving}
[[ -e "$BENCH_SERVING_DIR/benchmark_serving.py" ]] || { echo "error: InferenceX bench_serving not found at $BENCH_SERVING_DIR" >&2; exit 2; }
WORK_DIR=$SLURM_SUBMIT_DIR/kimi-k3-bcg-perf-$SLURM_JOB_ID
mkdir -p "$WORK_DIR"

# Shared serve config (tep8, overlap scheduler ON), from eval_extra_llm_options.
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
  echo "prefill_capture_num_tokens: [512, 1024, 2048, 3072, 4096, 6144, 8192]"; } > "$CONFIG_BCG"

MOUNTS="$REPO:$REPO:rw,$MODEL:$MODEL:ro,$WORK_DIR:$WORK_DIR:rw,$BENCH_SERVING_DIR:$BENCH_SERVING_DIR:ro"
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | awk 'NR == 1')

SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

start_server() {
    local tag=$1 config=$2
    echo "===== starting server [$tag] (config=$config) ====="
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
            exec '$REPO/tensorrt_llm/llmapi/trtllm-llmapi-launch' python3 \
                '$REPO/.venv-3.12/bin/trtllm-serve' \
                \"$MODEL\" \
                --backend pytorch \
                --host 0.0.0.0 --port $PORT \
                --tp_size 8 \
                --max_batch_size 16 \
                --max_num_tokens 8192 \
                --max_seq_len $MAX_SEQ_LEN \
                --trust_remote_code \
                --config '$config'
        " &
    SERVER_PID=$!
    local deadline=$((SECONDS + 1500))
    until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null || (( SECONDS >= deadline )); then
            echo "error: server [$tag] not healthy within 25 min (see $WORK_DIR/serve_${tag}.log)" >&2
            scancel "$SLURM_JOB_ID"; sleep 60; exit 1
        fi
        sleep 15
    done
    echo "===== server [$tag] healthy ====="
}

stop_server() {
    echo "===== stopping server ====="
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    sleep 20
}

run_client() {
    local tag=$1 conc=$2 nprompts=$3 result=$4
    srun --overlap --mpi=none --kill-on-bad-exit=1 --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
        --container-image="$CONTAINER_IMAGE" \
        --container-mount-home \
        --container-mounts="$MOUNTS" \
        --output="$WORK_DIR/client_${tag}_c${conc}.log" \
        bash -c "
            set -x
            for mpi_env in \$(env | awk -F= '/^(PMI|PMIX|MPI|OMPI|SLURM)_/{print \$1}'); do unset \"\$mpi_env\"; done
            export PATH=\"$REPO/.venv-3.12/bin:\$PATH\"
            python3 '$BENCH_SERVING_DIR/benchmark_serving.py' \
                --backend openai \
                --model \"$MODEL\" \
                --host 127.0.0.1 --port $PORT \
                --endpoint /v1/completions \
                --dataset-name random \
                --random-input-len $ISL \
                --random-output-len $OSL \
                --ignore-eos \
                --trust-remote-code \
                --num-prompts $nprompts \
                --max-concurrency $conc \
                --percentile-metrics 'ttft,tpot,itl,e2el' \
                ${result:+--save-result --result-dir '$WORK_DIR' --result-filename '$result'}
        "
}

for pair in "nobcg:$CONFIG_BASE" "bcg:$CONFIG_BCG"; do
    tag=${pair%%:*}; config=${pair#*:}
    start_server "$tag" "$config"
    for conc in $CONCURRENCIES; do
        echo "=== [$tag] warmup c=$conc ==="
        run_client "$tag" "$conc" "$conc" ""
        echo "=== [$tag] measured c=$conc ==="
        run_client "$tag" "$conc" "$((conc * 10 < 20 ? 20 : conc * 10))" "perf_${tag}_c${conc}.json"
    done
    stop_server
done

echo "===== BCG perf sweep complete; results in $WORK_DIR ====="
grep -hiE "Successful requests|Request throughput|Output token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Mean E2EL" \
    "$WORK_DIR"/client_*_c*.log 2>/dev/null || true
