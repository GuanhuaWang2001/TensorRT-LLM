#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Kimi K3 MMLU accuracy check on 8 NVIDIA Blackwell GPUs (tep8) — modeled on
# run_gsm8k_kimi_k3.sbatch (the proven eval launch) to validate breakable
# prefill CUDA graph (BCG) correctness.
#
# Runs trtllm-eval mmlu twice in one allocation — BCG OFF (baseline) then BCG
# ON — so correctness is judged by "accuracy matches baseline". Uses the
# checked-in eval settings (eval_extra_llm_options.yaml) with tep8 parallelism;
# NO KIMI_K3_FP8_WEIGHT_READ (that forces the DeepGEMM fp8_gemm_nt JIT path,
# which is unrelated to BCG and can fail to load a runtime in some containers).
#
#   sbatch examples/kimi_k3/run_mmlu_eval_kimi_k3.sbatch \
#       --image /path/to/tensorrt-llm-container.sqsh
#
# Optional: --model PATH (default shared kimi-k3), --num_samples N (default
# 1000; 0 = full MMLU), --num_fewshot N (default 5), --mmlu_data PATH (default
# the workspace copy). Provide partition/account/job-name on the sbatch line.
#
#SBATCH --job-name=kimi-k3-mmlu-eval
#SBATCH --partition=batch
#SBATCH --account=${account}
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=05:00:00
#SBATCH --output=kimi-k3-mmlu-eval-%j.log

set -euo pipefail

usage() {
    echo "Usage: sbatch $0 --image PATH [--model PATH] [--num_samples N] [--num_fewshot N] [--mmlu_data PATH]"
}

MODEL="/lustre/share/coreai_comparch_inferencex/models/kimi-k3"
CONTAINER_IMAGE=""
NUM_SAMPLES=1000
NUM_FEWSHOT=5
MMLU_DATA="/lustre/fsw/coreai_comparch_trtllm/guawang/datasets/mmlu/data"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|--image|--num_samples|--num_fewshot|--mmlu_data)
            [[ $# -ge 2 ]] || { echo "error: $1 requires a value" >&2; usage >&2; exit 2; }
            VALUE=$2; OPT=$1; shift 2 ;;
        --model=*|--image=*|--num_samples=*|--num_fewshot=*|--mmlu_data=*)
            VALUE=${1#*=}; OPT=${1%%=*}; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    case "$OPT" in
        --model) MODEL=$VALUE ;;
        --image) CONTAINER_IMAGE=$VALUE ;;
        --num_samples) NUM_SAMPLES=$VALUE ;;
        --num_fewshot) NUM_FEWSHOT=$VALUE ;;
        --mmlu_data) MMLU_DATA=$VALUE ;;
    esac
done

[[ -n "$CONTAINER_IMAGE" ]] || { echo "error: --image is required" >&2; usage >&2; exit 2; }
[[ -e "$MODEL" ]] || { echo "error: model path does not exist: $MODEL" >&2; exit 2; }
[[ -e "$CONTAINER_IMAGE" ]] || { echo "error: image path does not exist: $CONTAINER_IMAGE" >&2; exit 2; }
[[ -d "$MMLU_DATA/test" && -d "$MMLU_DATA/dev" ]] || { echo "error: MMLU data not at $MMLU_DATA (need dev/ and test/)" >&2; exit 2; }

REPO=${REPO:-$SLURM_SUBMIT_DIR}
WORK_DIR=$SLURM_SUBMIT_DIR/kimi-k3-mmlu-eval-$SLURM_JOB_ID
mkdir -p "$WORK_DIR"

# Extra LLM options: tep8 (attention TP8, no attention DP) + EP8 MoE, derived
# from examples/kimi_k3/eval_extra_llm_options.yaml. The two runs differ ONLY
# in prefill_cuda_graph_backend. The BCG run captures prefill buckets up to
# 8192 so MMLU few-shot prompts route through the breakable graph.
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

CONFIG_BASE=$WORK_DIR/eval_config_nobcg.yaml
CONFIG_BCG=$WORK_DIR/eval_config_bcg.yaml
{ echo "$COMMON_YAML"; echo "prefill_cuda_graph_backend: disabled"; } > "$CONFIG_BASE"
{ echo "$COMMON_YAML"
  echo "prefill_cuda_graph_backend: breakable"
  echo "prefill_capture_num_tokens: [256, 512, 1024, 2048, 3072, 4096, 8192]"; } > "$CONFIG_BCG"

MOUNTS="$REPO:$REPO:rw,$MODEL:$MODEL:ro,$WORK_DIR:$WORK_DIR:rw,$MMLU_DATA:$MMLU_DATA:ro"

# One eval = one fresh MPI job over all 8 ranks (fresh LLM load + teardown).
# Launch pattern mirrors run_gsm8k_kimi_k3.sbatch: invoke python3 with the
# trtllm-eval script path directly (the venv console-script shebang hardcodes
# /workspace/tekit, which is not mounted here), and node-local /tmp caches.
run_eval() {
    local tag=$1 config=$2
    echo "===== MMLU eval [$tag]: config=$config, num_samples=$NUM_SAMPLES, num_fewshot=$NUM_FEWSHOT ====="
    local samples_flag=""
    [[ "$NUM_SAMPLES" -gt 0 ]] && samples_flag="--num_samples $NUM_SAMPLES"
    srun --mpi=pmix --kill-on-bad-exit=1 \
        --container-image="$CONTAINER_IMAGE" \
        --container-mount-home \
        --container-mounts="$MOUNTS" \
        --output="$WORK_DIR/mmlu_${tag}.log" \
        bash -c "
            set -x
            ulimit -n 65536
            export TRITON_CACHE_DIR=/tmp/triton-cache-rank\${SLURM_PROCID:-0}
            export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer-rank\${SLURM_PROCID:-0}
            export HF_MODULES_CACHE=/tmp/hf-modules-rank\${SLURM_PROCID:-0}
            mkdir -p \"\$TRITON_CACHE_DIR\"
            export PATH=\"$REPO/.venv-3.12/bin:\$PATH\"
            export PYTHONPATH=\"$REPO\${PYTHONPATH:+:\$PYTHONPATH}\"
            export TLLM_EVAL_PARTIAL_SCORES_EVERY=\"\${TLLM_EVAL_PARTIAL_SCORES_EVERY:-100}\"

            exec '$REPO/tensorrt_llm/llmapi/trtllm-llmapi-launch' python3 \
                '$REPO/.venv-3.12/bin/trtllm-eval' \
                --model \"$MODEL\" \
                --backend pytorch \
                --tp_size 8 \
                --max_batch_size 16 \
                --max_seq_len 8192 \
                --max_num_tokens 8192 \
                --kv_cache_free_gpu_memory_fraction 0.25 \
                --disable_kv_cache_reuse \
                --trust_remote_code \
                --config '$config' \
                mmlu \
                --dataset_path '$MMLU_DATA' \
                $samples_flag \
                --num_fewshot $NUM_FEWSHOT \
                --output_dir '$WORK_DIR/mmlu_${tag}_out'
        "
    echo "===== [$tag] done; log: $WORK_DIR/mmlu_${tag}.log ====="
}

run_eval nobcg "$CONFIG_BASE"
run_eval bcg   "$CONFIG_BCG"

echo "MMLU eval complete. Accuracy comparison:"
echo "  baseline (BCG off): $WORK_DIR/mmlu_nobcg.log"
echo "  BCG on            : $WORK_DIR/mmlu_bcg.log"
grep -iE "accuracy|average|mmlu|weighted" "$WORK_DIR/mmlu_nobcg.log" "$WORK_DIR/mmlu_bcg.log" 2>/dev/null | tail -40 || true
