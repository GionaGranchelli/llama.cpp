#!/usr/bin/env bash
set -Eeuo pipefail

repo=/home/machinist/Development/llama.cpp-miinfer
branch=perf/m31-q8-dot4-prepack-stage
image=localhost/miinfer-dev:rocm-7.2.1
model=/home/machinist/models/Qwen3.8-27B-Q4_K_M.gguf
model_sha=7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
git -C "$repo" fetch origin "$branch"
commit="$(git -C "$repo" rev-parse FETCH_HEAD)"
worktree="/tmp/m31-q8-dot4-clean-${commit:0:10}"

free_kb="$(df -Pk /tmp | awk 'NR == 2 { print $4 }')"
(( free_kb > 12000000 )) || { echo "less than 12 GB free under /tmp: ${free_kb:-unknown} KB" >&2; exit 2; }
actual_model_sha="$(sha256sum "$model" | awk '{print $1}')"
[[ "$actual_model_sha" == "$model_sha" ]] || { echo "model SHA mismatch: $actual_model_sha" >&2; exit 3; }
if [[ -e "$worktree" ]]; then
    [[ "$(git -C "$worktree" rev-parse HEAD)" == "$commit" ]] || { echo "worktree path already has a different commit: $worktree" >&2; exit 4; }
    [[ -z "$(git -C "$worktree" status --porcelain)" ]] || { echo "existing worktree is dirty: $worktree" >&2; exit 5; }
else
    git -C "$repo" worktree add --detach "$worktree" "$commit"
fi

podman run --rm --security-opt label=disable -v "$worktree:/workspace" -w /workspace "$image" \
    cmake -S . -B build-mi50 -DGGML_HIP=ON -DGPU_TARGETS=gfx906 \
    -DCMAKE_BUILD_TYPE=Release -DGGML_BUILD_TESTS=ON -DLLAMA_BUILD_TESTS=OFF
podman run --rm --security-opt label=disable -v "$worktree:/workspace" -w /workspace "$image" \
    cmake --build build-mi50 --target llama-bench test-backend-ops -j4

M31_CANDIDATE_WORKSPACE="$worktree" M31_BUILD_DIR="$worktree/build-mi50" \
    bash "$worktree/.audit/run-m31-q8-dot4-lds-ops.sh"
M31_WORKSPACE="$worktree" M31_GPU=2 M31_GPU_BDF=0000:88:00.0 \
    bash "$worktree/.audit/run-m31-q8-dot4-8k-ab.sh"
