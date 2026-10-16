# Clean candidate build setup attempt

Date: 2026-10-16 (UTC)

The clean worktree at `ad9d59f75` was fetched and created on Machinist, but
CMake configuration stopped before compilation or any GPU test. No benchmark
was launched.

Two setup issues were identified:

1. `GGML_BUILD_TESTS=ON` asks `ggml/CMakeLists.txt` to add `ggml/tests`, which
   is not present in this source tree. The relevant `test-backend-ops` target
   is under the root `tests/` directory and is enabled with
   `LLAMA_BUILD_TESTS=ON` instead.
2. The container mounted only the linked worktree. Its `.git` file points into
   the host repository's common Git directory, which the container could not
   see. CMake printed `ggml commit: unknown`; this would also make the strict
   A/B runner reject the benchmark binary's embedded source identity.

The preparation script now disables ggml's unavailable standalone tests,
enables llama.cpp tests, and mounts the common Git metadata read-only at its
original path. The next attempt will use a new clean worktree at the updated
script commit. The failed worktree remains isolated under `/tmp` for audit;
the shared checkout was not changed.
