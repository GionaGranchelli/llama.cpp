# GQA48 8K screen: build path blocker

The guarded screen did not reach GPU work. On Machinist, the candidate worktree
was at `176480a57a5b18be94efdc7f97b9130545775860`; the test binary existed, but
`llama-bench` did not. The runner entered the pinned ROCm 7.2.1 container to
build `llama-bench`, where `cmake --build build-gqa48 --target llama-bench`
failed with:

```text
Error: /workspace/build-gqa48 is not a directory
```

The same worktree and build tree had previously produced `test-backend-ops`.
A follow-up host/container inspection found the build directory on the host,
but a minimal container probe without `--security-opt label=disable` received
`Permission denied` for `/workspace/build-gqa48`. The build-only Podman
invocation omitted that option, while the existing test and inference
containers include it. This is the likely cause of CMake's directory error.
The runner now applies the same label setting during the build step. The
benchmark still has not run.

Preflight immediately before the attempt found no KFD processes; GPU0 was at
28 C with a 225 W cap and GPU2 at 31 C with a 300 W cap. The intended screen
uses only GPU2/BDF `0000:88:00.0`, 8,192 prompt tokens, Q8 K/V, `-b 2048`,
`-ub 1024`, Flash Attention, and a hard 84 C stop. Both caps remain above the
180 W gate for 64K work.

Next inspect both host and container views of
`/tmp/m31-gqa48-rocm721-20261016/build-gqa48` before rebuilding. Do not treat
this attempt as evidence for or against the larger tile's performance.
