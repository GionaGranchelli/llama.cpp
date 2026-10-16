# GQA6 dot4 per-4 query-scale candidate build attempt

Commit `86f18bd9c` changes the opt-in query prepack from one FP32 scale per 32
dimensions to one scale per four dimensions, matching each signed-byte dot4
group. The motivation is to reduce quantization error seen in the matched
12,738-token final-logit comparison. The first per-four attempt did not compile
because its scale index was outside the dot4 loop. Commit `a432a8c0b` fixes the
accumulation and compiled successfully; see
`m31-gqa6-per4-thermal-abort-20261016.md` for the guarded GPU result. Correctness
and performance are still unverified.

The commit was fetched into the existing Machinist test checkout. A host-side
`cmake --build` attempt failed before compilation because its CMake cache was
created inside the ROCm container at `/workspace/build-m31`, while the host
path is `/tmp/m31-current-01a910937/build-m31`. The successful rebuild ran
inside the pinned container with the checkout mounted at `/workspace`.
