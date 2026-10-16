# Direct-sysfs thermal guard check

The updated 8K screen runner read the MI50 junction sensor from the BDF-matched
sysfs `temp*_input` file and stopped the benchmark at its 84 C threshold. This
confirms that the direct sensor lookup and fail-closed stop path worked during
a real inference attempt.

- Path: default Q8_0 K/V, Flash Attention, 99 GPU layers
- Prompt: 8,192 tokens, generation disabled, `-b 2048 -ub 1024`
- Host/GPU: Machinist, GPU0, BDF `0000:04:00.0`, GFX906
- Start junction: 28 C
- Stop: junction read 84 C; container killed, process status 137
- Result: no benchmark record; do not treat as a performance measurement
- Caps: GPU0 225 W, GPU2 300 W
- Source tree: `7f4a20998cdc8dd151b34180a03278685c96360e`
- Raw artifacts: `/tmp/m31-gqa48-8k-screen-gpu0-20261016T070653Z/`

The temperature trace contains multiple sysfs samples within each second,
unlike the previous `rocm-smi` poll. It stopped at 84 C rather than the 85 C
sample seen with the slower poll. GPU0 is still too thermally constrained at
the current cap for even this 8K default attempt to finish consistently; lower
both MI50 caps to the guarded 180 W limit before any sustained 64K run.
