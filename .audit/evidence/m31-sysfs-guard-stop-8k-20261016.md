# Sysfs thermal guard stop during default 8K run

The updated direct-sysfs guard stopped a default 8K prefill on GPU0 at 84 C.
It returned status 137 and produced no benchmark result. The temperature trace
contains several samples per second, and the recorded maximum is 84 C; unlike
the earlier `rocm-smi`-polled candidate run, this attempt did not record 85 C.

- Source tree: `7f4a20998cdc8dd151b34180a03278685c96360e`
- Binary build source: `01a91093734977e04de98ebd48ac2fe169b47a63`
- Host/GPU: Machinist, GPU0, BDF `0000:04:00.0`, GFX906
- Start junction: 28 C
- Workload: 8,192 prompt tokens, zero generation, `-b 2048 -ub 1024`,
  Q8_0 K/V, Flash Attention, 99 GPU layers
- Power caps: GPU0 225 W, GPU2 300 W
- Model SHA-256:
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Raw artifacts: `/tmp/m31-gqa48-8k-screen-gpu0-20261016T070653Z/`

This validates the direct junction sensor lookup and fail-closed stop path, not
the benchmark. The earlier GPU0 default 8K sample completed at 33.483 s with a
sampled peak of 83 C; this run stopped from a similar start temperature and
the same power caps, showing that GPU0 cannot reliably finish this workload
at the current 225 W cap. No inference process remained after the stop.
