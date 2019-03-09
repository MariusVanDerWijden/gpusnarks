# GPU-SNARK

A GPU snark accelerator written at ETHParis 

## Benchmarks

CPU Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz
runs with 8 threads
GPU: GTX 1080

Constraints | Constraints | GPU | CPU
------------|-------------|-----|-----
2^16 | 65536 | 1.67 s | 0.005 s
2^20 | 1048576 | 22.35 s | 84.61 s
2^22 | 4194304 | 189.87 s | 421.780 s