# GPU-SNARK

A GPU snark accelerator written at ETHParis.
It speeds up the Fast Fourier Transformation on Finite Elements by more than **40 times**

## Setup

```
mkdir build 
cd build
cmake ..
make
```

## Benchmarks

CPU Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz runs with 8 threads

GPU: GTX 1080

Constraints | Constraints | GPU | CPU
------------|-------------|-----|-----
2^16 | 65536 | 0.32 s | 0.21 s
2^20 | 1048576 | 0.33 s | 4.40 s
2^22 | 4194304 | 0.43 s | 18.03 s
2^25 | 33554432 | 10.57 s | *
2^28 | 268435456 | 227.25 s | *

* These benchmarks haven't finished in time for the submission to ETHParis

## License

This Software is currently under no license which means that *no one* is permitted to distribute or change this code.
The sole copyright owner is MariusVanDerWijden.

I will change the License in the future to a permissive Open Source License!
