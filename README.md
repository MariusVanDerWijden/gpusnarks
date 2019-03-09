# GPU-SNARK

A GPU snark accelerator written at ETHParis 

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
2^16 | 65536 | 1.67 s | 0.005 s
2^20 | 1048576 | 1.61 s | 2.06 s
2^22 | 4194304 | 1.70 s | 7.73 s
2^28 | 268435456 | 227.25 s | 

## License

This Software is currently under no license which means that *no one* is permitted to distribute or change this code.
The sole copyright owner is MariusVanDerWijden.

I will change the License in the future to a permissive Open Source License!
