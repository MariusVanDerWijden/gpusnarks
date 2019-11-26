# GPU-SNARKs

A GPU snark accelerator written at ETHParis.
It speeds up the Fast Fourier Transformation on Finite Elements by more than **40 times**.
Please note that this speedup is only for 32bit values. 
For 768-bit values (needed for ZK-Snarks) we observed speedups of around **20 times**

## Setup

```
mkdir build 
cd build
cmake .. -DCMAKE_C_COMPILER=/usr/bin/gcc-6
make
```

## Benchmarks 

The benchmarks are

### FFT 32-bit

CPU: Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz runs with 8 threads
GPU: GTX 1080

These benchmarks are only valid for 32 bit values.

| Constraints | Constraints | GPU      | CPU     |
| ----------- | ----------- | -------- | ------- |
| 2^16        | 65536       | 0.32 s   | 0.21 s  |
| 2^20        | 1048576     | 0.33 s   | 4.40 s  |
| 2^22        | 4194304     | 0.43 s   | 18.03 s |
| 2^25        | 33554432    | 10.57 s  | *       |
| 2^28        | 268435456   | 227.25 s | *       |

* These benchmarks haven't finished in time for the submission to ETHParis

### FFT 768-bit

CPU: Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz
GPU: GTX 1060

| Constraints | Constraints | GPU     | CPU      |
| ----------- | ----------- | ------- | -------- |
| 2^16        | 65536       | 6.95 s  | 105.57 s |
| 2^18        | 262144      | 21.42 s | 424.73 s |
| 2^19        | 524288      | 40.09 s | 858.22 s |

### Multiexp Scalar

CPU: Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz
GPU: GTX 1060

| Constraints | Constraints | GPU    | CPU    |
| ----------- | ----------- | ------ | ------ |
| 2^18        | 262144      | 0.15 s | 1.07 s |
| 2^19        | 524288      | 0.17 s | 2.16 s |
| 2^20        | 1048576     | 0.17 s | 4.38 s |

### Multiexp MNT4753_G1

CPU: Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz
GPU: GTX 1060

| Constraints | Constraints | GPU    | CPU      |
| ----------- | ----------- | ------ | -------- |
| 2^16        | 65536       | 0.29 s | 23.48 s  |
| 2^18        | 262144      | 0.86 s | 91.96 s  |
| 2^20        | 1048576     | 3.35 s | 372.67 s |

## License

Copyright [2019] [Marius van der Wijden]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
