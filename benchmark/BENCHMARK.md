# Preliminary Benchmark

## Setup

- Input: 60 min recording (wav); no overlap -> 1,200 3s chunks
- Drive: Samsung PM981A (M.2 2280; reads up to 3,500 MB/s)
- CPU: AMD Ryzen 7 3800X (8 cores / 16 threads)
- GPU: NVIDIA Titan RTX (24 GB memory; 4,608 CUDA cores, CUDA 12.6)
- OS: Ubuntu 22.04.4 LTS
- RAM: 64 GB DDR4 (4 x 16 GB; 3,200 MHz; C16; VENGEANCE LPX)

Clear cache memory before each run:

```sh
sudo sync
sudo bash -c 'echo 3 > /proc/sys/vm/drop_caches'
```

Setting thread count for:

- TFLite: `Interpreter(…, num_threads=n_threads)`
- Protobuf:
  - `tf.config.threading.set_inter_op_parallelism_threads(n_threads)`
  - `tf.config.threading.set_intra_op_parallelism_threads(n_threads)`

## Results

![Benchmark Single File](https://raw.githubusercontent.com/birdnet-team/birdnet/main/benchmark/single_file.png)

- 3 Models: Protobuf CPU, Protobuf GPU, TFLite
- 6 Batch Sizes: 1, 100, 300, 600, 1000, 1200
  - GPU memory was exceeded when using Protobuf GPU with batch size 1200
- 3 Threads: 1, 4, auto
  - only for Protobuf CPU and TFLite

![Benchmark Multiple Files](https://raw.githubusercontent.com/birdnet-team/birdnet/main/benchmark/single_file.png)

- 4 files: 4 x 1h
- 2 Models: Protobuf CPU, TFLite
- 2 Multithreading approaches
  - v1: share same model on all threads
    - not possible for Protobuf CPU
  - v2: each thread loads its own model (much more memory)

## Conclusion

- Single file:
  - GPU is fastest
  - Protobuf CPU model is faster than TFLite
  - use batch size 100 with threads to ‘auto’
  - no large RAM required
- Multiple files:
  - use Protobuf CPU or TFLite model with batch size 1
  - on >=4 cores CPU seems to be more effective than GPU
    - side note: 16 x 1h took 12s per hour (Protobuf/TFLite; batch size 1) compared to 24s on GPU
  - large RAM required
