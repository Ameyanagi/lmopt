# CubeCL GPU feasibility benchmark

This isolated experiment compares the residual-plus-analytical-Jacobian
workload from `benches/performance.rs` on one CPU core and CubeCL's WGPU
backend. It is not a GPU implementation of Levenberg-Marquardt: CubeCL and
Cubek do not currently provide the pivoted QR and SVD operations used by
lmopt's robust linear solver.

Run it on a WGPU-capable machine with:

```text
cargo run --locked --release --manifest-path experiments/cubecl/Cargo.toml
```

The benchmark reports:

- `cpu`: median time to produce host-resident residual and Jacobian arrays;
- `gpu_warm`: median dispatch plus synchronization time with buffers retained
  on the device;
- `roundtrip`: dispatch plus copying both outputs back to the host;
- `first`: first dispatch for the shape, including any uncached compilation;
- `speedup`: `cpu / gpu_warm`, so values above 1 favor the GPU.

The CPU and GPU execute the same `f32` formulas and the program validates both
outputs. Keeping the GPU result on-device is the optimistic case. A complete
optimizer also needs reductions, damping, a robust least-squares solve, and
iteration control, so these numbers must not be presented as end-to-end lmopt
speedups.

## Apple M4 result

Measured on 2026-08-11 on a 10-core CPU / 10-core GPU Apple M4 using CubeCL
0.10.0 over WGPU/Metal. Representative warm medians were:

| Residuals | Parameters | CPU | GPU, on-device | GPU, host roundtrip | Kernel speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 16 | 2.5–3.5 µs | 1.47 ms | 3.40 ms | 0.002x |
| 1,024 | 16 | 38–45 µs | 1.46 ms | 2.91 ms | 0.03x |
| 16,384 | 16 | 0.56–0.74 ms | 1.46 ms | 2.90 ms | 0.4–0.5x |
| 65,536 | 32 | 7.1–8.3 ms | 1.46 ms | 2.90 ms | 4.9–5.6x |
| 262,144 | 32 | 33.8–39.2 ms | 1.5–4.0 ms | 4.18 ms | 8.5–22x |

The first uncached shader dispatch was about 62 ms. Results varied under
macOS scheduling, especially at the largest size; use the benchmark on the
target hardware before making an API decision.
