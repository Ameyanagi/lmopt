# Performance baseline

This baseline was captured on 2026-08-11 with lmopt 0.2.0, Rust 1.95.0,
macOS arm64, and an Apple M4. It is a short 10-sample smoke measurement, useful
for detecting large regressions rather than small percentage changes.

| Benchmark | 64 residuals | 1,024 residuals |
| --- | ---: | ---: |
| Auto analytical Jacobian | 3.43 µs | 63.12 µs |
| Forced analytical Jacobian | 3.64 µs | 64.92 µs |
| Forward finite differences | 54.48 µs | 1.14 ms |
| Central finite differences | 118.37 µs | 1.96 ms |
| End-to-end analytical optimization | 29.72 µs | 391.11 µs |

Reproduce the short baseline with:

```text
cargo bench --bench performance -- \
  --warm-up-time 0.1 --measurement-time 0.2 --sample-size 10
```

For decisions involving small differences, run the default Criterion duration
on an otherwise idle machine and compare Criterion result directories from the
same host. Do not compare raw timings across different machines.

## GPU feasibility

The isolated `experiments/cubecl` program compares CPU and CubeCL/WGPU
residual-plus-Jacobian evaluation over a much larger size range. Its README
records the Apple M4 results and the important limitation: it benchmarks only
the stage CubeCL can currently provide, not the complete optimizer or its
pivoted-QR/SVD linear solve.
