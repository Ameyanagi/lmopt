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

## 0.3 solver smoke check

On 2026-08-14, a sequential old/new check on the same Apple M4 host with Rust
1.95 measured the 64-residual analytical solve at 121.46 µs before the 0.3
changes and 69.16 µs after them (median estimates). The short 20-sample check
also measured:

| Benchmark | Median |
| --- | ---: |
| End-to-end analytical optimization, 1,024 residuals | 766.97 µs |
| Retry-heavy Rosenbrock solve | 77.90 µs |

The 64-residual improvement comes from eliminating duplicate residual- and
Jacobian-norm passes and reusing the augmented solve workspace. Treat these as
smoke results, not portable absolute performance claims.
