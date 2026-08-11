# Experimental automatic differentiation

`lmopt` does not currently expose automatic differentiation in its stable API.
`JacobianMethod::AutoDiff` returns `Error::AutoDiffUnavailable` instead of
silently running finite differences.

Rust's `std::autodiff` implementation is nightly-only. It currently requires:

- the `enzyme` rustup component;
- `RUSTFLAGS="-Zautodiff=Enable"`;
- release mode with fat LTO; and
- a statically dispatched, monomorphic residual kernel (`dyn Trait` inputs are
  not supported).

The isolated prototype in `experiments/enzyme` differentiates a Rosenbrock
residual kernel with forward-mode Jacobian-vector products and verifies the
result against its analytical Jacobian.

Run it with:

```sh
rustup update nightly
rustup +nightly component add enzyme
RUSTFLAGS="-Zautodiff=Enable" \
  cargo +nightly run --locked --release \
  --manifest-path experiments/enzyme/Cargo.toml
```

Compare analytical, Enzyme forward-mode, and central-difference Jacobians with:

```sh
RUSTFLAGS="-Zautodiff=Enable" \
  cargo +nightly bench --locked --manifest-path experiments/enzyme/Cargo.toml \
  --bench jacobian
```

On the 2×2 Rosenbrock microkernel, the first Apple M4 smoke baseline measured
about 0.99 ns for the analytical Jacobian, 0.97 ns for Enzyme forward mode, and
1.99 ns for central differences. This validates that generated derivatives can
match hand-written derivative cost for a tiny static kernel; it does not yet
establish end-to-end gains for dynamically sized lmopt problems.

The intended library integration is a separate, statically dispatched problem
interface whose residual dimensions are known by the differentiated kernel.
The generated Jacobian can then enter the same checked QR/SVD solver path used
by analytical and numerical Jacobians. It should remain experimental until:

1. the prototype runs on every supported Enzyme host in CI;
2. its Jacobians match analytical derivatives across representative problems;
3. Criterion benchmarks show a benefit over analytical and finite-difference
   methods; and
4. nightly/compiler failures are isolated from the stable crate.

References:

- <https://doc.rust-lang.org/nightly/std/autodiff/index.html>
- <https://rustc-dev-guide.rust-lang.org/autodiff/installation.html>
- <https://github.com/rust-lang/rust/issues/124509>
