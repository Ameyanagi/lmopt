use criterion::{Criterion, criterion_group, criterion_main};
use lmopt_enzyme_experiment::{analytical_jacobian, central_difference_jacobian, enzyme_jacobian};
use std::hint::black_box;

fn benchmark_jacobians(criterion: &mut Criterion) {
    let parameters = [-1.2, 1.0];
    let mut group = criterion.benchmark_group("rosenbrock_jacobian_2x2");

    group.bench_function("analytical", |bencher| bencher.iter(|| analytical_jacobian(black_box(&parameters))));
    group.bench_function("enzyme_forward", |bencher| bencher.iter(|| enzyme_jacobian(black_box(&parameters))));
    group.bench_function("central_difference", |bencher| bencher.iter(|| central_difference_jacobian(black_box(&parameters))));
    group.finish();
}

criterion_group!(benches, benchmark_jacobians);
criterion_main!(benches);
