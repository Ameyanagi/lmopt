use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use faer::Mat;
use lmopt::utils::jacobian::get_jacobian_calculator;
use lmopt::{JacobianMethod, LeastSquaresProblem, LevenbergMarquardt, Result};
use std::hint::black_box;

struct DenseProblem {
    residual_count: usize,
    parameter_count: usize,
}

impl LeastSquaresProblem<f64> for DenseProblem {
    fn residuals(&self, parameters: &Mat<f64>) -> Result<Mat<f64>> {
        Ok(Mat::from_fn(self.residual_count, 1, |i, _| {
            let mut value = -0.25;
            for j in 0..self.parameter_count {
                let frequency = 1.0 + ((i + 3 * j) % 17) as f64 * 0.05;
                value += (parameters[(j, 0)] * frequency).sin();
            }
            value
        }))
    }

    fn jacobian(&self, parameters: &Mat<f64>) -> Option<Mat<f64>> {
        Some(Mat::from_fn(self.residual_count, self.parameter_count, |i, j| {
            let frequency = 1.0 + ((i + 3 * j) % 17) as f64 * 0.05;
            frequency * (parameters[(j, 0)] * frequency).cos()
        }))
    }
}

fn benchmark_jacobians(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("jacobian");
    for residual_count in [64, 1_024] {
        let parameter_count = 16;
        let problem = DenseProblem { residual_count, parameter_count };
        let parameters = Mat::from_fn(parameter_count, 1, |i, _| 0.02 * (i + 1) as f64);
        group.throughput(Throughput::Elements((residual_count * parameter_count) as u64));

        for method in [JacobianMethod::Auto, JacobianMethod::UserProvided, JacobianMethod::NumericalForward, JacobianMethod::NumericalCentral] {
            let calculator = get_jacobian_calculator::<f64>(method, 1e-6);
            group.bench_with_input(BenchmarkId::new(format!("{method:?}"), residual_count), &residual_count, |bencher, _| {
                bencher.iter(|| calculator.calculate_jacobian(black_box(&problem), black_box(&parameters)).unwrap())
            });
        }
    }
    group.finish();
}

fn benchmark_optimizer(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("optimizer");
    for residual_count in [64, 1_024] {
        let problem = DenseProblem { residual_count, parameter_count: 4 };
        let initial = Mat::from_fn(4, 1, |i, _| 0.1 * (i + 1) as f64);
        let optimizer = LevenbergMarquardt::new().with_max_iterations(50).with_jacobian_method(JacobianMethod::UserProvided);
        group.throughput(Throughput::Elements(residual_count as u64));
        group.bench_with_input(BenchmarkId::new("analytical", residual_count), &residual_count, |bencher, _| {
            bencher.iter(|| optimizer.minimize(black_box(&problem), black_box(&initial)).unwrap())
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_jacobians, benchmark_optimizer);
criterion_main!(benches);
