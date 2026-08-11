use cubecl::{future, prelude::*, server::Handle};
use std::hint::black_box;
use std::time::{Duration, Instant};

#[cube(launch_unchecked)]
fn residual_jacobian(parameters: &Array<f32>, residuals: &mut Array<f32>, jacobian: &mut Array<f32>) {
    let pos = ABSOLUTE_POS;
    let rows = residuals.len();
    let cols = parameters.len();

    if pos < rows * cols {
        let i = pos / cols;
        let j = pos % cols;
        let frequency = 1.0 + ((i + 3 * j) % 17) as f32 * 0.05;
        jacobian[pos] = frequency * f32::cos(parameters[j] * frequency);
    }

    if pos < rows {
        let mut value = f32::new(-0.25);
        for j in 0..cols {
            let frequency = 1.0 + ((pos + 3 * j) % 17) as f32 * 0.05;
            value += f32::sin(parameters[j] * frequency);
        }
        residuals[pos] = value;
    }
}

fn launch<R: Runtime>(client: &ComputeClient<R>, parameters: &Handle, residuals: &Handle, jacobian: &Handle, rows: usize, cols: usize) {
    let elements = rows * cols;
    let cube_dim = 256;
    let cube_count = elements.div_ceil(cube_dim as usize) as u32;
    unsafe {
        residual_jacobian::launch_unchecked::<R>(
            client,
            CubeCount::Static(cube_count, 1, 1),
            CubeDim::new_1d(cube_dim),
            ArrayArg::from_raw_parts(parameters.clone(), cols),
            ArrayArg::from_raw_parts(residuals.clone(), rows),
            ArrayArg::from_raw_parts(jacobian.clone(), elements),
        );
    }
}

fn cpu_once(parameters: &[f32], rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
    let mut residuals = vec![0.0; rows];
    let mut jacobian = vec![0.0; rows * cols];
    for i in 0..rows {
        let mut value = -0.25;
        for j in 0..cols {
            let frequency = 1.0 + ((i + 3 * j) % 17) as f32 * 0.05;
            value += (parameters[j] * frequency).sin();
            jacobian[i * cols + j] = frequency * (parameters[j] * frequency).cos();
        }
        residuals[i] = value;
    }
    (residuals, jacobian)
}

fn median(mut timings: Vec<Duration>) -> Duration {
    timings.sort_unstable();
    timings[timings.len() / 2]
}

fn main() {
    type R = cubecl::wgpu::WgpuRuntime;
    let runtime_start = Instant::now();
    let client = R::client(&Default::default());
    println!("runtime={:?} initialization={:.3}ms", R::name(&client), runtime_start.elapsed().as_secs_f64() * 1e3);

    for (rows, cols, iterations) in [(64, 16, 500), (1_024, 16, 300), (16_384, 16, 100), (65_536, 32, 40), (262_144, 32, 12)] {
        let parameters: Vec<f32> = (0..cols).map(|j| 0.02 * (j + 1) as f32).collect();
        let parameter_handle = client.create_from_slice(f32::as_bytes(&parameters));
        let residual_handle = client.empty(rows * core::mem::size_of::<f32>());
        let jacobian_handle = client.empty(rows * cols * core::mem::size_of::<f32>());

        let first_start = Instant::now();
        launch(&client, &parameter_handle, &residual_handle, &jacobian_handle, rows, cols);
        future::block_on(client.sync()).unwrap();
        let first = first_start.elapsed();

        let expected = cpu_once(&parameters, rows, cols);
        let residual_bytes = client.read_one(residual_handle.clone()).unwrap();
        let jacobian_bytes = client.read_one(jacobian_handle.clone()).unwrap();
        let gpu_residuals = f32::from_bytes(&residual_bytes);
        let gpu_jacobian = f32::from_bytes(&jacobian_bytes);
        let residual_error = expected.0.iter().zip(gpu_residuals).map(|(cpu, gpu)| (cpu - gpu).abs()).fold(0.0f32, f32::max);
        let jacobian_error = expected.1.iter().zip(gpu_jacobian).map(|(cpu, gpu)| (cpu - gpu).abs()).fold(0.0f32, f32::max);
        assert!(residual_error < 1e-4, "residual mismatch: {residual_error}");
        assert!(jacobian_error < 1e-5, "jacobian mismatch: {jacobian_error}");

        let mut gpu_timings = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            launch(&client, &parameter_handle, &residual_handle, &jacobian_handle, rows, cols);
            future::block_on(client.sync()).unwrap();
            gpu_timings.push(start.elapsed());
        }

        let transfer_iterations = iterations.min(20);
        let mut roundtrip_timings = Vec::with_capacity(transfer_iterations);
        for _ in 0..transfer_iterations {
            let start = Instant::now();
            launch(&client, &parameter_handle, &residual_handle, &jacobian_handle, rows, cols);
            let residual_bytes = client.read_one(residual_handle.clone()).unwrap();
            let jacobian_bytes = client.read_one(jacobian_handle.clone()).unwrap();
            black_box((residual_bytes, jacobian_bytes));
            roundtrip_timings.push(start.elapsed());
        }

        let mut cpu_timings = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            black_box(cpu_once(black_box(&parameters), rows, cols));
            cpu_timings.push(start.elapsed());
        }

        let cpu = median(cpu_timings);
        let gpu = median(gpu_timings);
        let roundtrip = median(roundtrip_timings);
        let ratio = cpu.as_secs_f64() / gpu.as_secs_f64();
        println!(
            "rows={rows:>7} cols={cols:>2} cpu={:>10.3}us gpu_warm={:>10.3}us roundtrip={:>10.3}us first={:>10.3}ms speedup={ratio:>7.3}x max_err={:.2e}",
            cpu.as_secs_f64() * 1e6,
            gpu.as_secs_f64() * 1e6,
            roundtrip.as_secs_f64() * 1e6,
            first.as_secs_f64() * 1e3,
            residual_error.max(jacobian_error),
        );
    }
}
