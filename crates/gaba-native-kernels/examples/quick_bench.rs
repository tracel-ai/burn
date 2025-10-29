use std::time::Instant;
use gaba_native_kernels::{gemm, gemm_rust};

fn time_gemm_rust(m: usize, n: usize, k: usize) -> f64 {
    let a: Vec<f32> = (0..(m * k)).map(|i| ((i * 23 + 5) % 97) as f32 * 0.0123).collect();
    let b: Vec<f32> = (0..(k * n)).map(|i| ((i * 19 + 11) % 89) as f32 * 0.0173).collect();
    let mut c = vec![0f32; m * n];

    let iters = 3;
    let start = Instant::now();
    for _ in 0..iters {
        gemm_rust(&a, &b, &mut c, m, n, k);
    }
    let dur = start.elapsed();
    dur.as_secs_f64() / (iters as f64)
}

fn time_gemm_native(m: usize, n: usize, k: usize) -> f64 {
    let a: Vec<f32> = (0..(m * k)).map(|i| ((i * 23 + 5) % 97) as f32 * 0.0123).collect();
    let b: Vec<f32> = (0..(k * n)).map(|i| ((i * 19 + 11) % 89) as f32 * 0.0173).collect();
    let mut c = vec![0f32; m * n];

    let iters = 3;
    let start = Instant::now();
    for _ in 0..iters {
        gemm(&a, &b, &mut c, m, n, k);
    }
    let dur = start.elapsed();
    dur.as_secs_f64() / (iters as f64)
}

fn main() {
    let m = 128usize;
    let n = 128usize;
    let k = 128usize;

    println!("Size: {}x{}x{}", m, n, k);
    let rust_t = time_gemm_rust(m, n, k);
    println!("rust gemm avg: {:.6} s", rust_t);
    let native_t = time_gemm_native(m, n, k);
    println!("native gemm avg: {:.6} s", native_t);
    println!("ratio (rust/native): {:.3}", rust_t / native_t);
}
