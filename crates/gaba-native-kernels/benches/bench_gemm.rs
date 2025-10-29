use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gaba_native_kernels::{gemm, gemm_rust};

fn bench_size(c: &mut Criterion, m: usize, n: usize, k: usize) {
    let a: Vec<f32> = (0..(m * k)).map(|i| ((i * 23 + 5) % 97) as f32 * 0.0123).collect();
    let b: Vec<f32> = (0..(k * n)).map(|i| ((i * 19 + 11) % 89) as f32 * 0.0173).collect();
    let mut c_ref = vec![0f32; m * n];
    let mut c_tgt = vec![0f32; m * n];

    let id = format!("gemm_{}x{}x{}", m, n, k);

    c.bench_function(&format!("rust/{}", id), |bencher| {
        bencher.iter(|| {
            gemm_rust(black_box(&a), black_box(&b), black_box(&mut c_ref), m, n, k);
        })
    });

    c.bench_function(&format!("native/{}", id), |bencher| {
        bencher.iter(|| {
            gemm(black_box(&a), black_box(&b), black_box(&mut c_tgt), m, n, k);
        })
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    bench_size(c, 64, 64, 64);
    bench_size(c, 128, 128, 128);
    bench_size(c, 256, 256, 256);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
