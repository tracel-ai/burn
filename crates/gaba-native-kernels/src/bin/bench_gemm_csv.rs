use std::time::Instant;
use std::fs::{create_dir_all, File};
use std::io::Write;

use gaba_native_kernels::{gemm, gemm_rust};

fn time_fn<F: FnMut()>(mut f: F) -> f64 {
    let start = Instant::now();
    f();
    start.elapsed().as_secs_f64()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n/2]
    } else {
        (v[n/2 - 1] + v[n/2]) / 2.0
    }
}

fn run_case(_label: &str, m: usize, n: usize, k: usize) -> (f64, f64) {
    let a: Vec<f32> = (0..(m * k)).map(|i| ((i * 23 + 5) % 97) as f32 * 0.0123).collect();
    let b: Vec<f32> = (0..(k * n)).map(|i| ((i * 19 + 11) % 89) as f32 * 0.0173).collect();
    let mut c_ref = vec![0f32; m * n];
    let mut c_tgt = vec![0f32; m * n];

    // warmup
    for _ in 0..3 {
        gemm_rust(&a, &b, &mut c_ref, m, n, k);
        gemm(&a, &b, &mut c_tgt, m, n, k);
    }

    // timed runs
    let runs = 9;
    let mut rust_times = Vec::with_capacity(runs);
    let mut native_times = Vec::with_capacity(runs);

    for _ in 0..runs {
        rust_times.push(time_fn(|| gemm_rust(&a, &b, &mut c_ref, m, n, k)));
        native_times.push(time_fn(|| gemm(&a, &b, &mut c_tgt, m, n, k)));
    }

    (median(rust_times), median(native_times))
}

fn main() -> anyhow::Result<()> {
    let sizes = vec![(64usize,64,64),(128,128,128),(256,256,256)];

    let out_dir = std::path::Path::new("target/bench-csv");
    create_dir_all(out_dir)?;
    let out_file = out_dir.join("gaba-native-kernels.csv");
    let mut f = File::create(&out_file)?;
    writeln!(f, "benchmark,median_s")?;

    for (m,n,k) in sizes {
        let id = format!("gemm_{}x{}x{}", m, n, k);
        let (rust_med, native_med) = run_case(&id, m, n, k);
        writeln!(f, "rust/{},{}", id, rust_med)?;
        writeln!(f, "native/{},{}", id, native_med)?;
        eprintln!("Wrote results for {} -> rust: {:.6}s native: {:.6}s", id, rust_med, native_med);
    }

    println!("Wrote CSV to {}", out_file.display());
    Ok(())
}
