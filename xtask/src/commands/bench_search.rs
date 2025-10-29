use std::time::Instant;
use gaba_burn_vector::{CpuSearchEngine, VectorMetadata};
use rand::Rng;
use rand::rng;

#[allow(dead_code)]
fn random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rng();
    (0..dim).map(|_| rng.random_range(0.0f32..1.0f32)).collect()
}

#[allow(dead_code)]
pub fn run_bench_search() {
    // Simple synthetic benchmark: build N vectors and run repeated searches
    let dim = 384;
    let n = 50_000usize;
    eprintln!("Building {} vectors (dim={})", n, dim);

    let mut vectors = Vec::with_capacity(n);
    for i in 0..n {
        let id = format!("doc-{}", i);
        let vec = random_vector(dim);
        let meta = VectorMetadata { title: id.clone(), source_path: id.clone(), content: format!("content {}", i) };
        vectors.push((id, vec, meta));
    }

    let engine = CpuSearchEngine::new();
    let query = random_vector(dim);

    // Warmup
    for _ in 0..5 {
        let _ = engine.search_parallel(&query, &vectors, 10);
    }

    // Timed runs
    let runs = 10;
    let mut durations = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let res = engine.search_parallel(&query, &vectors, 10);
        let dur = start.elapsed();
        eprintln!("Found {} results in {:?}", res.len(), dur);
        durations.push(dur.as_secs_f64());
    }

    let avg: f64 = durations.iter().sum::<f64>() / durations.len() as f64;
    eprintln!("Average search latency: {:.6} s", avg);
}
