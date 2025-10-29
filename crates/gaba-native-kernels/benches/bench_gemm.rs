// Criterion-based benches removed. Use the CSV bench runner instead:
//
// cargo run -p gaba-native-kernels --bin bench_gemm_csv --release
//
// This file kept as a placeholder so `cargo bench` won't fail when Criterion is removed.
fn main() {
    eprintln!("Criterion-based benches have been removed. Run 'cargo run --bin bench_gemm_csv' to produce CSV results.");
}
