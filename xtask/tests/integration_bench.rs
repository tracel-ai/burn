use std::process::Command;
use std::path::PathBuf;

// Integration test: run the bench CSV writer, then run xtask bench_compare with a large
// allowed regression so the test passes as long as the pipeline runs end-to-end.
#[test]
fn end_to_end_bench_compare_pipeline() {
    // xtask crate manifest dir is .../gaba-burn/xtask, parent is repo root (.../gaba-burn)
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();

    // Step 1: run bench_gemm_csv binary from gaba-native-kernels
    let status = Command::new("cargo")
        .current_dir(&repo_root)
        .args(["run", "-p", "gaba-native-kernels", "--bin", "bench_gemm_csv"])
        .status()
        .expect("failed to spawn bench_gemm_csv");
    assert!(status.success(), "bench_gemm_csv failed");

    // Step 2: run xtask bench_compare against the generated CSV
    let baseline = repo_root.join("crates/gaba-native-kernels/benches/baseline.csv");
    let current = repo_root.join("target/bench-csv/gaba-native-kernels.csv");

    let status2 = Command::new("cargo")
        .current_dir(&repo_root)
        .args([
            "run",
            "-p",
            "xtask",
            "--bin",
            "bench_compare",
            "--",
            "--baseline",
            baseline.to_str().unwrap(),
            "--current",
            current.to_str().unwrap(),
            "--max-regression-pct",
            "1000",
        ])
        .status()
        .expect("failed to spawn bench_compare");

    assert!(status2.success(), "bench_compare failed");
}
