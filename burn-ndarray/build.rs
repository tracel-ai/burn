fn main() {
    // https://github.com/rust-ndarray/ndarray/issues/1197
    if cfg!(feature = "blas-accelerate") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
