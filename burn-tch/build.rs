use std::env;
fn main() {
    // Temporary workaround for https://github.com/burn-rs/burn/issues/180
    // Remove this once tch-rs upgrades to Torch 2.0 at least

    if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
        let message = "Run scripts/fix-tch-build-arm64.py to fix environment variables for torch";
        env::var("LIBTORCH").expect(message);
        env::var("DYLD_LIBRARY_PATH").expect(message);
    } else if cfg!(all(target_arch = "aarch64", target_os = "linux")) {
        let message = "Libtorch for AARCH64 Linux must be manually installed and set up";
        env::var("LIBTORCH").expect(message);
        env::var("DYLD_LIBRARY_PATH").expect(message);
    }
}
