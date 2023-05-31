use std::{env, error::Error};

/// This build script is used to detect the target architecture and set the
/// correct cfg flag.
fn main() -> Result<(), Box<dyn Error>> {
    let target = env::var("TARGET")?;

    // Set the correct cfg flag depending on the target architecture.
    // This is used to enable the portable atomic implementation.
    if target.starts_with("thumbv6m-") {
        println!("cargo:rustc-cfg=use_portable_atomics");
    }
    Ok(())
}
