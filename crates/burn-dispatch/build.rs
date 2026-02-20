fn main() {
    println!("cargo::rustc-check-cfg=cfg(wgpu_metal)");
    println!("cargo::rustc-check-cfg=cfg(wgpu_vulkan)");
    println!("cargo::rustc-check-cfg=cfg(wgpu_webgpu)");

    // Detect which single wgpu backend is enabled
    let metal = cfg!(feature = "metal");
    let vulkan = cfg!(feature = "vulkan");
    let webgpu = cfg!(feature = "webgpu");
    let enabled = [(metal, "metal"), (vulkan, "vulkan"), (webgpu, "webgpu")]
        .iter()
        .filter(|x| x.0)
        .map(|x| x.1)
        .collect::<Vec<_>>();

    // WGPU features are mutually exclusive, but we don't want to workspace to throw a compile error.
    // In workspace builds with multiple features, we emit a warning and disable all WGPU backends.
    if enabled.len() > 1 {
        println!(
            "cargo:warning=Only one WGPU backend can be enabled at once. Detected: [{}]. No WGPU backend will be available in this build. This is expected in workspace builds. For production, enable only one of: metal, vulkan, or webgpu.",
            enabled.join(", ")
        );
        return;
    }

    if metal {
        println!("cargo:rustc-cfg=wgpu_metal");
    }
    if vulkan {
        println!("cargo:rustc-cfg=wgpu_vulkan");
    }
    if webgpu {
        println!("cargo:rustc-cfg=wgpu_webgpu");
    }
}
