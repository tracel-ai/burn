fn main() {
    println!("cargo::rustc-check-cfg=cfg(wgpu_metal)");
    println!("cargo::rustc-check-cfg=cfg(wgpu_vulkan)");
    println!("cargo::rustc-check-cfg=cfg(wgpu_webgpu)");

    // Detect which single wgpu backend is enabled
    let mut metal = cfg!(feature = "metal");
    let mut vulkan = cfg!(feature = "vulkan");
    let mut webgpu = cfg!(feature = "webgpu");
    let wgpu_only = cfg!(all(
        feature = "wgpu",
        not(feature = "metal"),
        not(feature = "vulkan")
    ));
    let enabled = [(metal, "metal"), (vulkan, "vulkan"), (webgpu, "webgpu")]
        .iter()
        .filter(|x| x.0)
        .map(|x| x.1)
        .collect::<Vec<_>>();

    // WGPU features are mutually exclusive, but we don't want to workspace to throw a compile error.
    // In workspace builds with multiple features, we emit a warning and fallback to WebGpu/Wgpu.
    if enabled.len() > 1 {
        webgpu = false;
        let fallback = if metal {
            vulkan = false;
            "metal"
        } else {
            metal = false;
            "vulkan"
        };
        println!(
            "cargo:warning=Only one WGPU backend can be enabled at once. Detected: [{}]. Falling back to {fallback}. For production, enable only one of: metal, vulkan, or webgpu.",
            enabled.join(", ")
        );
    }

    if metal {
        println!("cargo:rustc-cfg=wgpu_metal");
    }
    if vulkan {
        println!("cargo:rustc-cfg=wgpu_vulkan");
    }
    if webgpu || wgpu_only {
        println!("cargo:rustc-cfg=wgpu_webgpu");
    }
}
