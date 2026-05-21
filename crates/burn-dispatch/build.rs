fn main() {
    println!("cargo::rustc-check-cfg=cfg(wgpu_metal)");
    println!("cargo::rustc-check-cfg=cfg(wgpu_vulkan)");
    println!("cargo::rustc-check-cfg=cfg(wgpu_webgpu)");
    println!("cargo::rustc-check-cfg=cfg(default_backend)");

    // If you try to build with `--no-default-features`, we enable a cpu backend by default (Flex)
    let cuda = cfg!(feature = "cuda");
    let flex = cfg!(feature = "flex");
    let rocm = cfg!(feature = "rocm");
    let ndarray = cfg!(feature = "ndarray");
    let tch = cfg!(feature = "tch");
    let cpu = cfg!(feature = "cpu");

    let mut metal = cfg!(feature = "metal");
    let mut vulkan = cfg!(feature = "vulkan");
    let mut webgpu = cfg!(feature = "webgpu");

    // Detect which single wgpu backend is enabled
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
        webgpu = true;
        vulkan = false;
        metal = false;
        println!(
            "cargo:warning=Only one wgpu backend can be enabled at once. Detected: [{}]. Falling back to `wgpu`. For production, enable only one of: metal, vulkan, or webgpu.",
            enabled.join(", ")
        );
    }

    let no_backend_enabled =
        !(cuda || flex || rocm || ndarray || tch || cpu || metal || vulkan || webgpu || wgpu_only);

    if metal {
        println!("cargo:rustc-cfg=wgpu_metal");
    }
    if vulkan {
        println!("cargo:rustc-cfg=wgpu_vulkan");
    }
    if webgpu || wgpu_only {
        println!("cargo:rustc-cfg=wgpu_webgpu");
    }
    if no_backend_enabled {
        println!("cargo:rustc-cfg=default_backend");
    }
}
