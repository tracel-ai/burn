fn main() {
    println!("cargo::rustc-check-cfg=cfg(default_backend)");

    // If you try to build with `--no-default-features`, we enable a cpu backend by default
    let cuda = cfg!(feature = "cuda");
    let flex = cfg!(feature = "flex");
    let rocm = cfg!(feature = "rocm");
    let ndarray = cfg!(feature = "ndarray");
    let tch = cfg!(feature = "tch");
    let cpu = cfg!(feature = "cpu");
    let metal = cfg!(feature = "metal");
    let vulkan = cfg!(feature = "vulkan");
    let webgpu = cfg!(feature = "webgpu");
    let wgpu = cfg!(feature = "wgpu");

    let no_backend_enabled =
        !(cuda || flex || rocm || ndarray || tch || cpu || metal || vulkan || webgpu || wgpu);

    if no_backend_enabled {
        println!("cargo:rustc-cfg=default_backend");
    }
}
