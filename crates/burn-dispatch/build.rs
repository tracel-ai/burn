fn main() {
    println!("cargo::rustc-check-cfg=cfg(cube_backend)");
    println!("cargo::rustc-check-cfg=cfg(backend_enabled)");

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

    // Backend-free builds expose tensor/model APIs without installing an execution backend.
    if cuda
        || flex
        || rocm
        || ndarray
        || tch
        || cpu
        || metal
        || vulkan
        || webgpu
        || wgpu
        || cfg!(feature = "remote")
        || cfg!(feature = "capture")
    {
        println!("cargo::rustc-cfg=backend_enabled");
    }

    // Every cubecl-backed feature selects the same backend type now — the
    // runtime is what the device says, not what the type is — so they share one
    // variant, under one cfg rather than a seven-way list at each use.
    if cuda || rocm || cpu || metal || vulkan || webgpu || wgpu {
        println!("cargo:rustc-cfg=cube_backend");
    }
}
