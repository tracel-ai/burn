fn main() {
    println!("cargo::rustc-check-cfg=cfg(cubecl_backend)");

    // Every runtime feature selects one CubeCL backend. They share one cfg rather than an
    // eight-way `any(...)` at each use, mirroring `burn-dispatch`'s `cube_backend`.
    let runtime = cfg!(feature = "cpu")
        || cfg!(feature = "cuda")
        || cfg!(feature = "rocm")
        || cfg!(feature = "wgpu")
        || cfg!(feature = "wgpu-msl")
        || cfg!(feature = "vulkan")
        || cfg!(feature = "webgpu")
        || cfg!(feature = "metal");

    if runtime {
        println!("cargo::rustc-cfg=cubecl_backend");
    }
}
