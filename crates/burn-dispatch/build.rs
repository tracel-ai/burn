fn main() {
    println!("cargo::rustc-check-cfg=cfg(cube_backend)");
    println!("cargo::rustc-check-cfg=cfg(executing_backend)");
    println!("cargo::rustc-check-cfg=cfg(backend_enabled)");

    let cuda = cfg!(feature = "cuda");
    let flex = cfg!(feature = "flex");
    let rocm = cfg!(feature = "rocm");
    let cpu = cfg!(feature = "cpu");
    let metal = cfg!(feature = "metal");
    let vulkan = cfg!(feature = "vulkan");
    let webgpu = cfg!(feature = "webgpu");
    let wgpu = cfg!(feature = "wgpu");

    let remote = cfg!(feature = "remote");
    let capture = cfg!(feature = "capture");
    // A group's members run Cube or Flex, so it executes nothing of its own, but it still puts a
    // variant in the dispatch enums.
    let group = cfg!(feature = "group");

    let cube = cuda || rocm || cpu || metal || vulkan || webgpu || wgpu;
    let local = cube || flex;
    let executing = local || remote;

    // Backend-free builds expose tensor/model APIs without installing an execution backend.
    if executing || capture || group {
        println!("cargo::rustc-cfg=backend_enabled");
    }

    // Every cubecl-backed feature selects the same backend type, since the device says which
    // runtime, so they share one variant under one cfg rather than a seven-way list at each use.
    if cube {
        println!("cargo::rustc-cfg=cube_backend");
    }

    // Capture records operations but never executes them.
    if executing {
        println!("cargo::rustc-cfg=executing_backend");
    }
}
