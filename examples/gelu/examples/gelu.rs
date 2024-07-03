fn main() {
    #[cfg(feature = "cuda")]
    gelu::launch::<burn_cuda::CudaRuntime>(&Default::default());
    #[cfg(feature = "wgpu")]
    gelu::launch::<burn_wgpu::WgpuRuntime>(&Default::default());
}
