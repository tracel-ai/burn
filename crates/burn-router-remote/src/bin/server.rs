fn main() {
    #[cfg(feature = "ndarray")]
    {
        burn_router_remote::start_server::<burn_ndarray::NdArray>(Default::default(), 3000);
    }
    #[cfg(feature = "wgpu")]
    {
        burn_router_remote::start_server::<burn_wgpu::Wgpu>(Default::default(), 3000);
    }
    #[cfg(feature = "cuda")]
    {
        burn_router_remote::start_server::<burn_cuda::Cuda>(Default::default(), 3000);
    }
}
