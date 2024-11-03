fn main() {
    #[cfg(feature = "ndarray")]
    {
        burn_router_websocket::server::start::<burn_ndarray::NdArray>(
            Default::default(),
            "0.0.0.0:3000",
        );
    }
    #[cfg(feature = "wgpu")]
    {
        burn_router_websocket::server::start::<burn_wgpu::Wgpu>(Default::default(), "0.0.0.0:3000");
    }
    #[cfg(feature = "cuda")]
    {
        burn_router_websocket::server::start::<burn_cuda::Cuda>(Default::default(), "0.0.0.0:3000");
    }
}
