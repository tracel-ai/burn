#![recursion_limit = "141"]

pub fn start() {
    let port = std::env::var("REMOTE_BACKEND_PORT")
        .map(|port| match port.parse::<u16>() {
            Ok(val) => val,
            Err(err) => panic!("Invalid port, got {port} with error {err}"),
        })
        .unwrap_or(3000);

    cfg_if::cfg_if! {
        if #[cfg(feature = "ndarray")]{
            burn::server::start::<burn::backend::NdArray>(Default::default(), port);
        } else if #[cfg(feature = "cuda")]{
            burn::server::start::<burn::backend::Cuda>(Default::default(), port);
        } else if #[cfg(feature = "webgpu")] {
            burn::server::start::<burn::backend::WebGpu>(Default::default(), port);
        } else if #[cfg(feature = "vulkan")] {
            burn::server::start::<burn::backend::Vulkan>(Default::default(), port);
        } else {
            panic!("No backend selected, can't start server on port {port}");
        }
    }
}
