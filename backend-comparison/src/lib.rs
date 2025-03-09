use std::error::Error;

use tracing_subscriber::filter::LevelFilter;

pub mod burnbenchapp;
pub mod persistence;

/// Simple parse to retrieve additional argument passed to cargo bench command
/// We cannot use clap here as clap parser does not allow to have unknown arguments.
pub fn get_argument<'a>(args: &'a [String], arg_name: &'a str) -> Option<&'a str> {
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            arg if arg == arg_name && i + 1 < args.len() => {
                return Some(&args[i + 1]);
            }
            _ => i += 1,
        }
    }
    None
}

/// Specialized function to retrieve the sharing token
pub fn get_sharing_token(args: &[String]) -> Option<&str> {
    get_argument(args, "--sharing-token")
}

/// Specialized function to retrieve the sharing URL
pub fn get_sharing_url(args: &[String]) -> Option<&str> {
    get_argument(args, "--sharing-url")
}

pub fn init_log() -> Result<(), Box<dyn Error + Send + Sync>> {
    let result = tracing_subscriber::fmt()
        .with_max_level(LevelFilter::DEBUG)
        .without_time()
        .try_init();

    if result.is_ok() {
        update_panic_hook();
    }
    result
}

fn update_panic_hook() {
    let hook = std::panic::take_hook();

    std::panic::set_hook(Box::new(move |info| {
        log::error!("PANIC => {}", info.to_string());
        hook(info);
    }));
}

#[macro_export]
macro_rules! bench_on_backend {
    () => {
        $crate::bench_on_backend!(bench)
    };
    ($fn_name:ident) => {
        use std::env;
        backend_comparison::init_log().unwrap();

        let args: Vec<String> = env::args().collect();
        let url = backend_comparison::get_sharing_url(&args);
        let token = backend_comparison::get_sharing_token(&args);
        #[cfg(feature = "candle-accelerate")]
        let feature_name = "candle-accelerate";
        #[cfg(feature = "candle-cpu")]
        let feature_name = "candle-cpu";
        #[cfg(feature = "candle-cuda")]
        let feature_name = "candle-cuda";
        #[cfg(feature = "candle-metal")]
        let feature_name = "candle-metal";
        #[cfg(feature = "ndarray")]
        let feature_name = "ndarray";
        #[cfg(feature = "ndarray-simd")]
        let feature_name = "ndarray-simd";
        #[cfg(feature = "ndarray-blas-accelerate")]
        let feature_name = "ndarray-blas-accelerate";
        #[cfg(feature = "ndarray-blas-netlib")]
        let feature_name = "ndarray-blas-netlib";
        #[cfg(feature = "ndarray-blas-openblas")]
        let feature_name = "ndarray-blas-openblas";
        #[cfg(feature = "tch-cpu")]
        let feature_name = "tch-cpu";
        #[cfg(feature = "tch-gpu")]
        let feature_name = "tch-gpu";
        #[cfg(feature = "wgpu")]
        let feature_name = "wgpu";
        #[cfg(feature = "wgpu-fusion")]
        let feature_name = "wgpu-fusion";
        #[cfg(feature = "wgpu-spirv")]
        let feature_name = "wgpu-spirv";
        #[cfg(feature = "wgpu-spirv-fusion")]
        let feature_name = "wgpu-spirv-fusion";
        #[cfg(feature = "cuda")]
        let feature_name = "cuda";
        #[cfg(feature = "cuda-fusion")]
        let feature_name = "cuda-fusion";
        #[cfg(feature = "hip")]
        let feature_name = "hip";

        #[cfg(any(feature = "wgpu"))]
        {
            use burn::backend::wgpu::{Wgpu, WgpuDevice};

            $fn_name::<Wgpu<f32, i32>>(&WgpuDevice::default(), feature_name, url, token);
        }

        #[cfg(any(feature = "wgpu-spirv"))]
        {
            use burn::backend::wgpu::{Wgpu, WgpuDevice};

            $fn_name::<Wgpu<half::f16, i32>>(&WgpuDevice::default(), feature_name, url, token);
        }

        #[cfg(feature = "tch-gpu")]
        {
            use burn::backend::{libtorch::LibTorchDevice, LibTorch};

            #[cfg(not(target_os = "macos"))]
            let device = LibTorchDevice::Cuda(0);
            #[cfg(target_os = "macos")]
            let device = LibTorchDevice::Mps;
            $fn_name::<LibTorch<half::f16>>(&device, feature_name, url, token);
        }

        #[cfg(feature = "tch-cpu")]
        {
            use burn::backend::{libtorch::LibTorchDevice, LibTorch};

            let device = LibTorchDevice::Cpu;
            $fn_name::<LibTorch>(&device, feature_name, url, token);
        }

        #[cfg(any(
            feature = "ndarray",
            feature = "ndarray-simd",
            feature = "ndarray-blas-netlib",
            feature = "ndarray-blas-openblas",
            feature = "ndarray-blas-accelerate",
        ))]
        {
            use burn::backend::ndarray::NdArrayDevice;
            use burn::backend::NdArray;

            let device = NdArrayDevice::Cpu;
            $fn_name::<NdArray>(&device, feature_name, url, token);
        }

        #[cfg(feature = "candle-cpu")]
        {
            use burn::backend::candle::CandleDevice;
            use burn::backend::Candle;

            let device = CandleDevice::Cpu;
            $fn_name::<Candle>(&device, feature_name, url, token);
        }

        #[cfg(feature = "candle-cuda")]
        {
            use burn::backend::candle::CandleDevice;
            use burn::backend::Candle;

            let device = CandleDevice::cuda(0);
            $fn_name::<Candle>(&device, feature_name, url, token);
        }

        #[cfg(feature = "candle-metal")]
        {
            use burn::backend::candle::CandleDevice;
            use burn::backend::Candle;

            let device = CandleDevice::metal(0);
            $fn_name::<Candle>(&device, feature_name, url, token);
        }

        #[cfg(feature = "cuda")]
        {
            use burn::backend::cuda::{Cuda, CudaDevice};

            $fn_name::<Cuda<half::f16>>(&CudaDevice::default(), feature_name, url, token);
        }

        #[cfg(feature = "hip")]
        {
            use burn::backend::hip::{Hip, HipDevice};

            $fn_name::<Hip<half::f16>>(&HipDevice::default(), feature_name, url, token);
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::sharing_token_argument_with_value(&["--sharing-token", "token123"], Some("token123"))]
    #[case::sharing_token_argument_no_value(&["--sharing-token"], None)]
    #[case::sharing_token_argument_with_additional_arguments(&["--other-arg", "value", "--sharing-token", "token789"], Some("token789"))]
    #[case::other_argument(&["--other-arg", "value"], None)]
    #[case::no_argument(&[], None)]
    fn test_get_sharing_token(#[case] args: &[&str], #[case] expected: Option<&str>) {
        let args = args.iter().map(|s| s.to_string()).collect::<Vec<String>>();
        assert_eq!(get_sharing_token(&args), expected);
    }

    #[rstest]
    #[case::sharing_url_argument_with_value(&["--sharing-url", "url123"], Some("url123"))]
    #[case::sharing_url_argument_no_value(&["--sharing-url"], None)]
    #[case::sharing_url_argument_with_additional_arguments(&["--other-arg", "value", "--sharing-url", "url789"], Some("url789"))]
    #[case::other_argument(&["--other-arg", "value"], None)]
    #[case::no_argument(&[], None)]
    fn test_get_sharing_url(#[case] args: &[&str], #[case] expected: Option<&str>) {
        let args = args.iter().map(|s| s.to_string()).collect::<Vec<String>>();
        assert_eq!(get_sharing_url(&args), expected);
    }
}
