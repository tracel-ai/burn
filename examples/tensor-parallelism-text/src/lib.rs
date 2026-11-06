pub mod inference;
pub mod split;

use burn::{prelude::*, tensor::DeviceType};

/// Where `text-classification`'s `ag-news-train` saves the model it trains.
pub const AG_NEWS_ARTIFACTS: &str = "/tmp/text-classification-ag-news";

/// The artifact directory named on the command line, or [`AG_NEWS_ARTIFACTS`].
pub fn artifact_dir() -> String {
    std::env::args()
        .nth(1)
        .unwrap_or_else(|| AG_NEWS_ARTIFACTS.to_string())
}

/// Every GPU of the first enabled runtime that has one, or two Flex devices, so the split is a
/// real one on a machine without a GPU.
pub fn devices() -> Vec<Device> {
    let runtimes: &[DeviceType] = &[
        #[cfg(feature = "cuda")]
        DeviceType::Cuda,
        #[cfg(feature = "rocm")]
        DeviceType::Rocm,
        #[cfg(feature = "vulkan")]
        DeviceType::Vulkan,
    ];

    runtimes
        .iter()
        .map(|runtime| Device::enumerate(runtime.clone()).into_vec())
        .find(|devices| !devices.is_empty())
        .unwrap_or_else(|| vec![Device::flex(), Device::flex()])
}
