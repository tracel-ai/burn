//! LPIPS (Learned Perceptual Image Patch Similarity) metric module.
//!
//! LPIPS measures perceptual similarity between images using deep features.
//! Supports VGG16, AlexNet, and SqueezeNet as backbone networks.
//!
//! Reference: "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"
//! <https://arxiv.org/abs/1801.03924>

mod alexnet;
mod metric;
mod squeezenet;
mod vgg;
mod weights;

pub use metric::{Lpips, LpipsAlex, LpipsConfig, LpipsNet, LpipsSqueeze, LpipsVgg};
pub use weights::{get_weights_url, load_pretrained_weights};

// Re-export feature extractors for advanced use cases
pub use alexnet::AlexFeatureExtractor;
pub use squeezenet::{FireModule, SqueezeFeatureExtractor};
pub use vgg::VggFeatureExtractor;
