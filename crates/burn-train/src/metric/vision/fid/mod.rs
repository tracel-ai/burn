//! Frechet Inception Distance (FID) metric.
//!
//! Measures the distance between distributions of generated and real images
//! using InceptionV3 features. Lower FID = higher quality and diversity.
//!
//! Reference: <https://arxiv.org/abs/1706.08500>

mod inception;
mod metric;
mod weights;

pub use inception::InceptionV3FeatureExtractor;
pub use metric::{Fid, FidConfig};
