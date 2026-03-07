//! DISTS (Deep Image Structure and Texture Similarity) metric.
//!
//! This module implements DISTS, a full-reference image quality assessment metric
//! that combines structure and texture similarity using deep features.
//!
//! Reference: "Image Quality Assessment: Unifying Structure and Texture Similarity"
//! https://arxiv.org/abs/2004.07728

mod l2pool;
mod metric;
mod vgg16_l2pool;
mod weights;

pub use metric::{Dists, DistsConfig};
