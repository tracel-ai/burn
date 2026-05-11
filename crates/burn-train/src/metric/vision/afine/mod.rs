//! A-FINE (Adaptive Fidelity-Naturalness Evaluator) metric.
//!
//! Full-reference perceptual image quality metric that combines a naturalness
//! branch and a fidelity branch over CLIP ViT-B/32 features.
//!
//! Reference: "A Novel Fidelity-Naturalness Evaluator for Image Quality Assessment"
//! <https://arxiv.org/abs/2503.11221>

mod calibrators;
mod clip_attention;
mod clip_vit;
mod heads;
mod metric;
mod quick_gelu;
mod weights;

pub use clip_vit::{ClipOutput, ClipVisualEncoder, ClipVisualEncoderConfig};
pub use heads::{AfineDHead, AfineDHeadConfig, AfineQHead, AfineQHeadConfig};
pub use metric::{Afine, AfineConfig};
