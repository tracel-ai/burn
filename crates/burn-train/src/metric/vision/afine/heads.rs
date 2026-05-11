//! Naturalness and fidelity heads.
//!
//! Both heads consume the 12 per-block CLIP feature maps plus the raw
//! RGB image (re-introduced as a "level 0" feature). The heads' `mean`
//! and `std` buffers are CLIP's normalization constants, used here as
//! the **inverse** of preprocessing — they de-normalize the input back
//! into pixel space before computing raw-RGB statistics.
//!
//! - [`AfineQHead`] — naturalness, single-image, returns `[B, 1]`.
//! - [`AfineDHead`] — fidelity, two-image, returns `[B, 1]`.
//!
//! The `chns` tuple has 13 entries: `[3, 768, 768, ..., 768]`. The
//! leading 3 is the raw image; the remaining twelve 768s are the CLIP
//! transformer outputs after each block.

use burn_core as burn;

use burn::config::Config;
use burn::module::{Module, Param};
use burn::tensor::Device;
use burn::tensor::Tensor;
use burn::tensor::activation::{relu, softplus};
use burn_nn::{Gelu, Linear, LinearConfig};

/// CLIP RGB normalization mean (per-channel). Used as a de-normalization
/// constant inside the heads.
const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
/// CLIP RGB normalization std.
const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_58, 0.275_777_11];

/// Per-level channel counts: raw RGB (3) plus 12 CLIP layers (768 each).
const CHNS: [usize; 13] = [
    3, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768,
];
const NUM_LEVELS: usize = 13;

/// Hidden dim of the shared per-CLIP-level projection (`proj_feat`).
const Q_HIDDEN_DIM: usize = 128;
/// Output dim of the first MLP layer in `proj_head`.
const Q_PROJ_HEAD_OUT: usize = 768;
/// Input dim of `proj_head`: 6 (raw, k=0) + 128 * 12 (projected, k=1..=12).
const Q_PROJ_HEAD_IN: usize = 6 + Q_HIDDEN_DIM * 12;

/// Sum of `CHNS`. `alpha` and `beta` are length `D_CHNS_SUM` along the
/// last axis.
const D_CHNS_SUM: usize = 9219;

/// Numerical-stability epsilon used in the SSIM-like ratios. **Must be
/// 1e-10**, not the larger 1e-6 that DISTS uses; the feature magnitudes
/// after CLIP + ReLU are unit-scale-ish, so DISTS's eps would overwhelm
/// the signal.
const EPS: f64 = 1e-10;

fn build_clip_mean_std(device: &Device) -> (Tensor<4>, Tensor<4>) {
    let mean = Tensor::from_floats(
        [[[[CLIP_MEAN[0]]], [[CLIP_MEAN[1]]], [[CLIP_MEAN[2]]]]],
        device,
    );
    let std = Tensor::from_floats(
        [[[[CLIP_STD[0]]], [[CLIP_STD[1]]], [[CLIP_STD[2]]]]],
        device,
    );
    (mean, std)
}

/// Mean and biased variance over the token axis, returned as a 2-D
/// tensor `[B, 2 * channels]` (mean followed by var, both flattened).
fn level_mean_var(feat: Tensor<3>) -> (Tensor<3>, Tensor<3>, Tensor<2>) {
    let mean = feat.clone().mean_dim(1);
    let centered = feat - mean.clone();
    let var = centered.powi_scalar(2).mean_dim(1);
    let descriptor = Tensor::cat(
        vec![
            mean.clone().flatten::<2>(1, 2),
            var.clone().flatten::<2>(1, 2),
        ],
        1,
    );
    (mean, var, descriptor)
}

/// Configuration for [`AfineQHead`].
#[derive(Config, Debug)]
pub struct AfineQHeadConfig {}

impl AfineQHeadConfig {
    /// Initialize the naturalness head with random weights.
    pub fn init(&self, device: &Device) -> AfineQHead {
        let (mean, std) = build_clip_mean_std(device);
        AfineQHead {
            mean,
            std,
            proj_feat: LinearConfig::new(2 * 768, Q_HIDDEN_DIM)
                .with_bias(true)
                .init(device),
            proj_head_fc1: LinearConfig::new(Q_PROJ_HEAD_IN, Q_PROJ_HEAD_OUT)
                .with_bias(true)
                .init(device),
            proj_head_fc2: LinearConfig::new(Q_PROJ_HEAD_OUT, 1)
                .with_bias(true)
                .init(device),
            activation: Gelu::new(),
        }
    }
}

/// A-FINE naturalness head. Per-level mean+variance over CLIP tokens,
/// shared projection on the 12 CLIP levels, then a small MLP to a
/// scalar. The level-0 raw-image descriptor (`[B, 6]`) is concatenated
/// directly without going through `proj_feat`.
///
/// Activation is the erf-based [`burn_nn::Gelu`], not the QuickGELU used
/// elsewhere in this crate. PyIQA's reference is explicit on this.
#[derive(Module, Debug)]
pub struct AfineQHead {
    pub(crate) mean: Tensor<4>,
    pub(crate) std: Tensor<4>,
    pub(crate) proj_feat: Linear,
    pub(crate) proj_head_fc1: Linear,
    pub(crate) proj_head_fc2: Linear,
    pub(crate) activation: Gelu,
}

impl AfineQHead {
    /// Compute the naturalness score for one image plus its CLIP
    /// features.
    ///
    /// # Shapes
    /// - `image`: `[B, 3, H, W]`, **already CLIP-normalized**.
    /// - `clip_features`: 12 levels of `[B, num_patches, 768]`.
    /// - returns: `[B, 1]`.
    pub fn forward(&self, image: Tensor<4>, clip_features: &[Tensor<3>]) -> Tensor<2> {
        assert_eq!(
            clip_features.len(),
            12,
            "AfineQHead expects 12 CLIP feature maps, got {}",
            clip_features.len()
        );
        let [batch, channels, height, width] = image.dims();

        // De-normalize: x = x * std + mean.
        let img = image * self.std.clone() + self.mean.clone();

        // [B, 3, H, W] -> [B, H*W, 3]: token-major raw-image features.
        let img_feat = img
            .reshape([batch, channels, height * width])
            .swap_dims(1, 2);

        let mut level_descriptors: Vec<Tensor<2>> = Vec::with_capacity(NUM_LEVELS);

        // Level 0: raw image, no projection.
        let (_, _, raw_descriptor) = level_mean_var(img_feat);
        level_descriptors.push(raw_descriptor);

        // Levels 1..=12: ReLU'd CLIP features, then shared `proj_feat`.
        for h in clip_features {
            let activated = relu(h.clone());
            let (_, _, descriptor) = level_mean_var(activated);
            level_descriptors.push(self.proj_feat.forward(descriptor));
        }

        let concat_all = Tensor::cat(level_descriptors, 1);
        let hidden = self
            .activation
            .forward(self.proj_head_fc1.forward(concat_all));
        self.proj_head_fc2.forward(hidden)
    }
}

/// Configuration for [`AfineDHead`].
#[derive(Config, Debug)]
pub struct AfineDHeadConfig {}

impl AfineDHeadConfig {
    /// Initialize the fidelity head with random weights.
    pub fn init(&self, device: &Device) -> AfineDHead {
        let (mean, std) = build_clip_mean_std(device);
        // PyIQA initializes alpha/beta with `.normal_(0.1, 0.01)`. That
        // distribution doesn't really matter under random init — values
        // are always overwritten by the checkpoint at `init_pretrained`
        // time — but we mirror the magnitude here so the random-init
        // forward stays well-conditioned.
        let alpha = Tensor::random(
            [1, 1, D_CHNS_SUM],
            burn::tensor::Distribution::Normal(0.1, 0.01),
            device,
        );
        let beta = Tensor::random(
            [1, 1, D_CHNS_SUM],
            burn::tensor::Distribution::Normal(0.1, 0.01),
            device,
        );
        AfineDHead {
            mean,
            std,
            alpha: Param::from_tensor(alpha),
            beta: Param::from_tensor(beta),
        }
    }
}

/// A-FINE fidelity head. SSIM-like luminance and contrast statistics
/// across 13 levels, weighted by globally-normalized softplus(alpha) and
/// softplus(beta).
#[derive(Module, Debug)]
pub struct AfineDHead {
    pub(crate) mean: Tensor<4>,
    pub(crate) std: Tensor<4>,
    /// Luminance weights, shape `[1, 1, sum(chns)] = [1, 1, 9219]`.
    pub(crate) alpha: Param<Tensor<3>>,
    /// Contrast weights, same shape as `alpha`.
    pub(crate) beta: Param<Tensor<3>>,
}

impl AfineDHead {
    /// Compute the fidelity score between distorted and reference
    /// images.
    ///
    /// # Shapes
    /// - `distorted`, `reference`: `[B, 3, H, W]`, both CLIP-normalized.
    /// - `feat_dis`, `feat_ref`: 12 levels of `[B, num_patches, 768]`.
    /// - returns: `[B, 1]`.
    pub fn forward(
        &self,
        distorted: Tensor<4>,
        reference: Tensor<4>,
        feat_dis: &[Tensor<3>],
        feat_ref: &[Tensor<3>],
    ) -> Tensor<2> {
        assert_eq!(feat_dis.len(), 12);
        assert_eq!(feat_ref.len(), 12);
        let [batch, channels, height, width] = distorted.dims();

        // De-normalize both images and lay them out as token sequences.
        let raw_x = (distorted * self.std.clone() + self.mean.clone())
            .reshape([batch, channels, height * width])
            .swap_dims(1, 2);
        let raw_y = (reference * self.std.clone() + self.mean.clone())
            .reshape([batch, channels, height * width])
            .swap_dims(1, 2);

        // 13-entry feature lists: raw RGB at level 0, ReLU'd CLIP
        // features at levels 1..=12.
        let mut feat_x: Vec<Tensor<3>> = Vec::with_capacity(NUM_LEVELS);
        let mut feat_y: Vec<Tensor<3>> = Vec::with_capacity(NUM_LEVELS);
        feat_x.push(raw_x);
        feat_y.push(raw_y);
        for h in feat_dis {
            feat_x.push(relu(h.clone()));
        }
        for h in feat_ref {
            feat_y.push(relu(h.clone()));
        }

        // Global softplus normalization. `alpha_/w_sum + beta_/w_sum`
        // sums to ~1 across all 13 levels and both terms. We keep
        // `w_sum` as a tensor and broadcast-divide so the autodiff graph
        // stays intact for any future training-mode caller.
        let alpha_sp = softplus(self.alpha.val(), 1.0);
        let beta_sp = softplus(self.beta.val(), 1.0);
        let w_sum = (alpha_sp.clone().sum() + beta_sp.clone().sum())
            .add_scalar(EPS)
            .reshape([1, 1, 1]);
        let alpha_norm = alpha_sp / w_sum.clone();
        let beta_norm = beta_sp / w_sum;

        let mut dist1: Option<Tensor<3>> = None;
        let mut dist2: Option<Tensor<3>> = None;
        let mut offset: usize = 0;

        for k in 0..NUM_LEVELS {
            let cn = CHNS[k];
            let alpha_k = alpha_norm.clone().slice([0..1, 0..1, offset..offset + cn]);
            let beta_k = beta_norm.clone().slice([0..1, 0..1, offset..offset + cn]);

            let xm = feat_x[k].clone().mean_dim(1); // [B, 1, cn]
            let ym = feat_y[k].clone().mean_dim(1);

            // S1 (luminance): (2 * x_mean * y_mean + eps)
            //                 / (x_mean^2 + y_mean^2 + eps).
            let s1_num = (xm.clone() * ym.clone()).mul_scalar(2.0).add_scalar(EPS);
            let s1_den = xm.clone().powi_scalar(2) + ym.clone().powi_scalar(2);
            let s1 = s1_num / s1_den.add_scalar(EPS);
            let term1 = (alpha_k * s1).sum_dim(2); // [B, 1, 1]
            dist1 = Some(match dist1 {
                None => term1,
                Some(d) => d + term1,
            });

            // S2 (contrast/structure):
            //   var_x = mean((x - x_mean)^2) (biased),
            //   var_y = mean((y - y_mean)^2),
            //   cov_xy = mean(x*y) - x_mean * y_mean,
            //   S2 = (2 * cov_xy + eps) / (var_x + var_y + eps).
            let xc = feat_x[k].clone() - xm.clone();
            let yc = feat_y[k].clone() - ym.clone();
            let x_var = xc.powi_scalar(2).mean_dim(1);
            let y_var = yc.powi_scalar(2).mean_dim(1);
            let xy_mean = (feat_x[k].clone() * feat_y[k].clone()).mean_dim(1);
            let xy_cov = xy_mean - xm * ym;
            let s2_num = xy_cov.mul_scalar(2.0).add_scalar(EPS);
            let s2_den = (x_var + y_var).add_scalar(EPS);
            let s2 = s2_num / s2_den;
            let term2 = (beta_k * s2).sum_dim(2);
            dist2 = Some(match dist2 {
                None => term2,
                Some(d) => d + term2,
            });

            offset += cn;
        }

        let total = dist1.unwrap() + dist2.unwrap(); // [B, 1, 1]
        let score = total.ones_like() - total; // 1 - (dist1 + dist2)
        score.squeeze_dim::<2>(2) // [B, 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    #[test]
    fn afine_q_head_forward_shape() {
        let device = Default::default();
        let head = AfineQHeadConfig::new().init(&device);

        let image = Tensor::<4>::random([2, 3, 64, 64], Distribution::Default, &device);
        let features: Vec<_> = (0..12)
            .map(|_| Tensor::<3>::random([2, 4, 768], Distribution::Default, &device))
            .collect();

        let out = head.forward(image, &features);
        assert_eq!(out.dims(), [2, 1]);
    }

    #[test]
    fn afine_d_head_forward_shape() {
        let device = Default::default();
        let head = AfineDHeadConfig::new().init(&device);

        let dis = Tensor::<4>::random([2, 3, 64, 64], Distribution::Default, &device);
        let reference = Tensor::<4>::random([2, 3, 64, 64], Distribution::Default, &device);
        let feat_dis: Vec<_> = (0..12)
            .map(|_| Tensor::<3>::random([2, 4, 768], Distribution::Default, &device))
            .collect();
        let feat_ref: Vec<_> = (0..12)
            .map(|_| Tensor::<3>::random([2, 4, 768], Distribution::Default, &device))
            .collect();

        let out = head.forward(dis, reference, &feat_dis, &feat_ref);
        assert_eq!(out.dims(), [2, 1]);
    }

    #[test]
    fn afine_d_head_identical_inputs_unit_score() {
        // When dis == ref and CLIP features match, S1 and S2 should both
        // be ≈ 1, so dist1 + dist2 ≈ 1 and the final score 1 - (...) ≈ 0.
        // Random init alpha/beta sum to 1 so this is a property-test
        // sanity check on the global normalization.
        let device = Default::default();
        let head = AfineDHeadConfig::new().init(&device);

        let image = Tensor::<4>::random([1, 3, 32, 32], Distribution::Default, &device);
        let features: Vec<_> = (0..12)
            .map(|_| Tensor::<3>::random([1, 1, 768], Distribution::Default, &device))
            .collect();

        let out = head.forward(image.clone(), image, &features.clone(), &features);
        let value = out.into_data().to_vec::<f32>().unwrap()[0];
        assert!(
            value.abs() < 0.1,
            "fidelity head on identical inputs should yield ~0, got {value}"
        );
    }
}
