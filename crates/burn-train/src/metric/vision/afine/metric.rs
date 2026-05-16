//! A-FINE metric: top-level module, configuration, end-to-end forward
//! pass, `ModuleDisplay`, and property tests.

use burn_core as burn;

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::tensor::Device;
use burn::tensor::Tensor;

use super::calibrators::{
    AfineAdapter, AfineAdapterConfig, FrCalibratorWithLimit, FrCalibratorWithLimitConfig,
    NrCalibrator, NrCalibratorConfig, scale_finalscore,
};
use super::clip_vit::{ClipVisualEncoder, ClipVisualEncoderConfig};
use super::heads::{AfineDHead, AfineDHeadConfig, AfineQHead, AfineQHeadConfig};

/// CLIP RGB normalization mean (per-channel). Applied at the top of
/// `forward` when `normalize_input` is true.
const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
/// CLIP RGB normalization std.
const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_58, 0.275_777_11];

/// Configuration for the A-FINE metric.
#[derive(Config, Debug)]
pub struct AfineConfig {
    /// Expected input resolution. Must be a multiple of 32. Defaults to 256.
    #[config(default = "256")]
    pub image_size: usize,

    /// Apply CLIP RGB normalization inside the forward pass. Set to false
    /// if the caller has already normalized the input. Defaults to true.
    #[config(default = "true")]
    pub normalize_input: bool,
}

impl AfineConfig {
    /// Initialize an A-FINE module with default (random) weights.
    ///
    /// All six learnable submodules are initialized fresh; useful for
    /// shape/property tests but produces meaningless quality scores
    /// until [`Self::init_pretrained`] is called.
    pub fn init(&self, device: &Device) -> Afine {
        assert_eq!(
            self.image_size % 32,
            0,
            "A-FINE image_size must be a multiple of 32, got {}",
            self.image_size
        );

        Afine {
            clip_visual: ClipVisualEncoderConfig::new()
                .with_image_size(self.image_size)
                .init(device),
            qhead: AfineQHeadConfig::new().init(device),
            dhead: AfineDHeadConfig::new().init(device),
            nr_calibrator: NrCalibratorConfig::new().init(device),
            fr_calibrator: FrCalibratorWithLimitConfig::new().init(device),
            adapter: AfineAdapterConfig::new().init(device),
            image_size: self.image_size,
            normalize_input: self.normalize_input,
        }
    }

    /// Initialize an A-FINE module and load pretrained PyIQA weights.
    ///
    /// Downloads `afine.pth` (~600 MB) from the PyIQA Hugging Face
    /// mirror on first call, caches it under
    /// `~/.cache/burn-dataset/afine/`, and loads all six shards into the
    /// matching submodules.
    pub fn init_pretrained(&self, device: &Device) -> Afine {
        let afine = self.init(device);
        super::weights::load_pretrained_weights(afine)
    }
}

/// A-FINE (Adaptive Fidelity-Naturalness Evaluator) full-reference image
/// quality metric built on CLIP ViT-B/32 features.
///
/// `forward(distorted, reference)` returns a per-sample quality score in
/// `(0, 100)` — higher means worse perceptual quality. The metric is
/// **not symmetric**: `forward(a, b) != forward(b, a)` in general,
/// because the adapter term `exp(k * (N_ref - N_dis))` weights the two
/// inputs differently.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Afine {
    pub(crate) clip_visual: ClipVisualEncoder,
    pub(crate) qhead: AfineQHead,
    pub(crate) dhead: AfineDHead,
    pub(crate) nr_calibrator: NrCalibrator,
    pub(crate) fr_calibrator: FrCalibratorWithLimit,
    pub(crate) adapter: AfineAdapter,

    pub(crate) image_size: usize,
    pub(crate) normalize_input: bool,
}

impl Afine {
    /// Compute the A-FINE quality score.
    ///
    /// # Shapes
    /// - `distorted`, `reference`: `[batch, 3, H, W]` with `H` and `W`
    ///   both multiples of 32. Values in `[0, 1]` (RGB) when
    ///   `normalize_input` is true; already-normalized when false.
    /// - returns: `[batch]` per-sample score in `(0, 100)`.
    pub fn forward(&self, distorted: Tensor<4>, reference: Tensor<4>) -> Tensor<1> {
        let [_, _, height, width] = distorted.dims();
        assert_eq!(
            height % 32,
            0,
            "A-FINE input height ({height}) must be a multiple of 32"
        );
        assert_eq!(
            width % 32,
            0,
            "A-FINE input width ({width}) must be a multiple of 32"
        );

        let device = distorted.device();
        let (dis_norm, ref_norm) = if self.normalize_input {
            let mean = Tensor::<4>::from_floats(
                [[[[CLIP_MEAN[0]]], [[CLIP_MEAN[1]]], [[CLIP_MEAN[2]]]]],
                &device,
            );
            let std = Tensor::<4>::from_floats(
                [[[[CLIP_STD[0]]], [[CLIP_STD[1]]], [[CLIP_STD[2]]]]],
                &device,
            );
            (
                (distorted - mean.clone()) / std.clone(),
                (reference - mean) / std,
            )
        } else {
            (distorted, reference)
        };

        // CLIP gives us the 12 per-block patch-feature maps the heads
        // consume; the class-token vector is unused by A-FINE so we ask
        // the encoder to skip it.
        let feat_dis = self
            .clip_visual
            .forward_with_features(dis_norm.clone(), false)
            .features;
        let feat_ref = self
            .clip_visual
            .forward_with_features(ref_norm.clone(), false)
            .features;

        let natural_dis = self.qhead.forward(dis_norm.clone(), &feat_dis);
        let natural_ref = self.qhead.forward(ref_norm.clone(), &feat_ref);
        let natural_dis_scaled = self.nr_calibrator.forward(natural_dis);
        let natural_ref_scaled = self.nr_calibrator.forward(natural_ref);

        let fidelity = self.dhead.forward(dis_norm, ref_norm, &feat_dis, &feat_ref);
        let fidelity_scaled = self.fr_calibrator.forward(fidelity);

        let d = self
            .adapter
            .forward(natural_dis_scaled, natural_ref_scaled, fidelity_scaled);
        let score = scale_finalscore(d); // [B, 1]
        score.squeeze_dim::<1>(1) // [B]
    }
}

impl ModuleDisplay for Afine {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("backbone", &"CLIP ViT-B/32".to_string())
            .add("image_size", &self.image_size.to_string())
            .add("normalize_input", &self.normalize_input.to_string())
            .optional()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    fn small_metric() -> Afine {
        // A small image_size keeps tests fast — 12-layer CLIP at the
        // default 256x256 takes a noticeable fraction of a second per
        // forward.
        let device = Default::default();
        AfineConfig::new().with_image_size(64).init(&device)
    }

    #[test]
    fn afine_forward_shape() {
        let device = Default::default();
        let metric = small_metric();
        let dis = Tensor::<4>::random([2, 3, 64, 64], Distribution::Default, &device);
        let reference = Tensor::<4>::random([2, 3, 64, 64], Distribution::Default, &device);
        let score = metric.forward(dis, reference);
        assert_eq!(score.dims(), [2]);
    }

    #[test]
    fn afine_batch_processing() {
        let device = Default::default();
        let metric = small_metric();
        let dis = Tensor::<4>::random([4, 3, 64, 64], Distribution::Default, &device);
        let reference = Tensor::<4>::random([4, 3, 64, 64], Distribution::Default, &device);
        let score = metric.forward(dis, reference);
        let values = score.into_data().to_vec::<f32>().unwrap();
        assert_eq!(values.len(), 4);
        for v in values {
            assert!(v.is_finite(), "A-FINE produced non-finite value: {v}");
        }
    }

    #[test]
    fn afine_finite_on_constant_inputs() {
        // Guards against the c1=c2=1e-10 epsilon failing to keep the
        // SSIM ratios finite when feature variance is zero.
        let device = Default::default();
        let metric = small_metric();
        let zeros = Tensor::<4>::zeros([1, 3, 64, 64], &device);
        let ones = Tensor::<4>::ones([1, 3, 64, 64], &device);
        let score = metric.forward(zeros, ones);
        let value = score.into_data().to_vec::<f32>().unwrap()[0];
        assert!(value.is_finite(), "got non-finite value {value}");
    }

    #[test]
    fn afine_asymmetry() {
        // D(a, b) != D(b, a) — the adapter weights N_dis by an
        // exponential of (N_ref - N_dis), so swapping inputs flips the
        // sign in the exponent.
        let device = Default::default();
        let metric = small_metric();
        let a = Tensor::<4>::random([1, 3, 64, 64], Distribution::Default, &device);
        let b = Tensor::<4>::random([1, 3, 64, 64], Distribution::Default, &device);
        let forward = metric
            .forward(a.clone(), b.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];
        let reverse = metric.forward(b, a).into_data().to_vec::<f32>().unwrap()[0];
        assert!(
            (forward - reverse).abs() > 1e-6,
            "expected asymmetric output, got fwd={forward}, rev={reverse}"
        );
    }

    #[test]
    fn afine_image_size_must_be_multiple_of_32() {
        let result = std::panic::catch_unwind(|| {
            AfineConfig::new()
                .with_image_size(33)
                .init(&Default::default());
        });
        assert!(result.is_err(), "expected init to panic on bad image_size");
    }

    #[test]
    fn display_afine() {
        let metric = small_metric();
        let formatted = format!("{metric}");
        assert!(
            formatted.contains("CLIP ViT-B/32"),
            "expected backbone name in display output: {formatted}"
        );
    }

    #[test]
    #[ignore = "downloads pre-trained weights"]
    fn test_afine_pretrained_parity() {
        // Numerical parity against PyIQA on a deterministic input pair.
        // Catches silent QuickGELU-vs-GELU swaps, fused-QKV transpose-
        // direction bugs, channel-order mistakes, `c1=c2=1e-10` epsilon
        // drift, and any partially loaded shard — all of which the
        // property tests miss.
        //
        // Expected value captured 2026-04-28 via
        // `Notes/capture_afine_fixtures.py` (torch=2.11.0,
        // pyiqa=0.1.15.post2). Input pair: `arange/(N-1)` vs `arange/N`
        // reshaped to `[1, 3, 224, 224]`. Re-capture if the upstream
        // PyIQA checkpoint or normalization conventions change.
        const D_EXPECTED: f32 = 43.15711594_f32;

        let device = Default::default();
        let metric = AfineConfig::new()
            .with_image_size(224)
            .init_pretrained(&device);

        let total: i64 = 1 * 3 * 224 * 224;
        let dis = Tensor::<1, burn::tensor::Int>::arange(0..total, &device)
            .float()
            .div_scalar((total - 1) as f64)
            .reshape([1, 3, 224, 224]);
        let reference = Tensor::<1, burn::tensor::Int>::arange(0..total, &device)
            .float()
            .div_scalar(total as f64)
            .reshape([1, 3, 224, 224]);

        let value = metric
            .forward(dis, reference)
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];

        // Tolerance: relative 5e-3 covers fp32 attention drift across
        // 12 transformer layers. Absolute floor of 5e-3 prevents a
        // captured value near zero from forcing absurd precision. In
        // practice the burn output matches the captured PyIQA scalar to
        // within ~5e-5 absolute, well inside this budget.
        let tolerance = D_EXPECTED.abs() * 5e-3 + 5e-3;
        assert!(
            (value - D_EXPECTED).abs() < tolerance,
            "expected {D_EXPECTED}, got {value} (tolerance {tolerance})"
        );
    }

    #[test]
    #[ignore = "downloads pre-trained weights"]
    fn test_afine_pretrained() {
        // Loads the real PyIQA checkpoint (~600 MB on first run, cached
        // afterwards) and verifies the forward pass produces a finite
        // score that meaningfully differs from the random-init metric.
        // The differs-from-random check guards against a silent
        // `allow_partial(true)` skip — if a regex remap is wrong the
        // weights stay at random init and the property tests still
        // pass, but the pretrained output should land in a very
        // different region of the score range.
        let device = Default::default();
        let dis = Tensor::<4>::random([1, 3, 256, 256], Distribution::Default, &device);
        let reference = Tensor::<4>::random([1, 3, 256, 256], Distribution::Default, &device);

        let random_metric = AfineConfig::new().init(&device);
        let random_score = random_metric
            .forward(dis.clone(), reference.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];

        let pretrained_metric = AfineConfig::new().init_pretrained(&device);
        let pretrained_score = pretrained_metric
            .forward(dis, reference)
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];

        assert!(
            pretrained_score.is_finite(),
            "pretrained A-FINE produced non-finite output: {pretrained_score}"
        );
        assert!(
            (pretrained_score - random_score).abs() > 1e-3,
            "pretrained output ({pretrained_score}) too close to random-init ({random_score}) — \
             a shard may have failed to load silently"
        );
    }
}
