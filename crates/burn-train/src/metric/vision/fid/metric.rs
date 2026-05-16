use burn_core as burn;

use burn::config::Config;
use burn::module::Module;
use burn::tensor::Device;
use burn::tensor::Tensor;
use burn::tensor::linalg;

use super::inception::InceptionV3FeatureExtractor;

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
const EPS: f64 = 1e-6;

/// Configuration for [Fid].
///
/// ```ignore
/// let fid = FidConfig::new().init_pretrained(&device);
/// let score = fid.forward(real_images, generated_images);
/// ```
#[derive(Config, Debug)]
pub struct FidConfig {
    /// Normalize input images from [0,1] to ImageNet range.
    #[config(default = true)]
    pub normalize: bool,

    /// Number of Newton-Schulz iterations for matrix square root.
    #[config(default = 50)]
    pub num_iterations: usize,
}

impl FidConfig {
    /// Initialize with pretrained InceptionV3 weights from pytorch-fid.
    pub fn init_pretrained(&self, device: &Device) -> Fid {
        let fid = self.init(device);
        super::weights::load_pretrained_weights(fid)
    }

    /// Initialize with random weights.
    pub fn init(&self, device: &Device) -> Fid {
        Fid {
            extractor: InceptionV3FeatureExtractor::new(device),
            normalize: self.normalize,
            num_iterations: self.num_iterations,
        }
    }
}

/// Frechet Inception Distance metric.
///
/// Computes the Frechet distance between feature distributions of real and
/// generated images using an InceptionV3 feature extractor.
///
/// ```ignore
/// let fid = FidConfig::new().init_pretrained(&device);
/// let feats_real = fid.extract_features(real_images);
/// let feats_gen = fid.extract_features(generated_images);
/// let score = fid.compute_fid(feats_real, feats_gen);
/// ```
#[derive(Module, Debug)]
pub struct Fid {
    extractor: InceptionV3FeatureExtractor,
    normalize: bool,
    num_iterations: usize,
}

impl Fid {
    /// Extract 2048-dim InceptionV3 features from images of shape `[batch, 3, H, W]`.
    pub fn extract_features(&self, images: Tensor<4>) -> Tensor<2> {
        let images = if self.normalize {
            imagenet_normalize(images)
        } else {
            images
        };
        self.extractor.forward(images)
    }

    /// Compute FID from pre-extracted feature tensors of shape `[N, 2048]`.
    pub fn compute_fid(&self, features_real: Tensor<2>, features_gen: Tensor<2>) -> Tensor<1> {
        let (mu1, sigma1) = compute_statistics(features_real);
        let (mu2, sigma2) = compute_statistics(features_gen);
        frechet_distance(mu1, sigma1, mu2, sigma2, self.num_iterations)
    }

    /// Compute FID end-to-end from image tensors of shape `[N, 3, H, W]` in [0,1].
    pub fn forward(&self, images_real: Tensor<4>, images_gen: Tensor<4>) -> Tensor<1> {
        let features_real = self.extract_features(images_real);
        let features_gen = self.extract_features(images_gen);
        self.compute_fid(features_real, features_gen)
    }
}

fn imagenet_normalize(x: Tensor<4>) -> Tensor<4> {
    let device = x.device();

    let mean = Tensor::<1>::from_floats(IMAGENET_MEAN, &device).reshape([1, 3, 1, 1]);
    let std = Tensor::<1>::from_floats(IMAGENET_STD, &device).reshape([1, 3, 1, 1]);

    x.sub(mean).div(std)
}

/// Mean vector `[D]` and unbiased covariance matrix `[D, D]` from feature rows `[N, D]`.
fn compute_statistics(features: Tensor<2>) -> (Tensor<1>, Tensor<2>) {
    let [n, d] = features.dims();
    let n_f = n as f64;

    let mean = features.clone().mean_dim(0).squeeze_dim::<1>(0);
    let centered = features.sub(mean.clone().unsqueeze_dim::<2>(0).expand([n, d]));

    let cov = centered
        .clone()
        .transpose()
        .matmul(centered)
        .div_scalar(n_f - 1.0);

    (mean, cov)
}

/// Newton-Schulz iteration for the square root of a symmetric PD matrix.
/// Input must be symmetric positive-definite for convergence.
fn matrix_sqrt_newton_schulz(a: Tensor<2>, num_iterations: usize) -> Tensor<2> {
    let [d, _] = a.dims();
    let device = a.device();

    // Clamp to avoid division by near-zero norms (also avoids a GPU sync).
    let norm_a = a.clone().mul(a.clone()).sum().sqrt().clamp_min(EPS);

    let identity = Tensor::<2>::eye(d, &device);
    let mut y = a.div(norm_a.clone().unsqueeze_dim::<2>(0).expand([d, d]));
    let mut z = identity.clone();
    let three_i = identity.clone().mul_scalar(3.0);

    for _ in 0..num_iterations {
        let t = three_i
            .clone()
            .sub(z.clone().matmul(y.clone()))
            .mul_scalar(0.5);
        y = y.matmul(t.clone());
        z = t.matmul(z);
    }

    let sqrt_norm = norm_a.sqrt().unsqueeze_dim::<2>(0).expand([d, d]);
    y.mul(sqrt_norm)
}

/// Frechet distance between two multivariate Gaussians.
///
/// Uses the symmetric form (S @ sigma2 @ S where S = sqrtm(sigma1)) so that
/// Newton-Schulz converges — the naive sqrtm(sigma1 @ sigma2) is non-symmetric.
fn frechet_distance(
    mu1: Tensor<1>,
    sigma1: Tensor<2>,
    mu2: Tensor<1>,
    sigma2: Tensor<2>,
    num_iterations: usize,
) -> Tensor<1> {
    let [d, _] = sigma1.dims();
    let device = sigma1.device();

    let diff = mu1.sub(mu2);
    let mean_term = diff.clone().mul(diff).sum();

    // Small regularization (eps · I) scaled to the average variance for numerical
    // stability with near-singular covariances. Done entirely with tensor ops to
    // avoid forcing a GPU sync.
    let tr_sum = linalg::trace::<2, 1>(sigma1.clone()).add(linalg::trace::<2, 1>(sigma2.clone()));
    let avg_variance = tr_sum.div_scalar(2.0 * d as f64).clamp_min(EPS);
    let reg = Tensor::<2>::eye(d, &device).mul(avg_variance.mul_scalar(EPS).unsqueeze_dim::<2>(0));
    let sigma1 = sigma1.add(reg.clone());
    let sigma2 = sigma2.add(reg);

    let sqrt_sigma1 = matrix_sqrt_newton_schulz(sigma1.clone(), num_iterations);
    let m = sqrt_sigma1
        .clone()
        .matmul(sigma2.clone())
        .matmul(sqrt_sigma1);
    let sqrt_m = matrix_sqrt_newton_schulz(m, num_iterations);

    let cov_term = sigma1.add(sigma2).sub(sqrt_m.mul_scalar(2.0));
    let trace = linalg::trace::<2, 1>(cov_term);

    mean_term.add(trace).reshape([1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_core::tensor::{TensorData, Tolerance};

    type FT = f32;

    #[test]
    fn test_newton_schulz_identity() {
        let device = Default::default();
        let identity = Tensor::<2>::eye(3, &device);
        let sqrt_i = matrix_sqrt_newton_schulz(identity.clone(), 50);

        sqrt_i
            .into_data()
            .assert_approx_eq::<FT>(&identity.into_data(), Tolerance::relative(1e-4));
    }

    #[test]
    fn test_newton_schulz_diagonal() {
        // sqrt(diag(4, 9, 16)) should be diag(2, 3, 4)
        let device = Default::default();
        let a = Tensor::<2>::from_floats(
            [[4.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 16.0]],
            &device,
        );
        let expected =
            Tensor::<2>::from_floats([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]], &device);

        let sqrt_a = matrix_sqrt_newton_schulz(a, 50);

        sqrt_a
            .into_data()
            .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::relative(1e-3));
    }

    #[test]
    fn test_compute_statistics() {
        // [[1,2],[3,4],[5,6]] -> mean [3,4], centered [[-2,-2],[0,0],[2,2]]
        // cov = [[8,8],[8,8]] / 2 = [[4,4],[4,4]]
        let device = Default::default();
        let features = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &device);

        let (mean, cov) = compute_statistics(features);

        mean.into_data()
            .assert_approx_eq::<FT>(&TensorData::from([3.0_f32, 4.0]), Tolerance::default());
        cov.into_data().assert_approx_eq::<FT>(
            &TensorData::from([[4.0_f32, 4.0], [4.0, 4.0]]),
            Tolerance::default(),
        );
    }

    #[test]
    fn test_fid_same_features_is_zero() {
        let device = Default::default();
        let features = Tensor::<2>::from_floats(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            &device,
        );

        let (mu, sigma) = compute_statistics(features.clone());
        let fid = frechet_distance(mu.clone(), sigma.clone(), mu, sigma, 50);

        assert!(fid.into_data().to_vec::<f32>().unwrap()[0].abs() < 0.1);
    }

    #[test]
    fn test_fid_shifted_mean() {
        // Shift one distribution by [2, 0, 0] — FID should be ~||shift||² = 4.0
        let device = Default::default();

        // Handcrafted feature rows with some spread
        let base = Tensor::<2>::from_floats(
            [
                [-0.3, 0.1, 0.4],
                [0.2, -0.5, 0.1],
                [0.5, 0.3, -0.2],
                [-0.1, 0.4, -0.3],
                [0.0, -0.2, 0.5],
                [0.3, 0.0, -0.4],
                [-0.4, 0.2, 0.3],
                [0.1, -0.1, 0.0],
                [0.4, 0.5, -0.1],
                [-0.2, -0.3, 0.2],
            ],
            &device,
        );

        // Same data but shifted by [2, 0, 0]
        let shift = Tensor::<2>::from_floats([[2.0, 0.0, 0.0]], &device).expand([10, 3]);
        let shifted = base.clone().add(shift);

        let (mu1, sigma1) = compute_statistics(base);
        let (mu2, sigma2) = compute_statistics(shifted);
        let fid_val = frechet_distance(mu1, sigma1, mu2, sigma2, 50)
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];

        assert!((fid_val - 4.0).abs() < 0.5);
    }

    #[test]
    fn test_fid_symmetry() {
        let device = Default::default();
        let features1 =
            Tensor::<2>::from_floats([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]], &device);
        let features2 =
            Tensor::<2>::from_floats([[2.0, 1.0], [1.0, 2.0], [2.0, 2.0], [1.5, 1.5]], &device);

        let (mu1, sigma1) = compute_statistics(features1.clone());
        let (mu2, sigma2) = compute_statistics(features2.clone());

        let fid_forward =
            frechet_distance(mu1.clone(), sigma1.clone(), mu2.clone(), sigma2.clone(), 50);
        let fid_reverse = frechet_distance(mu2, sigma2, mu1, sigma1, 50);

        fid_forward
            .into_data()
            .assert_approx_eq::<FT>(&fid_reverse.into_data(), Tolerance::relative(1e-3));
    }

    #[test]
    fn test_inception_output_shape() {
        let device = Default::default();
        let extractor = InceptionV3FeatureExtractor::new(&device);
        let input = Tensor::<4>::zeros([1, 3, 299, 299], &device);
        assert_eq!(extractor.forward(input).dims(), [1, 2048]);
    }

    #[test]
    fn test_fid_extract_features_shape() {
        let device = Default::default();
        let fid = FidConfig::new().init(&device);
        let images = Tensor::<4>::zeros([2, 3, 299, 299], &device);
        assert_eq!(fid.extract_features(images).dims(), [2, 2048]);
    }

    #[test]
    #[ignore = "downloads pre-trained weights"]
    fn test_fid_pretrained_features() {
        let device = Default::default();
        let fid = FidConfig::new().init_pretrained(&device);

        let images = Tensor::<4>::ones([1, 3, 299, 299], &device).mul_scalar(0.5);
        let features = fid.extract_features(images);

        assert_eq!(features.dims(), [1, 2048]);
        let feat_data = features.into_data().to_vec::<f32>().unwrap();
        assert!(feat_data.iter().all(|v| v.is_finite()));
        let norm: f32 = feat_data.iter().map(|v| v * v).sum();
        assert!(norm > 0.0);
    }
}
