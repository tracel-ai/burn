use burn_core as burn;

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::module::avg_pool1d;
use burn::tensor::ops::PadMode;

/// Configuration to create a [LocalResponseNorm](LocalResponseNorm) layer
/// using the [init function](LocalResponseNormConfig::init).
#[derive(Config, Debug)]
pub struct LocalResponseNormConfig {
    /// Number of channels in the sliding normalization window.
    pub size: usize,
    /// Scaling parameter. Default: 0.0001
    #[config(default = 0.0001)]
    pub alpha: f64,
    /// Exponent. Default: 0.75
    #[config(default = 0.75)]
    pub beta: f64,
    /// Bias constant (called `bias` in ONNX). Default: 1.0
    #[config(default = 1.0)]
    pub k: f64,
}

impl LocalResponseNormConfig {
    /// Initialize a new [LocalResponseNorm](LocalResponseNorm) module.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.
    pub fn init(&self) -> LocalResponseNorm {
        assert!(self.size > 0, "size must be greater than 0.");

        LocalResponseNorm {
            size: self.size,
            alpha: self.alpha,
            beta: self.beta,
            k: self.k,
        }
    }
}

/// Applies Local Response Normalization as described in
/// [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html).
///
/// `Y = X / (k + (alpha / size) * sum(X^2))^beta`
///
/// Where the sum is computed over a sliding window of `size` channels.
///
/// For odd `size`, the window is centered on each channel position.
/// For even `size`, the window uses asymmetric padding and includes the current
/// channel plus one extra channel on the positive side.
///
/// Should be created using [LocalResponseNormConfig](LocalResponseNormConfig).
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct LocalResponseNorm {
    /// Number of channels in the sliding window.
    size: usize,
    /// Scaling parameter.
    alpha: f64,
    /// Exponent.
    beta: f64,
    /// Bias constant.
    k: f64,
}

impl LocalResponseNorm {
    /// Applies Local Response Normalization on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[N, C, D1, D2, ..., Dk]` (rank >= 3)
    /// - output: same shape as input
    ///
    /// # Panics
    ///
    /// Panics if the input tensor rank is less than 3.
    pub fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        assert!(
            D >= 3,
            "LocalResponseNorm requires input rank >= 3, got {D}"
        );

        let shape = input.dims();
        let n = shape[0];
        let c = shape[1];
        let d_flat: usize = shape[2..].iter().product();

        // Square the input
        let squared = input.clone().square();

        // Flatten spatial dims: [N, C, D1..Dk] -> [N, C, D_flat]
        let flat: Tensor<3> = squared.reshape([n, c, d_flat]);

        // Move channel to last dim: [N, D_flat, C]
        let transposed = flat.swap_dims(1, 2);

        // Batch all spatial positions: [N*D_flat, 1, C]
        let batched: Tensor<3> = transposed.reshape([n * d_flat, 1, c]);

        let pad_left = (self.size - 1) / 2;
        let pad_right = self.size / 2;
        let square_avg = if pad_left != pad_right {
            let padded = batched.pad((pad_left, pad_right, 0, 0), PadMode::Constant(0.0));
            avg_pool1d(padded, self.size, 1, 0, true, false)
        } else {
            avg_pool1d(batched, self.size, 1, pad_left, true, false)
        };

        // Restore shape: [N*D_flat, 1, C] -> [N, D_flat, C] -> [N, C, D_flat] -> original
        let unbatched: Tensor<3> = square_avg.reshape([n, d_flat, c]);
        let untransposed = unbatched.swap_dims(1, 2);
        let square_avg_restored: Tensor<D> = untransposed.reshape(shape);

        // Apply LRN formula: output = input / (k + alpha * avg(x^2))^beta
        input
            / square_avg_restored
                .mul_scalar(self.alpha)
                .add_scalar(self.k)
                .powf_scalar(self.beta)
    }
}

impl ModuleDisplay for LocalResponseNorm {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("size", &self.size)
            .add("alpha", &self.alpha)
            .add("beta", &self.beta)
            .add("k", &self.k)
            .optional()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use burn::tensor::TensorData;
    use burn::tensor::Tolerance;

    type FT = f32;

    // --- Correctness tests (values from PyTorch, torch.manual_seed(42)) ---

    #[test]
    fn forward_default_params() {
        // size=5, alpha=0.0001, beta=0.75, k=1.0, input [1,3,4,4]
        let device = Default::default();
        let module = LocalResponseNormConfig::new(5).init();
        let input = Tensor::<4>::from_data(
            TensorData::from([[
                [
                    [1.9269, 1.4873, 0.9007, -2.1055],
                    [0.6784, -1.2345, -0.0431, -1.6047],
                    [-0.7521, 1.6487, -0.3925, -1.4036],
                    [-0.7279, -0.5594, -0.7688, 0.7624],
                ],
                [
                    [1.6423, -0.1596, -0.4974, 0.4396],
                    [-0.7581, 1.0783, 0.8008, 1.6806],
                    [1.2791, 1.2964, 0.6105, 1.3347],
                    [-0.2316, 0.0418, -0.2516, 0.8599],
                ],
                [
                    [-1.3847, -0.8712, -0.2234, 1.7174],
                    [0.3189, -0.4245, 0.3057, -0.7746],
                    [-1.5576, 0.9956, -0.8798, -0.6011],
                    [-1.2742, 2.1228, -1.2347, -0.4879],
                ],
            ]]),
            &device,
        );

        let output = module.forward(input);

        let expected = TensorData::from([[
            [
                [1.9267, 1.4872, 0.9007, -2.1053],
                [0.6784, -1.2345, -0.0431, -1.6045],
                [-0.7521, 1.6486, -0.3925, -1.4035],
                [-0.7279, -0.5594, -0.7688, 0.7624],
            ],
            [
                [1.6421, -0.1596, -0.4974, 0.4395],
                [-0.7581, 1.0783, 0.8008, 1.6805],
                [1.2790, 1.2963, 0.6105, 1.3347],
                [-0.2316, 0.0418, -0.2516, 0.8598],
            ],
            [
                [-1.3845, -0.8712, -0.2234, 1.7172],
                [0.3189, -0.4245, 0.3057, -0.7745],
                [-1.5575, 0.9956, -0.8798, -0.6011],
                [-1.2741, 2.1226, -1.2346, -0.4879],
            ],
        ]]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::rel_abs(5e-3, 1e-4));
    }

    #[test]
    fn forward_custom_params() {
        // size=3, alpha=0.001, beta=0.5, k=2.0, input [1,4,3,3]
        let device = Default::default();
        let module = LocalResponseNormConfig::new(3)
            .with_alpha(0.001)
            .with_beta(0.5)
            .with_k(2.0)
            .init();
        let input = Tensor::<4>::from_data(
            TensorData::from([[
                [
                    [1.9269, 1.4873, 0.9007],
                    [-2.1055, 0.6784, -1.2345],
                    [-0.0431, -1.6047, -0.7521],
                ],
                [
                    [1.6487, -0.3925, -1.4036],
                    [-0.7279, -0.5594, -0.7688],
                    [0.7624, 1.6423, -0.1596],
                ],
                [
                    [-0.4974, 0.4396, 0.3189],
                    [-0.4245, 0.3057, -0.7746],
                    [0.0349, 0.3211, 1.5736],
                ],
                [
                    [-0.8455, -1.2742, 2.1228],
                    [-1.2347, -0.4879, -1.4181],
                    [0.8963, 0.0499, 2.2667],
                ],
            ]]),
            &device,
        );

        let output = module.forward(input);

        let expected = TensorData::from([[
            [
                [1.3618, 1.0515, 0.6368],
                [-1.4882, 0.4797, -0.8728],
                [-0.0305, -1.1342, -0.5318],
            ],
            [
                [1.1652, -0.2775, -0.9923],
                [-0.5145, -0.3955, -0.5435],
                [0.5391, 1.1608, -0.1128],
            ],
            [
                [-0.3516, 0.3108, 0.2254],
                [-0.3001, 0.2162, -0.5476],
                [0.0247, 0.2270, 1.1120],
            ],
            [
                [-0.5978, -0.9008, 1.5005],
                [-0.8729, -0.3450, -1.0025],
                [0.6337, 0.0353, 1.6018],
            ],
        ]]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::rel_abs(5e-3, 1e-4));
    }

    #[test]
    fn forward_even_size() {
        // size=2, alpha=0.0001, beta=0.75, k=1.0, input [1,3,2,2]
        let device = Default::default();
        let module = LocalResponseNormConfig::new(2).init();
        let input = Tensor::<4>::from_data(
            TensorData::from([[
                [[0.3367, 0.1288], [0.2345, 0.2303]],
                [[-1.1229, -0.1863], [2.2082, -0.6380]],
                [[0.4617, 0.2674], [0.5349, 0.8094]],
            ]]),
            &device,
        );

        let output = module.forward(input);

        let expected = TensorData::from([[
            [[0.3367, 0.1288], [0.2345, 0.2303]],
            [[-1.1228, -0.1863], [2.2078, -0.6380]],
            [[0.4616, 0.2673], [0.5348, 0.8093]],
        ]]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::rel_abs(5e-3, 1e-4));
    }

    #[test]
    fn forward_even_size_uses_asymmetric_positive_side_window() {
        let device = Default::default();
        let module = LocalResponseNormConfig::new(2)
            .with_alpha(1.0)
            .with_beta(1.0)
            .with_k(0.0)
            .init();
        let input = Tensor::<3>::from_data(TensorData::from([[[1.0], [2.0], [4.0]]]), &device);

        let output = module.forward(input);

        // For size=2, the implementation pads asymmetrically and uses:
        // c0 -> avg([1^2, 2^2]) = 2.5
        // c1 -> avg([2^2, 4^2]) = 10.0
        // c2 -> avg([4^2, 0]) = 8.0
        let expected = TensorData::from([[[0.4], [0.2], [0.5]]]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::rel_abs(1e-5, 1e-6));
    }

    #[test]
    fn forward_3d() {
        // size=3, input [1,4,6]
        let device = Default::default();
        let module = LocalResponseNormConfig::new(3).init();
        let input = Tensor::<3>::from_data(
            TensorData::from([[
                [1.9269, 1.4873, 0.9007, -2.1055, 0.6784, -1.2345],
                [-0.0431, -1.6047, 0.3559, -0.6866, -0.4934, 0.2415],
                [-1.1109, 0.0915, -2.3169, -0.2168, -0.3097, -0.3957],
                [0.8034, -0.6216, -0.5920, -0.0631, -0.8286, 0.3309],
            ]]),
            &device,
        );

        let output = module.forward(input);

        let expected = TensorData::from([[
            [1.9267, 1.4871, 0.9007, -2.1053, 0.6784, -1.2345],
            [-0.0431, -1.6045, 0.3558, -0.6865, -0.4933, 0.2415],
            [-1.1109, 0.0915, -2.3166, -0.2168, -0.3097, -0.3957],
            [0.8034, -0.6216, -0.5919, -0.0631, -0.8285, 0.3309],
        ]]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::rel_abs(5e-3, 1e-4));
    }

    #[test]
    fn forward_5d() {
        // size=3, input [1,3,2,2,2]
        let device = Default::default();
        let module = LocalResponseNormConfig::new(3).init();
        let input = Tensor::<5>::from_data(
            TensorData::from([[
                [
                    [[1.9269, 1.4873], [0.9007, -2.1055]],
                    [[0.6784, -1.2345], [-0.0431, -1.6047]],
                ],
                [
                    [[0.3559, -0.6866], [-0.4934, 0.2415]],
                    [[-1.1109, 0.0915], [-2.3169, -0.2168]],
                ],
                [
                    [[-0.3097, -0.3957], [0.8034, -0.6216]],
                    [[-0.5920, -0.0631], [-0.8286, 0.3309]],
                ],
            ]]),
            &device,
        );

        let output = module.forward(input);

        let expected = TensorData::from([[
            [
                [[1.9267, 1.4872], [0.9007, -2.1053]],
                [[0.6784, -1.2345], [-0.0431, -1.6046]],
            ],
            [
                [[0.3558, -0.6866], [-0.4933, 0.2415]],
                [[-1.1108, 0.0915], [-2.3166, -0.2168]],
            ],
            [
                [[-0.3097, -0.3957], [0.8034, -0.6216]],
                [[-0.5920, -0.0631], [-0.8284, 0.3309]],
            ],
        ]]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::rel_abs(5e-3, 1e-4));
    }

    // --- Edge case tests ---

    #[test]
    fn forward_size_1() {
        // size=1: window covers only self-channel, input [1,3,3,3]
        let device = Default::default();
        let module = LocalResponseNormConfig::new(1).init();
        let input = Tensor::<4>::from_data(
            TensorData::from([[
                [
                    [1.9269, 1.4873, 0.9007],
                    [-2.1055, 0.6784, -1.2345],
                    [-0.0431, -1.6047, -0.7521],
                ],
                [
                    [1.6487, -0.3925, 0.2415],
                    [-1.1109, 0.0915, -2.3169],
                    [-0.2168, -1.3847, -0.8712],
                ],
                [
                    [-0.2234, -0.6216, -0.5920],
                    [-0.0631, -0.8286, 0.3309],
                    [-1.5576, 0.9956, -0.8798],
                ],
            ]]),
            &device,
        );

        let output = module.forward(input);

        let expected = TensorData::from([[
            [
                [1.9264, 1.4870, 0.9007],
                [-2.1048, 0.6784, -1.2344],
                [-0.0431, -1.6044, -0.7521],
            ],
            [
                [1.6484, -0.3925, 0.2415],
                [-1.1108, 0.0915, -2.3160],
                [-0.2168, -1.3845, -0.8712],
            ],
            [
                [-0.2234, -0.6216, -0.5920],
                [-0.0631, -0.8285, 0.3309],
                [-1.5573, 0.9956, -0.8797],
            ],
        ]]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::rel_abs(5e-3, 1e-4));
    }

    #[test]
    fn forward_c_less_than_size() {
        // C=2 < size=5, input [1,2,3,3]
        let device = Default::default();
        let module = LocalResponseNormConfig::new(5).init();
        let input = Tensor::<4>::from_data(
            TensorData::from([[
                [
                    [1.9269, 1.4873, -0.4974],
                    [0.4396, -0.7581, 1.0783],
                    [0.8008, 1.6806, 0.3559],
                ],
                [
                    [-0.6866, 0.6105, 1.3347],
                    [-0.2316, 0.0418, -0.2516],
                    [0.8599, -0.3097, -0.3957],
                ],
            ]]),
            &device,
        );

        let output = module.forward(input);

        let expected = TensorData::from([[
            [
                [1.9268, 1.4872, -0.4974],
                [0.4396, -0.7581, 1.0783],
                [0.8008, 1.6805, 0.3559],
            ],
            [
                [-0.6866, 0.6104, 1.3347],
                [-0.2316, 0.0418, -0.2516],
                [0.8598, -0.3097, -0.3957],
            ],
        ]]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::rel_abs(5e-3, 1e-4));
    }

    #[test]
    fn forward_multi_batch() {
        // N=2, size=5, input [2,3,4,4]
        let device = Default::default();
        let module = LocalResponseNormConfig::new(5).init();
        let input = Tensor::<4>::from_data(
            TensorData::from([
                [
                    [
                        [1.9269, 1.4873, 0.9007, -2.1055],
                        [0.6784, -1.2345, -0.0431, -1.6047],
                        [-0.7521, 1.6487, -0.3925, -1.4036],
                        [-0.7279, -0.5594, -0.7688, 0.7624],
                    ],
                    [
                        [1.6423, -0.1596, -0.4974, 0.4396],
                        [-0.7581, 1.0783, 0.8008, 1.6806],
                        [1.2791, 1.2964, 0.6105, 1.3347],
                        [-0.2316, 0.0418, -0.2516, 0.8599],
                    ],
                    [
                        [-1.3847, -0.8712, -0.2234, 1.7174],
                        [0.3189, -0.4245, 0.3057, -0.7746],
                        [-1.5576, 0.9956, -0.8798, -0.6011],
                        [-1.2742, 2.1228, -1.2347, -0.4879],
                    ],
                ],
                [
                    [
                        [-0.9138, -0.6581, 0.0780, 0.5258],
                        [-0.4880, 1.1914, -0.8140, -0.7360],
                        [-1.4032, 0.0360, -0.0635, 0.6756],
                        [-0.0978, 1.8446, -1.1845, 1.3835],
                    ],
                    [
                        [1.4451, 0.8564, 2.2181, 0.5232],
                        [0.3466, -0.1973, -1.0546, 1.2780],
                        [-0.1722, 0.5238, 0.0566, 0.4263],
                        [0.5750, -0.6417, -2.2064, -0.7508],
                    ],
                    [
                        [0.0109, -0.3387, -1.3407, -0.5854],
                        [0.5362, 0.5246, 1.1412, 0.0516],
                        [0.7440, -0.4816, -1.0495, 0.6039],
                        [-1.7223, -0.8278, 1.3347, 0.4835],
                    ],
                ],
            ]),
            &device,
        );

        let output = module.forward(input);

        let out_data = output.to_data();
        assert_eq!(out_data.shape, [2, 3, 4, 4].into());
        let expected_full = TensorData::from([
            [
                [
                    [1.9267, 1.4872, 0.9007, -2.1053],
                    [0.6784, -1.2345, -0.0431, -1.6045],
                    [-0.7521, 1.6486, -0.3925, -1.4035],
                    [-0.7279, -0.5594, -0.7688, 0.7624],
                ],
                [
                    [1.6421, -0.1596, -0.4974, 0.4395],
                    [-0.7581, 1.0783, 0.8008, 1.6805],
                    [1.2790, 1.2963, 0.6105, 1.3347],
                    [-0.2316, 0.0418, -0.2516, 0.8598],
                ],
                [
                    [-1.3845, -0.8712, -0.2234, 1.7172],
                    [0.3189, -0.4245, 0.3057, -0.7745],
                    [-1.5575, 0.9956, -0.8798, -0.6011],
                    [-1.2741, 2.1226, -1.2346, -0.4879],
                ],
            ],
            [
                [
                    [-0.9138, -0.6581, 0.0780, 0.5258],
                    [-0.4880, 1.1913, -0.8140, -0.7360],
                    [-1.4032, 0.0360, -0.0635, 0.6756],
                    [-0.0978, 1.8445, -1.1844, 1.3835],
                ],
                [
                    [1.4451, 0.8564, 2.2179, 0.5232],
                    [0.3466, -0.1973, -1.0545, 1.2780],
                    [-0.1722, 0.5238, 0.0566, 0.4263],
                    [0.5750, -0.6417, -2.2061, -0.7508],
                ],
                [
                    [0.0109, -0.3387, -1.3405, -0.5854],
                    [0.5362, 0.5246, 1.1411, 0.0516],
                    [0.7439, -0.4816, -1.0494, 0.6039],
                    [-1.7222, -0.8277, 1.3345, 0.4835],
                ],
            ],
        ]);
        out_data.assert_approx_eq::<FT>(&expected_full, Tolerance::rel_abs(5e-3, 1e-4));
    }

    // --- Validation / panic tests ---

    #[test]
    #[should_panic(expected = "size must be greater than 0")]
    fn config_size_zero_panics() {
        LocalResponseNormConfig::new(0).init();
    }

    #[test]
    #[should_panic(expected = "LocalResponseNorm requires input rank >= 3")]
    fn forward_rank_2_panics() {
        let module = LocalResponseNormConfig::new(3).init();
        let input = Tensor::<2>::zeros([2, 4], &Default::default());
        module.forward(input);
    }

    // --- Autodiff ---

    #[cfg(feature = "std")]
    #[test]
    fn backward() {
        use burn_core::tensor::Device;

        let device = Device::default().autodiff();
        let module = LocalResponseNormConfig::new(5).init();
        let input = Tensor::<4>::from_data(
            TensorData::from([[
                [
                    [1.9269, 1.4873, 0.9007, -2.1055],
                    [0.6784, -1.2345, -0.0431, -1.6047],
                    [-0.7521, 1.6487, -0.3925, -1.4036],
                    [-0.7279, -0.5594, -0.7688, 0.7624],
                ],
                [
                    [1.6423, -0.1596, -0.4974, 0.4396],
                    [-0.7581, 1.0783, 0.8008, 1.6806],
                    [1.2791, 1.2964, 0.6105, 1.3347],
                    [-0.2316, 0.0418, -0.2516, 0.8599],
                ],
                [
                    [-1.3847, -0.8712, -0.2234, 1.7174],
                    [0.3189, -0.4245, 0.3057, -0.7746],
                    [-1.5576, 0.9956, -0.8798, -0.6011],
                    [-1.2742, 2.1228, -1.2347, -0.4879],
                ],
            ]]),
            &device,
        )
        .require_grad();

        let output = module.forward(input.clone());
        let grads = output.sum().backward();
        let input_grad = input.grad(&grads).unwrap();

        assert_eq!(input_grad.dims(), [1, 3, 4, 4]);

        let expected_grad = TensorData::from([[
            [
                [0.9997, 0.9999, 1.0000, 0.9999],
                [1.0000, 0.9999, 1.0000, 0.9999],
                [0.9999, 0.9997, 1.0000, 0.9999],
                [0.9999, 1.0000, 0.9999, 1.0000],
            ],
            [
                [0.9998, 1.0000, 1.0000, 0.9999],
                [1.0000, 1.0000, 1.0000, 0.9999],
                [1.0000, 0.9998, 1.0000, 1.0000],
                [1.0000, 0.9999, 1.0000, 0.9999],
            ],
            [
                [1.0000, 1.0000, 1.0000, 0.9999],
                [1.0000, 1.0000, 1.0000, 0.9999],
                [0.9999, 0.9998, 1.0000, 0.9999],
                [0.9999, 0.9998, 0.9999, 1.0000],
            ],
        ]]);
        input_grad
            .to_data()
            .assert_approx_eq::<FT>(&expected_grad, Tolerance::rel_abs(5e-3, 1e-4));
    }

    // --- Display ---

    #[test]
    fn display() {
        let config = LocalResponseNormConfig::new(5);
        let module = config.init();
        assert_eq!(
            format!("{module}"),
            "LocalResponseNorm {size: 5, alpha: 0.0001, beta: 0.75, k: 1}"
        );
    }
}
