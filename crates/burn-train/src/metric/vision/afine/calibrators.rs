//! Calibrators, adapter, and final-score mapping for A-FINE.
//!
//! Four small modules sit between the heads and the final per-sample
//! quality score:
//!
//! - [`NrCalibrator`] — logistic mapping of the naturalness head's raw
//!   output into `(-2, 2)`. Two learnable scalars.
//! - [`FrCalibratorWithLimit`] — logistic mapping of the fidelity head's
//!   raw output into `(-2, 2)`, with `yita3` clamped to `[0.05, 0.95]`
//!   and `yita4` to `[0.01, 0.70]` on every forward.
//! - [`AfineAdapter`] — `D = exp(softplus(k) * (N_ref - N_dis)) * N_dis + F`.
//!   Single learnable scalar `k`.
//! - [`scale_finalscore`] — fixed logistic into `(0, 100)` with the
//!   paper-reported constants.
//!
//! All three calibrators implement the same logistic shape:
//! `out = (yita1 - yita2) * sigmoid((x - yita3) / (|yita4| + eps)) + yita2`.
//! This is the algebraic equivalent of PyIQA's two-branch
//! `if exp_pow >= 10` formulation, rewritten as a single expression so
//! it batches correctly. PyIQA's branch only works on 0-D scalar
//! tensors.

use burn_core as burn;

use burn::config::Config;
use burn::module::{Module, Param};
use burn::tensor::Tensor;
use burn::tensor::activation::{sigmoid, softplus};
use burn::tensor::backend::Backend;

const NR_YITA1: f64 = 2.0;
const NR_YITA2: f64 = -2.0;
const NR_YITA3_INIT: f32 = 4.9592;
const NR_YITA4_INIT: f32 = 21.5968;

const FR_YITA1: f64 = 2.0;
const FR_YITA2: f64 = -2.0;
const FR_YITA3_INIT: f32 = 0.5;
const FR_YITA4_INIT: f32 = 0.15;
const FR_YITA3_MIN: f32 = 0.05;
const FR_YITA3_MAX: f32 = 0.95;
const FR_YITA4_MIN: f32 = 0.01;
const FR_YITA4_MAX: f32 = 0.70;

const ADAPTER_K_INIT: f32 = 5.0;

const SCALE_YITA1: f64 = 100.0;
const SCALE_YITA2: f64 = 0.0;
const SCALE_YITA3: f64 = -1.971_0;
const SCALE_YITA4: f64 = -2.373_4;

/// Numerical-stability epsilon in the logistic denominator. Matches
/// PyIQA exactly; do not change without coordinating a parity-test
/// re-capture.
const EPS: f64 = 1e-10;

/// Apply `(yita1 - yita2) * sigmoid((x - yita3) / (|yita4| + eps)) + yita2`
/// element-wise, broadcasting the 1-D scalar parameters over the input.
fn logistic_calibrate<B: Backend>(
    x: Tensor<B, 2>,
    yita3: Tensor<B, 1>,
    yita4_abs: Tensor<B, 1>,
    yita1: f64,
    yita2: f64,
) -> Tensor<B, 2> {
    let yita3 = yita3.reshape([1, 1]);
    let denom = yita4_abs.reshape([1, 1]).add_scalar(EPS);
    let inner = (x - yita3) / denom;
    sigmoid(inner).mul_scalar(yita1 - yita2).add_scalar(yita2)
}

/// Configuration for [`NrCalibrator`].
#[derive(Config, Debug)]
pub(crate) struct NrCalibratorConfig {}

impl NrCalibratorConfig {
    pub(crate) fn init<B: Backend>(&self, device: &B::Device) -> NrCalibrator<B> {
        NrCalibrator {
            yita3: Param::from_tensor(Tensor::from_floats([NR_YITA3_INIT], device)),
            yita4: Param::from_tensor(Tensor::from_floats([NR_YITA4_INIT], device)),
        }
    }
}

/// Naturalness logistic calibrator. Maps `[B, 1]` into `(-2, 2)`.
#[derive(Module, Debug)]
pub(crate) struct NrCalibrator<B: Backend> {
    pub(crate) yita3: Param<Tensor<B, 1>>,
    pub(crate) yita4: Param<Tensor<B, 1>>,
}

impl<B: Backend> NrCalibrator<B> {
    pub(crate) fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        logistic_calibrate(
            x,
            self.yita3.val(),
            self.yita4.val().abs(),
            NR_YITA1,
            NR_YITA2,
        )
    }
}

/// Configuration for [`FrCalibratorWithLimit`].
#[derive(Config, Debug)]
pub(crate) struct FrCalibratorWithLimitConfig {}

impl FrCalibratorWithLimitConfig {
    pub(crate) fn init<B: Backend>(&self, device: &B::Device) -> FrCalibratorWithLimit<B> {
        FrCalibratorWithLimit {
            yita3: Param::from_tensor(Tensor::from_floats([FR_YITA3_INIT], device)),
            yita4: Param::from_tensor(Tensor::from_floats([FR_YITA4_INIT], device)),
        }
    }
}

/// Fidelity logistic calibrator with on-forward clamping of `yita3` and
/// `yita4`. PyIQA clamps the values used in the formula on every call;
/// the stored parameter is unchanged.
#[derive(Module, Debug)]
pub(crate) struct FrCalibratorWithLimit<B: Backend> {
    pub(crate) yita3: Param<Tensor<B, 1>>,
    pub(crate) yita4: Param<Tensor<B, 1>>,
}

impl<B: Backend> FrCalibratorWithLimit<B> {
    pub(crate) fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // Match PyIQA semantics exactly: clamp first, then abs. The
        // clamp range is positive so the abs is a no-op for in-range
        // values, but for an out-of-range checkpoint or a parameter
        // that drifts negative during training the order matters.
        let yita3 = self.yita3.val().clamp(FR_YITA3_MIN, FR_YITA3_MAX);
        let yita4 = self.yita4.val().clamp(FR_YITA4_MIN, FR_YITA4_MAX);
        logistic_calibrate(x, yita3, yita4.abs(), FR_YITA1, FR_YITA2)
    }
}

/// Configuration for [`AfineAdapter`].
#[derive(Config, Debug)]
pub(crate) struct AfineAdapterConfig {}

impl AfineAdapterConfig {
    pub(crate) fn init<B: Backend>(&self, device: &B::Device) -> AfineAdapter<B> {
        AfineAdapter {
            k: Param::from_tensor(Tensor::from_floats([ADAPTER_K_INIT], device)),
        }
    }
}

/// Fuses the calibrated naturalness and fidelity scores into a single
/// raw `D` value.
///
/// `D = exp(softplus(k) * (N_ref - N_dis)) * N_dis + F`. The `softplus`
/// wrapper enforces `k > 0` without constraining the stored parameter.
#[derive(Module, Debug)]
pub(crate) struct AfineAdapter<B: Backend> {
    pub(crate) k: Param<Tensor<B, 1>>,
}

impl<B: Backend> AfineAdapter<B> {
    pub(crate) fn forward(
        &self,
        x_nr: Tensor<B, 2>,
        ref_nr: Tensor<B, 2>,
        xref_fr: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let k_pos = softplus(self.k.val(), 1.0).reshape([1, 1]);
        let weight = (k_pos * (ref_nr - x_nr.clone())).exp();
        weight * x_nr + xref_fr
    }
}

/// Map a raw adapter score into `(0, 100)` via a fixed 4-parameter
/// logistic. Constants are the paper-reported defaults.
pub(crate) fn scale_finalscore<B: Backend>(score: Tensor<B, 2>) -> Tensor<B, 2> {
    let denom = SCALE_YITA4.abs() + EPS;
    let inner = score.sub_scalar(SCALE_YITA3).div_scalar(denom);
    sigmoid(inner)
        .mul_scalar(SCALE_YITA1 - SCALE_YITA2)
        .add_scalar(SCALE_YITA2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_flex::Flex;

    type TestBackend = Flex;

    #[test]
    fn nr_calibrator_maps_to_bounded_range() {
        let device = Default::default();
        let calibrator = NrCalibratorConfig::new().init::<TestBackend>(&device);

        let extremes = Tensor::<TestBackend, 2>::from_floats([[-1000.0], [0.0], [1000.0]], &device);
        let out = calibrator.forward(extremes);
        let values = out.into_data().to_vec::<f32>().unwrap();

        for v in &values {
            assert!(*v >= -2.0 && *v <= 2.0, "out-of-range value: {v}");
        }
        // Monotonic increasing.
        assert!(values[0] < values[1]);
        assert!(values[1] < values[2]);
    }

    #[test]
    fn fr_calibrator_clamp_does_not_panic() {
        let device = Default::default();
        let calibrator = FrCalibratorWithLimitConfig::new().init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 2>::from_floats([[0.5], [1.5], [-0.5]], &device);
        let out = calibrator.forward(input);

        assert_eq!(out.dims(), [3, 1]);
    }

    #[test]
    fn adapter_forward_propagates_shape() {
        let device = Default::default();
        let adapter = AfineAdapterConfig::new().init::<TestBackend>(&device);

        let nr_dis = Tensor::<TestBackend, 2>::from_floats([[0.5], [-0.3]], &device);
        let nr_ref = Tensor::<TestBackend, 2>::from_floats([[0.7], [-0.1]], &device);
        let fr = Tensor::<TestBackend, 2>::from_floats([[0.2], [0.4]], &device);

        let out = adapter.forward(nr_dis, nr_ref, fr);
        assert_eq!(out.dims(), [2, 1]);
    }

    #[test]
    fn scale_finalscore_maps_to_0_100_range() {
        let device = Default::default();

        let scores =
            Tensor::<TestBackend, 2>::from_floats([[-1000.0], [-1.971], [1000.0]], &device);
        let out = scale_finalscore(scores);
        let values = out.into_data().to_vec::<f32>().unwrap();

        assert!(values[0] >= 0.0 && values[0] <= 100.0);
        assert!(values[2] >= 0.0 && values[2] <= 100.0);
        // At yita3 = -1.971 the sigmoid argument is 0, so the output is
        // 100 * 0.5 = 50.
        assert!(
            (values[1] - 50.0).abs() < 0.5,
            "midpoint should be ~50, got {}",
            values[1]
        );
    }
}
