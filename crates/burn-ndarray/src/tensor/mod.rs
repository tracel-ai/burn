mod base;
mod float;
mod int;
mod quant;

pub use base::*;
pub use float::*;
pub use int::*;
pub use quant::*;

#[cfg(test)]
mod tests {
    use crate::NdArray;

    use super::*;
    use burn_common::rand::get_seeded_rng;
    use burn_tensor::{
        Distribution, Shape, TensorData,
        ops::{FloatTensorOps, QTensorOps},
        quantization::QuantizationParametersPrimitive,
    };

    #[test]
    fn should_support_into_and_from_data_1d() {
        let data_expected = TensorData::random::<f32, _, _>(
            Shape::new([3]),
            Distribution::Default,
            &mut get_seeded_rng(),
        );
        let tensor = NdArrayTensor::<f32>::from_data(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_into_and_from_data_2d() {
        let data_expected = TensorData::random::<f32, _, _>(
            Shape::new([2, 3]),
            Distribution::Default,
            &mut get_seeded_rng(),
        );
        let tensor = NdArrayTensor::<f32>::from_data(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_into_and_from_data_3d() {
        let data_expected = TensorData::random::<f32, _, _>(
            Shape::new([2, 3, 4]),
            Distribution::Default,
            &mut get_seeded_rng(),
        );
        let tensor = NdArrayTensor::<f32>::from_data(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_into_and_from_data_4d() {
        let data_expected = TensorData::random::<f32, _, _>(
            Shape::new([2, 3, 4, 2]),
            Distribution::Default,
            &mut get_seeded_rng(),
        );
        let tensor = NdArrayTensor::<f32>::from_data(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_qtensor_strategy() {
        type B = NdArray<f32, i64, i8>;
        let scale: f32 = 0.009_019_608;
        let device = Default::default();

        let tensor = B::float_from_data(TensorData::from([-1.8f32, -1.0, 0.0, 0.5]), &device);
        let scheme = QuantScheme::default();
        let qparams = QuantizationParametersPrimitive {
            scale: B::float_from_data(TensorData::from([scale]), &device),
            offset: None,
        };
        let qtensor: NdArrayQTensor<i8> = B::quantize(tensor, &scheme, qparams);

        assert_eq!(qtensor.scheme(), &scheme);
        assert_eq!(
            qtensor.strategy(),
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(scale))
        );
    }
}
