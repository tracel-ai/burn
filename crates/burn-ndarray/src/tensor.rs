use burn_tensor::{
    quantization::{QTensorPrimitive, QuantizationScheme, QuantizationStrategy},
    Element, Shape, TensorData,
};

use ndarray::{ArcArray, Array, Dim, IxDyn};

use crate::element::QuantElement;

/// Tensor primitive used by the [ndarray backend](crate::NdArray).
#[derive(new, Debug, Clone)]
pub struct NdArrayTensor<E> {
    /// Dynamic array that contains the data of type E.
    pub array: ArcArray<E, IxDyn>,
}

impl<E> NdArrayTensor<E> {
    pub(crate) fn shape(&self) -> Shape {
        Shape::from(self.array.shape().to_vec())
    }
}

#[cfg(test)]
mod utils {
    use super::*;
    use crate::element::FloatNdArrayElement;

    impl<E> NdArrayTensor<E>
    where
        E: Default + Clone,
    {
        pub(crate) fn into_data(self) -> TensorData
        where
            E: FloatNdArrayElement,
        {
            let shape = self.shape();
            let values = self.array.into_iter().collect();

            TensorData::new(values, shape)
        }
    }
}

/// Converts a slice of usize to a typed dimension.
#[macro_export(local_inner_macros)]
macro_rules! to_typed_dims {
    (
        $n:expr,
        $dims:expr,
        justdim
    ) => {{
        let mut dims = [0; $n];
        for i in 0..$n {
            dims[i] = $dims[i];
        }
        let dim: Dim<[usize; $n]> = Dim(dims);
        dim
    }};
}

/// Reshapes an array into a tensor.
#[macro_export(local_inner_macros)]
macro_rules! reshape {
    (
        ty $ty:ty,
        n $n:expr,
        shape $shape:expr,
        array $array:expr
    ) => {{
        let dim = $crate::to_typed_dims!($n, $shape.dims, justdim);
        let array: ndarray::ArcArray<$ty, Dim<[usize; $n]>> = match $array.is_standard_layout() {
            true => $array
                .to_shape(dim)
                .expect("Safe to change shape without relayout")
                .into_shared(),
            false => $array.to_shape(dim).unwrap().as_standard_layout().into_shared(),
        };
        let array = array.into_dyn();

        NdArrayTensor::new(array)
    }};
    (
        ty $ty:ty,
        shape $shape:expr,
        array $array:expr,
        d $D:expr
    ) => {{
        match $D {
            1 => reshape!(ty $ty, n 1, shape $shape, array $array),
            2 => reshape!(ty $ty, n 2, shape $shape, array $array),
            3 => reshape!(ty $ty, n 3, shape $shape, array $array),
            4 => reshape!(ty $ty, n 4, shape $shape, array $array),
            5 => reshape!(ty $ty, n 5, shape $shape, array $array),
            6 => reshape!(ty $ty, n 6, shape $shape, array $array),
            _ => core::panic!("NdArray supports arrays up to 6 dimensions, received: {}", $D),
        }
    }};
}

impl<E> NdArrayTensor<E>
where
    E: Element,
{
    /// Create a new [ndarray tensor](NdArrayTensor) from [data](TensorData).
    pub fn from_data(data: TensorData) -> NdArrayTensor<E> {
        let shape: Shape = data.shape.clone().into();
        let to_array = |data: TensorData| Array::from_iter(data.iter()).into_shared();
        let array = to_array(data);
        let ndims = shape.num_dims();

        reshape!(
            ty E,
            shape shape,
            array array,
            d ndims
        )
    }
}

/// A quantized tensor for the ndarray backend.
#[derive(Clone, Debug)]
pub struct NdArrayQTensor<Q: QuantElement> {
    /// The quantized tensor.
    pub qtensor: NdArrayTensor<Q>,
    /// The quantization scheme.
    pub scheme: QuantizationScheme,
    /// The quantization strategy.
    pub strategy: QuantizationStrategy,
}

impl<Q: QuantElement> QTensorPrimitive for NdArrayQTensor<Q> {
    fn scheme(&self) -> &QuantizationScheme {
        &self.scheme
    }

    fn strategy(&self) -> QuantizationStrategy {
        self.strategy
    }
}

#[cfg(test)]
mod tests {
    use crate::NdArray;

    use super::*;
    use burn_common::rand::get_seeded_rng;
    use burn_tensor::{
        ops::QTensorOps,
        quantization::{AffineQuantization, QuantizationParametersPrimitive, QuantizationType},
        Distribution,
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
        let tensor = NdArrayTensor::<f32>::from_data(TensorData::from([-1.8, -1.0, 0.0, 0.5]));
        let scheme = QuantizationScheme::PerTensorAffine(QuantizationType::QInt8);
        let qparams = QuantizationParametersPrimitive {
            scale: NdArrayTensor::from_data(TensorData::from([0.009_019_608])),
            offset: Some(NdArrayTensor::from_data(TensorData::from([72]))),
        };
        let qtensor: NdArrayQTensor<i8> = NdArray::quantize(tensor, &scheme, qparams);

        assert_eq!(qtensor.scheme(), &scheme);
        assert_eq!(
            qtensor.strategy(),
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.009_019_608, 72))
        );
    }
}
