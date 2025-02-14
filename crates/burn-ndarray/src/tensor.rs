use burn_tensor::{
    quantization::{
        AffineQuantization, QParams, QTensorPrimitive, QuantizationMode, QuantizationScheme,
        QuantizationStrategy, QuantizationType, SymmetricQuantization,
    },
    DType, Element, Shape, TensorData, TensorMetadata,
};

use ndarray::{ArcArray, Array, Dim, IxDyn};

use crate::element::QuantElement;

/// Tensor primitive used by the [ndarray backend](crate::NdArray).
#[derive(new, Debug, Clone)]
pub struct NdArrayTensor<E> {
    /// Dynamic array that contains the data of type E.
    pub array: ArcArray<E, IxDyn>,
}

impl<E: Element> TensorMetadata for NdArrayTensor<E> {
    fn dtype(&self) -> DType {
        E::dtype()
    }

    fn shape(&self) -> Shape {
        Shape::from(self.array.shape().to_vec())
    }
}

/// Float tensor primitive.
#[derive(Debug, Clone)]
pub enum NdArrayTensorFloat {
    /// 32-bit float.
    F32(NdArrayTensor<f32>),
    /// 64-bit float.
    F64(NdArrayTensor<f64>),
}

impl From<NdArrayTensor<f32>> for NdArrayTensorFloat {
    fn from(value: NdArrayTensor<f32>) -> Self {
        NdArrayTensorFloat::F32(value)
    }
}

impl From<NdArrayTensor<f64>> for NdArrayTensorFloat {
    fn from(value: NdArrayTensor<f64>) -> Self {
        NdArrayTensorFloat::F64(value)
    }
}

impl TensorMetadata for NdArrayTensorFloat {
    fn dtype(&self) -> DType {
        match self {
            NdArrayTensorFloat::F32(tensor) => tensor.dtype(),
            NdArrayTensorFloat::F64(tensor) => tensor.dtype(),
        }
    }

    fn shape(&self) -> Shape {
        match self {
            NdArrayTensorFloat::F32(tensor) => tensor.shape(),
            NdArrayTensorFloat::F64(tensor) => tensor.shape(),
        }
    }
}

/// Macro to create a new [float tensor](NdArrayTensorFloat) based on the element type.
#[macro_export]
macro_rules! new_tensor_float {
    // Op executed with default dtype
    ($tensor:expr) => {{
        match E::dtype() {
            burn_tensor::DType::F64 => $crate::NdArrayTensorFloat::F64($tensor),
            burn_tensor::DType::F32 => $crate::NdArrayTensorFloat::F32($tensor),
            // FloatNdArrayElement only implemented for f64 and f32
            _ => unimplemented!("Unsupported dtype"),
        }
    }};
}

/// Macro to execute an operation a given element type.
///
/// # Panics
/// Since there is no automatic type cast at this time, binary operations for different
/// floating point precision data types will panic with a data type mismatch.
#[macro_export]
macro_rules! execute_with_float_dtype {
    // Binary op: type automatically inferred by the compiler
    (($lhs:expr, $rhs:expr), $op:expr) => {{
        let lhs_dtype = burn_tensor::TensorMetadata::dtype(&$lhs);
        let rhs_dtype = burn_tensor::TensorMetadata::dtype(&$rhs);
        match ($lhs, $rhs) {
            ($crate::NdArrayTensorFloat::F64(lhs), $crate::NdArrayTensorFloat::F64(rhs)) => {
                $crate::NdArrayTensorFloat::F64($op(lhs, rhs))
            }
            ($crate::NdArrayTensorFloat::F32(lhs), $crate::NdArrayTensorFloat::F32(rhs)) => {
                $crate::NdArrayTensorFloat::F32($op(lhs, rhs))
            }
            _ => panic!(
                "Data type mismatch (lhs: {:?}, rhs: {:?})",
                lhs_dtype, rhs_dtype
            ),
        }
    }};

    // Binary op: generic type cannot be inferred for an operation
    (($lhs:expr, $rhs:expr), $element:ident, $op:expr) => {{
        let lhs_dtype = burn_tensor::TensorMetadata::dtype(&$lhs);
        let rhs_dtype = burn_tensor::TensorMetadata::dtype(&$rhs);
        match ($lhs, $rhs) {
            ($crate::NdArrayTensorFloat::F64(lhs), $crate::NdArrayTensorFloat::F64(rhs)) => {
                type $element = f64;
                $crate::NdArrayTensorFloat::F64($op(lhs, rhs))
            }
            ($crate::NdArrayTensorFloat::F32(lhs), $crate::NdArrayTensorFloat::F32(rhs)) => {
                type $element = f32;
                $crate::NdArrayTensorFloat::F32($op(lhs, rhs))
            }
            _ => panic!(
                "Data type mismatch (lhs: {:?}, rhs: {:?})",
                lhs_dtype, rhs_dtype
            ),
        }
    }};

    // Binary op: type automatically inferred by the compiler but return type is not a float tensor
    (($lhs:expr, $rhs:expr) => $op:expr) => {{
        let lhs_dtype = burn_tensor::TensorMetadata::dtype(&$lhs);
        let rhs_dtype = burn_tensor::TensorMetadata::dtype(&$rhs);
        match ($lhs, $rhs) {
            ($crate::NdArrayTensorFloat::F64(lhs), $crate::NdArrayTensorFloat::F64(rhs)) => {
                $op(lhs, rhs)
            }
            ($crate::NdArrayTensorFloat::F32(lhs), $crate::NdArrayTensorFloat::F32(rhs)) => {
                $op(lhs, rhs)
            }
            _ => panic!(
                "Data type mismatch (lhs: {:?}, rhs: {:?})",
                lhs_dtype, rhs_dtype
            ),
        }
    }};

    // Unary op: type automatically inferred by the compiler
    ($tensor:expr, $op:expr) => {{
        match $tensor {
            $crate::NdArrayTensorFloat::F64(tensor) => $crate::NdArrayTensorFloat::F64($op(tensor)),
            $crate::NdArrayTensorFloat::F32(tensor) => $crate::NdArrayTensorFloat::F32($op(tensor)),
        }
    }};

    // Unary op: generic type cannot be inferred for an operation
    ($tensor:expr, $element:ident, $op:expr) => {{
        match $tensor {
            $crate::NdArrayTensorFloat::F64(tensor) => {
                type $element = f64;
                $crate::NdArrayTensorFloat::F64($op(tensor))
            }
            $crate::NdArrayTensorFloat::F32(tensor) => {
                type $element = f32;
                $crate::NdArrayTensorFloat::F32($op(tensor))
            }
        }
    }};

    // Unary op: type automatically inferred by the compiler but return type is not a float tensor
    ($tensor:expr => $op:expr) => {{
        match $tensor {
            $crate::NdArrayTensorFloat::F64(tensor) => $op(tensor),
            $crate::NdArrayTensorFloat::F32(tensor) => $op(tensor),
        }
    }};

    // Unary op: generic type cannot be inferred for an operation and return type is not a float tensor
    ($tensor:expr, $element:ident => $op:expr) => {{
        match $tensor {
            $crate::NdArrayTensorFloat::F64(tensor) => {
                type $element = f64;
                $op(tensor)
            }
            $crate::NdArrayTensorFloat::F32(tensor) => {
                type $element = f32;
                $op(tensor)
            }
        }
    }};
}

mod utils {
    use super::*;

    impl<E> NdArrayTensor<E>
    where
        E: Element,
    {
        pub(crate) fn into_data(self) -> TensorData {
            let shape = self.shape();

            let vec = if self.is_contiguous() {
                match self.array.try_into_owned_nocopy() {
                    Ok(owned) => {
                        let (mut vec, offset) = owned.into_raw_vec_and_offset();
                        if let Some(offset) = offset {
                            vec.drain(..offset);
                        }
                        vec
                    }
                    Err(array) => array.into_iter().collect(),
                }
            } else {
                self.array.into_iter().collect()
            };

            TensorData::new(vec, shape)
        }

        pub(crate) fn is_contiguous(&self) -> bool {
            let shape = self.array.shape();
            let strides = self.array.strides();

            if shape.is_empty() {
                return true;
            }

            if shape.len() == 1 {
                return strides[0] == 1;
            }

            let mut prev_stride = 1;
            let mut current_num_elems_shape = 1;

            for (i, (stride, shape)) in strides.iter().zip(shape).rev().enumerate() {
                let stride = if *stride <= 0 {
                    return false;
                } else {
                    *stride as usize
                };
                if i > 0 {
                    if current_num_elems_shape != stride {
                        return false;
                    }

                    if prev_stride >= stride {
                        return false;
                    }
                }

                current_num_elems_shape *= shape;
                prev_stride = stride;
            }

            true
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

        let array = match data.into_vec::<E>() {
            Ok(vec) => Array::from_vec(vec).into_shared(),
            Err(err) => panic!("Data should have the same element type as the tensor {err:?}"),
        };
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
    /// The quantization parameters.
    pub qparams: QParams<f32, Q>,
}

impl<Q: QuantElement> NdArrayQTensor<Q> {
    /// Returns the quantization strategy, including quantization parameters, for the given tensor.
    pub fn strategy(&self) -> QuantizationStrategy {
        match self.scheme {
            QuantizationScheme::PerTensor(QuantizationMode::Affine, QuantizationType::QInt8) => {
                QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(
                    self.qparams.scale,
                    self.qparams.offset.unwrap().elem(),
                ))
            }
            QuantizationScheme::PerTensor(QuantizationMode::Symmetric, QuantizationType::QInt8) => {
                QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(
                    self.qparams.scale,
                ))
            }
            QuantizationScheme::PerBlock(_mode, _dtype, _block_layout) => todo!(),
        }
    }
}

impl<Q: QuantElement> QTensorPrimitive for NdArrayQTensor<Q> {
    fn scheme(&self) -> &QuantizationScheme {
        &self.scheme
    }
}

impl<Q: QuantElement> TensorMetadata for NdArrayQTensor<Q> {
    fn dtype(&self) -> DType {
        DType::QFloat(self.scheme)
    }

    fn shape(&self) -> Shape {
        self.qtensor.shape()
    }
}

#[cfg(test)]
mod tests {
    use crate::NdArray;

    use super::*;
    use burn_common::rand::get_seeded_rng;
    use burn_tensor::{
        ops::{FloatTensorOps, IntTensorOps, QTensorOps},
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
        type B = NdArray<f32, i64, i8>;
        let scale: f32 = 0.009_019_608;
        let offset: i8 = 72;
        let device = Default::default();

        let tensor = B::float_from_data(TensorData::from([-1.8f32, -1.0, 0.0, 0.5]), &device);
        let scheme =
            QuantizationScheme::PerTensor(QuantizationMode::Affine, QuantizationType::QInt8);
        let qparams = QuantizationParametersPrimitive {
            scale: B::float_from_data(TensorData::from([scale]), &device),
            offset: Some(B::int_from_data(TensorData::from([offset as i64]), &device)),
        };
        let qtensor: NdArrayQTensor<i8> = B::quantize(tensor, &scheme, qparams);

        assert_eq!(qtensor.scheme(), &scheme);
        assert_eq!(
            qtensor.strategy(),
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(scale, offset))
        );
    }
}
