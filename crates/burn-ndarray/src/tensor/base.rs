use burn_tensor::{DType, Element, Shape, TensorData, TensorMetadata};
use core::mem;
use ndarray::{ArcArray, ArrayD, IxDyn};

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

impl<E> NdArrayTensor<E>
where
    E: Element,
{
    /// Create a new [ndarray tensor](NdArrayTensor) from [data](TensorData).
    pub fn from_data(mut data: TensorData) -> NdArrayTensor<E> {
        let shape = mem::take(&mut data.shape);

        let array = match data.into_vec::<E>() {
            // Safety: TensorData checks shape validity on creation, so we don't need to repeat that check here
            Ok(vec) => unsafe { ArrayD::from_shape_vec_unchecked(shape, vec) }.into_shared(),
            Err(err) => panic!("Data should have the same element type as the tensor {err:?}"),
        };

        NdArrayTensor::new(array)
    }
}

mod utils {
    use burn_common::tensor::is_contiguous;

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
                        if vec.len() > shape.num_elements() {
                            vec.drain(shape.num_elements()..vec.len());
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
            let mut strides = Vec::with_capacity(self.array.strides().len());

            for &stride in self.array.strides() {
                if stride <= 0 {
                    return false;
                }
                strides.push(stride as usize);
            }
            is_contiguous(shape, &strides)
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
            true => {
                match $array.to_shape(dim) {
                    Ok(val) => val.into_shared(),
                    Err(err) => {
                        core::panic!("Shape should be compatible shape={dim:?}: {err:?}");
                    }
                }
            },
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
