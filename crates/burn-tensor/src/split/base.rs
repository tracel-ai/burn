use std::marker::PhantomData;

use burn_backend::{Backend, BackendTypes, DTypeUsageSet, TensorMetadata};

use crate::ops::CompoundTensorKind;

#[derive(Debug)]
pub struct SplitTensor<const D: usize, K>
where
    K: CompoundTensorKind,
{
    pub(crate) _kind: PhantomData<K>,
    pub(crate) components: K::ComponentsArray,
}

impl<const D: usize, K> Clone for SplitTensor<D, K>
where
    K: CompoundTensorKind,
    K::ComponentsArray: Clone,
{
    fn clone(&self) -> Self {
        Self {
            _kind: PhantomData,
            components: self.components.clone(),
        }
    }
}

#[derive(Debug, Clone)]
/// A newtype that wraps a real backend B and exposes a split-layout backend.
pub struct SplitBackend<B: Backend>(core::marker::PhantomData<B>);

impl<B: Backend> BackendTypes for SplitBackend<B> {
    type Device = B::Device;

    type FloatTensorPrimitive = B::FloatTensorPrimitive;

    type IntTensorPrimitive = B::IntTensorPrimitive;

    type BoolTensorPrimitive = B::BoolTensorPrimitive;

    type QuantizedTensorPrimitive = B::QuantizedTensorPrimitive;

    fn dtype_usage(device: &Self::Device, dtype: burn_std::DType) -> DTypeUsageSet {
        B::dtype_usage(device, dtype)
    }

    fn device_count(type_id: u16) -> usize {
        B::device_count(type_id)
    }

    type ComplexTensorPrimitive = SplitPrimitive<B::FloatTensorPrimitive, 2>;
}

// Needs to be public to avoid a compile time error related to the visibility of the associated type for the tensor primitive in BackendTypes
#[derive(Debug, Clone)]
pub struct SplitPrimitive<T, const N: usize>(pub(super) [T; N]);

pub(crate) trait IsNotEmpty {
    const VALID: ();
}

// Implement it for all N. If N == 0, it will fail to compile.
impl<const N: usize> IsNotEmpty for [(); N] {
    const VALID: () = {
        if N == 0 {
            panic!("SplitPrimitive cannot be empty! N must be greater than 0.");
        }
    };
}

impl<T, const N: usize> TensorMetadata for SplitPrimitive<T, N>
where
    T: TensorMetadata + Send + Sync + 'static,
    [(); N]: IsNotEmpty, // <--- This bound forces the evaluation
{
    fn shape(&self) -> burn_std::Shape {
        self.0[0].shape()
    }

    fn rank(&self) -> usize {
        self.shape().num_dims()
    }

    fn dtype(&self) -> burn_std::DType {
        self.0[0].dtype()
    }
}
