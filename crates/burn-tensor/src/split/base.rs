use core::marker::PhantomData;

use burn_backend::{BackendTypes, TensorMetadata};
use burn_dispatch::{Dispatch, DispatchTensor};

use crate::{
    Complex,
    ops::{BridgeTensor, CompoundTensorKind},
};

/// A tensor type that represents compound elements as separate component tensors.
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
pub struct SplitBackend;

impl BackendTypes for SplitBackend {
    type Device = <Dispatch as BackendTypes>::Device;

    type FloatTensorPrimitive = <Dispatch as BackendTypes>::FloatTensorPrimitive;

    type IntTensorPrimitive = <Dispatch as BackendTypes>::IntTensorPrimitive;

    type BoolTensorPrimitive = <Dispatch as BackendTypes>::BoolTensorPrimitive;

    type QuantizedTensorPrimitive = <Dispatch as BackendTypes>::QuantizedTensorPrimitive;

    type ComplexTensorPrimitive =
        SplitPrimitive<<Dispatch as BackendTypes>::FloatTensorPrimitive, 2>;
}

// Needs to be public to avoid a compile time error related to the visibility of the associated type for the tensor primitive in BackendTypes
/// A generic over the component tensors of a split-layout tensor. The components are stored as an array of tensors of the same primitive type
#[derive(Debug, Clone)]
#[allow(private_bounds)]
pub struct SplitPrimitive<T, const N: usize>(pub(super) [T; N])
where
    [(); N]: IsNotEmpty;

impl<const D: usize> From<SplitPrimitive<DispatchTensor, 2>> for SplitTensor<D, Complex> {
    fn from(val: SplitPrimitive<DispatchTensor, 2>) -> Self {
        let [left, right] = val.0;
        SplitTensor::new(BridgeTensor::float(left), BridgeTensor::float(right))
    }
}

impl<const D: usize> From<SplitTensor<D, Complex>> for SplitPrimitive<DispatchTensor, 2> {
    fn from(val: SplitTensor<D, Complex>) -> Self {
        let [left, right] = val.components;
        SplitPrimitive([left.into(), right.into()])
    }
}

#[allow(private_bounds)]
pub(crate) trait IsNotEmpty {
    #[allow(dead_code)]
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
