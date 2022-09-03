use crate::ops::activation::*;
use crate::ops::*;
use crate::tensor::ops::{TensorOpsIndex, TensorOpsReshape};
use crate::tensor::Element;
use crate::tensor::{Data, Distribution, Shape};
use crate::Gradients;

pub trait Backend: Clone + Sized + Default + Send + Sync + std::fmt::Debug + 'static {
    type Device: Copy + Clone + Default + std::fmt::Debug + Send + Sync;
    type Elem: Element;
    type FullPrecisionElem: Element;
    type FullPrecisionBackend: Backend<Elem = Self::FullPrecisionElem, Device = Self::Device>;
    type IntegerBackend: Backend<Elem = i64, Device = Self::Device>;
    type TensorPrimitive<const D: usize>: TensorOpsUtilities<Self::Elem, D>
        + TensorOpsMatmul<Self::Elem, D>
        + TensorOpsTranspose<Self::Elem, D>
        + TensorOpsMul<Self::Elem, D>
        + TensorOpsDiv<Self::Elem, D>
        + TensorOpsNeg<Self::Elem, D>
        + TensorOpsAdd<Self::Elem, D>
        + TensorOpsSub<Self::Elem, D>
        + Zeros<Self::TensorPrimitive<D>>
        + Ones<Self::TensorPrimitive<D>>
        + TensorOpsReshape<Self, D>
        + TensorOpsPrecision<Self, D>
        + TensorOpsDevice<Self, D>
        + TensorOpsIndex<Self::Elem, D>
        + TensorOpsAggregation<Self, D>
        + TensorOpsExp<Self::Elem, D>
        + TensorOpsArg<Self, D>
        + TensorOpsCat<Self::Elem, D>
        + TensorOpsLog<Self::Elem, D>
        + TensorOpsMask<Self, D>
        + TensorOpsMapComparison<Self, D>
        + ReLU<Self::Elem, D>
        + Clone
        + Send
        + Sync
        + Send
        + Sync
        + 'static
        + std::fmt::Debug;

    type BoolTensorPrimitive<const D: usize>: TensorOpsUtilities<bool, D>
        + Clone
        + Send
        + Sync
        + 'static
        + std::fmt::Debug;

    fn from_data<const D: usize>(
        data: Data<Self::Elem, D>,
        device: Self::Device,
    ) -> Self::TensorPrimitive<D>;

    fn from_data_bool<const D: usize>(
        data: Data<bool, D>,
        device: Self::Device,
    ) -> Self::BoolTensorPrimitive<D>;

    fn random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution<Self::Elem>,
        device: Self::Device,
    ) -> Self::TensorPrimitive<D>;

    fn zeros<const D: usize>(shape: Shape<D>, device: Self::Device) -> Self::TensorPrimitive<D>;

    fn ones<const D: usize>(shape: Shape<D>, device: Self::Device) -> Self::TensorPrimitive<D>;

    fn ad_enabled() -> bool;
    fn name() -> String;
}

pub(crate) type ADBackendTensorPrimitive<const D: usize, B> =
    <<B as ADBackend>::InnerBackend as Backend>::TensorPrimitive<D>;

pub trait ADBackend: Backend {
    type InnerBackend: Backend<Device = Self::Device>;

    fn backward<const D: usize>(tensor: &Self::TensorPrimitive<D>) -> Gradients;
    fn grad<const D: usize>(
        tensor: &Self::TensorPrimitive<D>,
        grads: &Gradients,
    ) -> Option<ADBackendTensorPrimitive<D, Self>>;
    fn inner<const D: usize>(
        tensor: &Self::TensorPrimitive<D>,
    ) -> <Self::InnerBackend as Backend>::TensorPrimitive<D>;
    fn from_inner<const D: usize>(
        tensor: <Self::InnerBackend as Backend>::TensorPrimitive<D>,
    ) -> Self::TensorPrimitive<D>;
}
