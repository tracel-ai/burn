use crate::ops::activation::*;
use crate::ops::*;
use crate::tensor::ops::{TensorOpsIndex, TensorOpsReshape};
use crate::tensor::Element;
use crate::tensor::{Data, Distribution, Shape};
use crate::Gradients;

pub trait Backend:
    TensorOps<Self>
    + ModuleOps<Self>
    + Clone
    + Sized
    + Default
    + Send
    + Sync
    + std::fmt::Debug
    + 'static
{
    type Device: Copy + Clone + Default + std::fmt::Debug + Send + Sync;
    type Elem: Element;
    type FullPrecisionElem: Element;
    type FullPrecisionBackend: Backend<Elem = Self::FullPrecisionElem, Device = Self::Device>;
    type IntegerBackend: Backend<Elem = i64, Device = Self::Device>;
    type TensorPrimitive<const D: usize>: TensorOpsMatmul<Self::Elem, D>
        + std::ops::Add<Self::TensorPrimitive<D>, Output = Self::TensorPrimitive<D>>
        + TensorOpsTranspose<Self::Elem, D>
        + TensorOpsNeg<Self::Elem, D>
        + TensorOpsDetach<Self::Elem, D>
        + Zeros<Self::TensorPrimitive<D>>
        + Ones<Self::TensorPrimitive<D>>
        + TensorOpsReshape<Self, D>
        + TensorOpsPrecision<Self, D>
        + TensorOpsIndex<Self::Elem, D>
        + TensorOpsAggregation<Self, D>
        + TensorOpsExp<Self::Elem, D>
        + TensorOpsArg<Self, D>
        + TensorOpsCat<Self::Elem, D>
        + TensorOpsLog<Self::Elem, D>
        + TensorOpsErf<Self::Elem, D>
        + TensorOpsPow<Self::Elem, D>
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

    type BoolTensorPrimitive<const D: usize>: Clone + Send + Sync + 'static + std::fmt::Debug;

    fn from_data<const D: usize>(
        data: Data<Self::Elem, D>,
        device: Self::Device,
    ) -> Self::TensorPrimitive<D>;

    fn from_data_bool<const D: usize>(
        data: Data<bool, D>,
        device: Self::Device,
    ) -> Self::BoolTensorPrimitive<D>;

    fn ad_enabled() -> bool;
    fn name() -> String;
    fn seed(seed: u64);

    fn random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution<Self::Elem>,
        device: Self::Device,
    ) -> Self::TensorPrimitive<D>;

    fn zeros<const D: usize>(shape: Shape<D>, device: Self::Device) -> Self::TensorPrimitive<D> {
        Self::from_data(Data::zeros(shape), device)
    }

    fn ones<const D: usize>(shape: Shape<D>, device: Self::Device) -> Self::TensorPrimitive<D> {
        Self::from_data(Data::ones(shape), device)
    }
}

pub(crate) type ADBackendTensorPrimitive<const D: usize, B> =
    <<B as ADBackend>::InnerBackend as Backend>::TensorPrimitive<D>;

pub trait ADBackend: Backend {
    type InnerBackend: Backend<Device = Self::Device, Elem = Self::Elem>;

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
