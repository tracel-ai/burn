use crate::backend::Backend;
use crate::{execute_ops, register_ops};
use crate::{
    graph::ops::{UnaryOps, UnaryOpsNodeState},
    tensor::backend::autodiff::ADTensor,
};
use crate::{ops::TensorOpsDevice, Element};
use rand::distributions::Standard;

register_ops!(
    ops UnaryOps,
    name ADTensorDeviceOps state B::Device,
    partial |device, state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>|{
        state.output.grad().to_device(device)
    },
);

macro_rules! define_impl {
    (
        $backend:ty,
        $backend_inner:ty
    ) => {
        impl<E: Element, const D: usize> TensorOpsDevice<$backend, D>
            for <$backend as Backend>::TensorPrimitive<D>
        where
            E: Element,
            Standard: rand::distributions::Distribution<E>,
        {
            fn device(&self) -> <$backend as Backend>::Device {
                TensorOpsDevice::device(&self.tensor())
            }

            fn to_device(
                &self,
                device: <$backend as Backend>::Device,
            ) -> ADTensor<D, $backend_inner> {
                let tensor = self.tensor();
                let node = execute_ops!(
                    input self.node.clone(),
                    out TensorOpsDevice::to_device(&tensor, device),
                    ops ADTensorDeviceOps::<$backend_inner, D>::new(tensor.device()),
                );
                self.from_existing(node)
            }
        }
    };
}

#[cfg(feature = "ndarray")]
define_impl!(
    crate::tensor::backend::autodiff::ADBackendNdArray::<E>,
    crate::tensor::backend::ndarray::NdArrayBackend::<E>
);

#[cfg(feature = "tch")]
define_impl!(
    crate::tensor::backend::autodiff::ADBackendTch::<E>,
    crate::tensor::backend::tch::TchBackend::<E>
);
