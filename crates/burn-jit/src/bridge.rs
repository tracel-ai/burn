use crate::{
    kernel, ops::to_device, tensor::JitTensor, FloatElement, IntElement, JitBackend, Runtime,
};
use burn_tensor::{
    backend::BackendBridge,
    ops::{FloatElem, FloatTensor},
};
use core::marker::PhantomData;

/// Handle precision conversion for the jit backend.
#[derive(Debug)]
pub struct PrecisionBridge<R, F: FloatElement, I: IntElement> {
    _runtime: PhantomData<R>,
    _float_elem: PhantomData<F>,
    _int_elem: PhantomData<I>,
}

impl<R, FOrigin, IOrigin, FTarget, ITarget> BackendBridge<JitBackend<R, FOrigin, IOrigin>>
    for PrecisionBridge<R, FTarget, ITarget>
where
    R: Runtime,
    FOrigin: FloatElement,
    IOrigin: IntElement,
    FTarget: FloatElement,
    ITarget: IntElement,
{
    type Target = JitBackend<R, FTarget, ITarget>;

    fn into_target<const D: usize>(
        tensor: FloatTensor<JitBackend<R, FOrigin, IOrigin>, D>,
        device: Option<burn_tensor::Device<Self::Target>>,
    ) -> FloatTensor<Self::Target, D> {
        let tensor = kernel::cast::<
            R,
            FloatElem<JitBackend<R, FOrigin, IOrigin>>,
            FloatElem<JitBackend<R, FTarget, ITarget>>,
            D,
        >(tensor);

        // The line below does the backend type cast.
        let tensor = JitTensor::new(tensor.client, tensor.device, tensor.shape, tensor.handle);

        if let Some(device) = &device {
            to_device(tensor, device)
        } else {
            tensor
        }
    }

    fn from_target<const D: usize>(
        tensor: FloatTensor<Self::Target, D>,
        device: Option<burn_tensor::Device<JitBackend<R, FOrigin, IOrigin>>>,
    ) -> FloatTensor<JitBackend<R, FOrigin, IOrigin>, D> {
        let tensor = kernel::cast::<
            R,
            FloatElem<JitBackend<R, FTarget, ITarget>>,
            FloatElem<JitBackend<R, FOrigin, IOrigin>>,
            D,
        >(tensor);
        // The line below does the backend type cast.
        let tensor = JitTensor::new(tensor.client, tensor.device, tensor.shape, tensor.handle);

        if let Some(device) = &device {
            to_device(tensor, device)
        } else {
            tensor
        }
    }
}
