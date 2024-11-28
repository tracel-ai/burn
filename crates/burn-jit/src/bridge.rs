use crate::{
    element::BoolElement, kernel, ops::to_device, tensor::JitTensor, FloatElement, IntElement,
    JitBackend, JitRuntime,
};
use burn_tensor::{
    backend::BackendBridge,
    ops::{FloatElem, FloatTensor},
};
use core::marker::PhantomData;

/// Handle precision conversion for the jit backend.
#[derive(Debug)]
pub struct PrecisionBridge<R, F: FloatElement, I: IntElement, BT: BoolElement = u32> {
    _runtime: PhantomData<R>,
    _float_elem: PhantomData<F>,
    _int_elem: PhantomData<I>,
    _bool_elem: PhantomData<BT>,
}

impl<R, FOrigin, IOrigin, BOrigin, FTarget, ITarget, BTarget>
    BackendBridge<JitBackend<R, FOrigin, IOrigin, BOrigin>>
    for PrecisionBridge<R, FTarget, ITarget, BTarget>
where
    R: JitRuntime,
    FOrigin: FloatElement,
    IOrigin: IntElement,
    BOrigin: BoolElement,
    FTarget: FloatElement,
    ITarget: IntElement,
    BTarget: BoolElement,
{
    type Target = JitBackend<R, FTarget, ITarget, BTarget>;

    fn into_target(
        tensor: FloatTensor<JitBackend<R, FOrigin, IOrigin, BOrigin>>,
        device: Option<burn_tensor::Device<Self::Target>>,
    ) -> FloatTensor<Self::Target> {
        let tensor = kernel::cast::<
            R,
            FloatElem<JitBackend<R, FOrigin, IOrigin, BOrigin>>,
            FloatElem<JitBackend<R, FTarget, ITarget, BTarget>>,
        >(tensor);

        // The line below does the backend type cast.
        let tensor = JitTensor::new_contiguous(
            tensor.client,
            tensor.device,
            tensor.shape,
            tensor.handle,
            FTarget::dtype(),
        );

        if let Some(device) = &device {
            to_device(tensor, device)
        } else {
            tensor
        }
    }

    fn from_target(
        tensor: FloatTensor<Self::Target>,
        device: Option<burn_tensor::Device<JitBackend<R, FOrigin, IOrigin, BOrigin>>>,
    ) -> FloatTensor<JitBackend<R, FOrigin, IOrigin, BOrigin>> {
        let tensor = kernel::cast::<
            R,
            FloatElem<JitBackend<R, FTarget, ITarget, BTarget>>,
            FloatElem<JitBackend<R, FOrigin, IOrigin, BOrigin>>,
        >(tensor);
        // The line below does the backend type cast.
        let tensor = JitTensor::new_contiguous(
            tensor.client,
            tensor.device,
            tensor.shape,
            tensor.handle,
            FOrigin::dtype(),
        );

        if let Some(device) = &device {
            to_device(tensor, device)
        } else {
            tensor
        }
    }
}
