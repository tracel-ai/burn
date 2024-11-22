use crate::{
    element::{BoolElement, ByteElement},
    kernel,
    ops::to_device,
    tensor::JitTensor,
    FloatElement, IntElement, JitBackend, JitRuntime,
};
use burn_tensor::{
    backend::BackendBridge,
    ops::{FloatElem, FloatTensor},
};
use core::marker::PhantomData;

/// Handle precision conversion for the jit backend.
#[derive(Debug)]
pub struct PrecisionBridge<
    R,
    F: FloatElement,
    I: IntElement,
    B: BoolElement = u32,
    P: ByteElement = u32,
> {
    _runtime: PhantomData<R>,
    _float_elem: PhantomData<F>,
    _int_elem: PhantomData<I>,
    _bool_elem: PhantomData<B>,
    _byte_elem: PhantomData<P>,
}

impl<R, FOrigin, IOrigin, BOrigin, POrigin, FTarget, ITarget, BTarget, PTarget>
    BackendBridge<JitBackend<R, FOrigin, IOrigin, BOrigin, POrigin>>
    for PrecisionBridge<R, FTarget, ITarget, BTarget, PTarget>
where
    R: JitRuntime,
    FOrigin: FloatElement,
    IOrigin: IntElement,
    BOrigin: BoolElement,
    POrigin: ByteElement,
    FTarget: FloatElement,
    ITarget: IntElement,
    BTarget: BoolElement,
    PTarget: ByteElement,
{
    type Target = JitBackend<R, FTarget, ITarget, BTarget, PTarget>;

    fn into_target(
        tensor: FloatTensor<JitBackend<R, FOrigin, IOrigin, BOrigin, POrigin>>,
        device: Option<burn_tensor::Device<Self::Target>>,
    ) -> FloatTensor<Self::Target> {
        let tensor = kernel::cast::<
            R,
            FloatElem<JitBackend<R, FOrigin, IOrigin, BOrigin, POrigin>>,
            FloatElem<JitBackend<R, FTarget, ITarget, BTarget, PTarget>>,
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
        device: Option<burn_tensor::Device<JitBackend<R, FOrigin, IOrigin, BOrigin, POrigin>>>,
    ) -> FloatTensor<JitBackend<R, FOrigin, IOrigin, BOrigin, POrigin>> {
        let tensor = kernel::cast::<
            R,
            FloatElem<JitBackend<R, FTarget, ITarget, BTarget, PTarget>>,
            FloatElem<JitBackend<R, FOrigin, IOrigin, BOrigin, POrigin>>,
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
