use crate::{
    backend::{Backend, DeviceOps},
    repr::{OperationDescription, ReprBackend, TensorDescription},
    router::{RouterTensor, Runner, RunnerClient},
    DType, Shape, TensorData,
};

pub trait MultiBackendBridge: Send + Sync + 'static {
    // for now, but we might just change `to_backend` to return a TensorDescription instead
    // and since quantized tensor actually have a diff description, we might need to have backend switches
    // for all primitive types
    type TensorHandle;
    type Device;

    // TODO: change_backend_int, change_backend_bool, change_backend_quantized
    fn change_backend_float(
        tensor: Self::TensorHandle,
        shape: Shape,
        device: &Self::Device,
    ) -> Self::TensorHandle;

    fn change_backend_int(
        tensor: Self::TensorHandle,
        shape: Shape,
        device: &Self::Device,
    ) -> Self::TensorHandle;

    fn change_backend_bool(
        tensor: Self::TensorHandle,
        shape: Shape,
        device: &Self::Device,
    ) -> Self::TensorHandle;
}
