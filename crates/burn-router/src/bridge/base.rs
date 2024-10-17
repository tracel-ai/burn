use burn_tensor::{backend::DeviceOps, Element, Shape};

/// Allows tensors to be transferred between multiple backends.
pub trait MultiBackendBridge: Send + Sync + 'static {
    /// The type that can be used to point to a tensor of any kind.
    type TensorHandle;
    /// Device type used by the backends.
    type Device: DeviceOps;
    /// Float element type.
    type FloatElem: Element;
    /// Int element type.
    type IntElem: Element;

    /// Change the backend of the given float tensor.
    fn change_backend_float(
        tensor: Self::TensorHandle,
        shape: Shape,
        target_device: &Self::Device,
    ) -> Self::TensorHandle;

    /// Change the backend of the given int tensor.
    fn change_backend_int(
        tensor: Self::TensorHandle,
        shape: Shape,
        target_device: &Self::Device,
    ) -> Self::TensorHandle;

    /// Change the backend of the given bool tensor.
    fn change_backend_bool(
        tensor: Self::TensorHandle,
        shape: Shape,
        target_device: &Self::Device,
    ) -> Self::TensorHandle;

    // TODO: change_backend_quantized
}
