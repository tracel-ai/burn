use core::marker::PhantomData;

use burn_tensor::{
    repr::{ReprBackend, TensorHandle},
    try_read_sync, Shape,
};

use super::base::MultiBackendBridge;
use crate::{MultiDevice2, TensorHandle2};

/// Simply transfers tensors between backends via the underlying [tensor data](burn_tensor::TensorData).
pub struct ByteBridge<Backends> {
    backends: PhantomData<Backends>,
}

// TODO: refactor w/ visitor?
// pub trait BackendSwitchVisitor<B1: Backend> {
//     fn from_backend<B2: Backend>(handle: B2::FloatTensorPrimitive) -> B1::FloatTensorPrimitive;
// }

impl<B1: ReprBackend, B2: ReprBackend> MultiBackendBridge for ByteBridge<(B1, B2)> {
    type TensorHandle = TensorHandle2<B1, B2>;
    type Device = MultiDevice2<B1, B2>;

    fn change_backend_float(
        tensor: Self::TensorHandle,
        shape: Shape,
        target_device: &Self::Device,
    ) -> Self::TensorHandle {
        let msg = "Failed to read tensor data synchronously.
This can happen on platforms that don't support blocking futures like WASM.";
        match tensor {
            TensorHandle2::Handle1(handle) => match target_device {
                MultiDevice2::Device1(device) => {
                    // Same backend
                    let tensor = B1::float_tensor(TensorHandle { handle, shape });
                    let tensor = B1::float_to_device(tensor, device);
                    let handle = B1::float_tensor_handle(tensor);
                    TensorHandle2::Handle1(handle)
                }
                MultiDevice2::Device2(device) => {
                    let tensor = B1::float_tensor(TensorHandle { handle, shape });
                    let data = try_read_sync(B1::float_into_data(tensor)).expect(msg);
                    let tensor = B2::float_from_data(data, device);
                    let handle = B2::float_tensor_handle(tensor);
                    TensorHandle2::Handle2(handle)
                }
            },
            TensorHandle2::Handle2(handle) => match target_device {
                MultiDevice2::Device1(device) => {
                    let tensor = B2::float_tensor(TensorHandle { handle, shape });
                    let data = try_read_sync(B2::float_into_data(tensor)).expect(msg);
                    let tensor = B1::float_from_data(data, device);
                    let handle = B1::float_tensor_handle(tensor);
                    TensorHandle2::Handle1(handle)
                }
                MultiDevice2::Device2(device) => {
                    // Same backend
                    let tensor = B2::float_tensor(TensorHandle { handle, shape });
                    let tensor = B2::float_to_device(tensor, device);
                    let handle = B2::float_tensor_handle(tensor);
                    TensorHandle2::Handle2(handle)
                }
            },
        }
    }

    fn change_backend_int(
        tensor: Self::TensorHandle,
        shape: Shape,
        target_device: &Self::Device,
    ) -> Self::TensorHandle {
        let msg = "Failed to read tensor data synchronously.
This can happen on platforms that don't support blocking futures like WASM.";
        match tensor {
            TensorHandle2::Handle1(handle) => match target_device {
                MultiDevice2::Device1(device) => {
                    // Same backend
                    let tensor = B1::int_tensor(TensorHandle { handle, shape });
                    let tensor = B1::int_to_device(tensor, device);
                    let handle = B1::int_tensor_handle(tensor);
                    TensorHandle2::Handle1(handle)
                }
                MultiDevice2::Device2(device) => {
                    let tensor = B1::int_tensor(TensorHandle { handle, shape });
                    let data = try_read_sync(B1::int_into_data(tensor)).expect(msg);
                    let tensor = B2::int_from_data(data, device);
                    let handle = B2::int_tensor_handle(tensor);
                    TensorHandle2::Handle2(handle)
                }
            },
            TensorHandle2::Handle2(handle) => match target_device {
                MultiDevice2::Device1(device) => {
                    let tensor = B2::int_tensor(TensorHandle { handle, shape });
                    let data = try_read_sync(B2::int_into_data(tensor)).expect(msg);
                    let tensor = B1::int_from_data(data, device);
                    let handle = B1::int_tensor_handle(tensor);
                    TensorHandle2::Handle1(handle)
                }
                MultiDevice2::Device2(device) => {
                    // Same backend
                    let tensor = B2::int_tensor(TensorHandle { handle, shape });
                    let tensor = B2::int_to_device(tensor, device);
                    let handle = B2::int_tensor_handle(tensor);
                    TensorHandle2::Handle2(handle)
                }
            },
        }
    }

    fn change_backend_bool(
        tensor: Self::TensorHandle,
        shape: Shape,
        target_device: &Self::Device,
    ) -> Self::TensorHandle {
        let msg = "Failed to read tensor data synchronously.
        This can happen on platforms that don't support blocking futures like WASM.";
        match tensor {
            TensorHandle2::Handle1(handle) => match target_device {
                MultiDevice2::Device1(device) => {
                    // Same backend
                    let tensor = B1::bool_tensor(TensorHandle { handle, shape });
                    let tensor = B1::bool_to_device(tensor, device);
                    let handle = B1::bool_tensor_handle(tensor);
                    TensorHandle2::Handle1(handle)
                }
                MultiDevice2::Device2(device) => {
                    let tensor = B1::bool_tensor(TensorHandle { handle, shape });
                    let data = try_read_sync(B1::bool_into_data(tensor)).expect(msg);
                    let tensor = B2::bool_from_data(data, device);
                    let handle = B2::bool_tensor_handle(tensor);
                    TensorHandle2::Handle2(handle)
                }
            },
            TensorHandle2::Handle2(handle) => match target_device {
                MultiDevice2::Device1(device) => {
                    let tensor = B2::bool_tensor(TensorHandle { handle, shape });
                    let data = try_read_sync(B2::bool_into_data(tensor)).expect(msg);
                    let tensor = B1::bool_from_data(data, device);
                    let handle = B1::bool_tensor_handle(tensor);
                    TensorHandle2::Handle1(handle)
                }
                MultiDevice2::Device2(device) => {
                    // Same backend
                    let tensor = B2::bool_tensor(TensorHandle { handle, shape });
                    let tensor = B2::bool_to_device(tensor, device);
                    let handle = B2::bool_tensor_handle(tensor);
                    TensorHandle2::Handle2(handle)
                }
            },
        }
    }
}
