use core::marker::PhantomData;

use super::{base::MultiBackendBridge, Handle2, MultiDevice2, TensorHandle2};
use crate::{
    repr::{ReprBackend, TensorHandle},
    Shape,
};

pub struct ByteBridge<Backends> {
    backends: PhantomData<Backends>,
}

// TODO: refactor w/ visitor?
// pub trait BackendSwitchVisitor<B1: Backend> {
//     fn from_backend<B2: Backend>(handle: B2::FloatTensorPrimitive) -> B1::FloatTensorPrimitive;
// }

// Concrete implementation for bridge between two backends.
impl<B1: ReprBackend, B2: ReprBackend> MultiBackendBridge for ByteBridge<(B1, B2)>
// where
//     B1: ReprBackend<Handle = TensorHandle<<B1 as ReprBackend>::Handle>>,
//     B2: ReprBackend<Handle = TensorHandle<<B2 as ReprBackend>::Handle>>,
{
    // TensorType should be handle type
    type TensorType = TensorHandle2<B1, B2>;
    type Device = MultiDevice2<B1, B2>;

    fn change_backend_float(
        tensor: Self::TensorType,
        shape: Shape,
        device: &Self::Device,
    ) -> Self::TensorType {
        let msg = "Failed to read tensor data synchronously.
This can happen on platforms that don't support blocking futures like WASM.";
        match tensor {
            TensorHandle2::Handle1(handle) => match device {
                MultiDevice2::Device1(device) => {
                    // Same backend
                    let tensor = B1::float_tensor(TensorHandle { handle, shape });
                    let tensor = B1::float_to_device(tensor, device);
                    let handle = B1::float_tensor_handle(tensor);
                    TensorHandle2::Handle1(handle)
                }
                MultiDevice2::Device2(device) => {
                    let tensor = B1::float_tensor(TensorHandle { handle, shape });
                    let data = crate::try_read_sync(B1::float_into_data(tensor)).expect(msg);
                    let tensor = B2::float_from_data(data, device);
                    let handle = B2::float_tensor_handle(tensor);
                    TensorHandle2::Handle2(handle)
                }
            },
            TensorHandle2::Handle2(handle) => match device {
                MultiDevice2::Device1(device) => {
                    let tensor = B2::float_tensor(TensorHandle { handle, shape });
                    let data = crate::try_read_sync(B2::float_into_data(tensor)).expect(msg);
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

    //     fn to_backend(tensor: Self::TensorType, device: &Self::Device) -> Self::TensorType {
    //         let msg = "Failed to read tensor data synchronously.
    // This can happen on platforms that don't support blocking futures like WASM.";
    //         match tensor {
    //             Handle2::FloatHandle1(tensor) => {
    //                 match device {
    //                     MultiDevice2::Device1(device) => {
    //                         Handle2::FloatHandle1(B1::float_to_device(tensor, device))
    //                     } // same backend
    //                     MultiDevice2::Device2(device) => {
    //                         let data = crate::try_read_sync(B1::float_into_data(tensor)).expect(msg);
    //                         Handle2::FloatHandle2(B2::float_from_data(data, device))
    //                     }
    //                 }
    //             }
    //             Handle2::FloatHandle2(tensor) => {
    //                 match device {
    //                     MultiDevice2::Device1(device) => {
    //                         let data = crate::try_read_sync(B2::float_into_data(tensor)).expect(msg);
    //                         Handle2::FloatHandle1(B1::float_from_data(data, device))
    //                     }
    //                     MultiDevice2::Device2(device) => {
    //                         Handle2::FloatHandle2(B2::float_to_device(tensor, device))
    //                     } // same backend
    //                 }
    //             }
    //             Handle2::IntHandle1(tensor) => todo!(),
    //             Handle2::IntHandle2(tensor) => todo!(),
    //         }
    //     }
}
