use core::marker::PhantomData;

use super::{base::MultiBackendBridge, Handle2, MultiDevice2};
use crate::repr::ReprBackend;

pub struct ByteBridge<Backends> {
    backends: PhantomData<Backends>,
}

// Concrete implementation for bridge between two backends.
impl<B1: ReprBackend, B2: ReprBackend> MultiBackendBridge for ByteBridge<(B1, B2)> {
    type TensorType = Handle2<B1, B2>;
    type Device = MultiDevice2<B1, B2>;

    fn to_backend(&self, tensor: Self::TensorType, device: &Self::Device) -> Self::TensorType {
        let msg = "Failed to read tensor data synchronously.
This can happen on platforms that don't support blocking futures like WASM.";
        match tensor {
            Handle2::FloatHandle1(tensor) => {
                match device {
                    MultiDevice2::Device1(device) => {
                        Handle2::FloatHandle1(B1::float_to_device(tensor, device))
                    } // same backend
                    MultiDevice2::Device2(device) => {
                        let data = crate::try_read_sync(B1::float_into_data(tensor)).expect(msg);
                        Handle2::FloatHandle2(B2::float_from_data(data, device))
                    }
                }
            }
            Handle2::FloatHandle2(tensor) => {
                match device {
                    MultiDevice2::Device1(device) => {
                        let data = crate::try_read_sync(B2::float_into_data(tensor)).expect(msg);
                        Handle2::FloatHandle1(B1::float_from_data(data, device))
                    }
                    MultiDevice2::Device2(device) => {
                        Handle2::FloatHandle2(B2::float_to_device(tensor, device))
                    } // same backend
                }
            }
            Handle2::IntHandle1(tensor) => todo!(),
            Handle2::IntHandle2(tensor) => todo!(),
        }
    }
}
