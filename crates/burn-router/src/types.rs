use burn_std::future::DynFut;
use burn_ir::{BackendIr, OperationIr, TensorHandle, TensorId, TensorIr};
use burn_tensor::{
    DType, Shape, TensorData,
    backend::{Backend, DeviceId, DeviceOps},
    try_read_sync,
};

use crate::{
    ByteBridge, DirectChannel, MultiBackendBridge, RouterTensor, Runner, RunnerChannel,
    RunnerClient,
};

/// Implement multi backend types, with enums having one variant per backend.
macro_rules! impl_multi_backend_types {
    // Match the default backend and at least one other backend, with rest being optional
    ($module_name:ident, $DefaultBackend:ident, $($OtherBackend:ident),+) => {
        /// Module containing the essential types for multi-backend operations.
        ///
        /// - `Handle`: the type used to point to a tensor (defined for all backends).
        /// - `MultiRunnerClient`: a client for multiple runners (each responsible to execute tensor operations on a given backend).
        /// - `DirectChannel`: a local channel with direct connection to the backend runner clients.
        /// - `ByteBridge`: a simple multi-backend bridge that transfers tensors via the underlying [tensor data](burn_tensor::TensorData).
        ///
        /// Each enum type is defined with backend identifiers as variant names (e.g., `B1` and `B2` for dual backends).
        pub mod $module_name {
            use super::*;

            /// The type that can be used to point to a tensor of any kind.
            /// Each backend has its own variant.
            pub enum Handle<$DefaultBackend: BackendIr, $($OtherBackend: BackendIr),+> {
                #[allow(missing_docs)]
                $DefaultBackend($DefaultBackend::Handle),
                $(
                    #[allow(missing_docs)]
                    $OtherBackend($OtherBackend::Handle),
                )+
            }

            /// The device type used by a backend.
            /// Each backend has its own variant.
            #[derive(Clone, Debug)]
            pub enum MultiDevice<$DefaultBackend: Backend, $($OtherBackend: Backend),+> {
                #[allow(missing_docs)]
                $DefaultBackend($DefaultBackend::Device),
                $(
                    #[allow(missing_docs)]
                    $OtherBackend($OtherBackend::Device),
                )+
            }
            impl<$DefaultBackend: Backend, $($OtherBackend: Backend),+> PartialEq for MultiDevice<$DefaultBackend, $($OtherBackend),+> {
                fn eq(&self, other: &Self) -> bool {
                    match (self, other) {
                        (Self::$DefaultBackend(lhs), Self::$DefaultBackend(rhs)) => lhs == rhs,
                        $(
                            (Self::$OtherBackend(lhs), Self::$OtherBackend(rhs)) => lhs == rhs,
                        )+
                        _ => false,
                    }
                }
            }

            // Default implementation always returns the first backend's device
            impl<$DefaultBackend: Backend, $($OtherBackend: Backend),+> Default for MultiDevice<$DefaultBackend, $($OtherBackend),+> {
                fn default() -> Self {
                    Self::$DefaultBackend($DefaultBackend::Device::default())
                }
            }

            impl<$DefaultBackend: Backend, $($OtherBackend: Backend),+> burn_std::device::Device for MultiDevice<$DefaultBackend, $($OtherBackend),+> {
                fn from_id(_device_id: DeviceId) -> Self {
                    // TODO: Should be fix with the new router backend.
                    Default::default()
                }

                fn to_id(&self) -> DeviceId {
                    match self {
                        Self::$DefaultBackend(device) => device.id(),
                        $(
                            Self::$OtherBackend(device) => device.id(),
                        )+
                    }
                }

                fn device_count(_type_id: u16) -> usize {
                    1
                }
            }

            impl<$DefaultBackend: Backend, $($OtherBackend: Backend),+> DeviceOps for MultiDevice<$DefaultBackend, $($OtherBackend),+> {}

            /// A local client with multiple runners (each responsible to execute tensor operations on a given backend).
            #[derive(Clone)]
            pub enum MultiRunnerClient<$DefaultBackend: BackendIr, $($OtherBackend: BackendIr),+> {
                #[allow(missing_docs)]
                $DefaultBackend(Runner<$DefaultBackend>),
                $(
                    #[allow(missing_docs)]
                    $OtherBackend(Runner<$OtherBackend>),
                )+
            }

            impl<$DefaultBackend: BackendIr, $($OtherBackend: BackendIr),+> RunnerClient for MultiRunnerClient<$DefaultBackend, $($OtherBackend),+>
            {
               type Device = MultiDevice<$DefaultBackend, $($OtherBackend),+>;

                fn register_op(&self, op: OperationIr) {
                    match self {
                        Self::$DefaultBackend(runner) => runner.register_op(op),
                        $(
                            Self::$OtherBackend(runner) => runner.register_op(op),
                        )+
                    }
                }

                fn read_tensor(&self, tensor: TensorIr) -> DynFut<TensorData> {
                    match self {
                        Self::$DefaultBackend(runner) => runner.read_tensor(tensor),
                        $(
                            Self::$OtherBackend(runner) => runner.read_tensor(tensor),
                        )+
                    }
                }

                fn register_tensor_data(&self, data: TensorData) -> RouterTensor<Self> {
                    match self {
                        Self::$DefaultBackend(runner) => {
                            let desc = runner.register_tensor_data_desc(data);
                            RouterTensor::new(desc.id, desc.shape, desc.dtype, self.clone())
                        }
                        $(
                            Self::$OtherBackend(runner) => {
                                let desc = runner.register_tensor_data_desc(data);
                                RouterTensor::new(desc.id, desc.shape, desc.dtype, self.clone())
                            }
                        )+
                    }
                }

                fn device(&self) -> Self::Device {
                    match self {
                        Self::$DefaultBackend(runner) => MultiDevice::$DefaultBackend(runner.device()),
                        $(
                            Self::$OtherBackend(runner) => MultiDevice::$OtherBackend(runner.device()),
                        )+
                    }
                }

                fn sync(&self) {
                    match self {
                        Self::$DefaultBackend(runner) => runner.sync(),
                        $(
                            Self::$OtherBackend(runner) => runner.sync(),
                        )+
                    };
                }

                fn seed(&self, seed: u64) {
                    match self {
                        Self::$DefaultBackend(runner) => runner.seed(seed),
                        $(
                            Self::$OtherBackend(runner) => runner.seed(seed),
                        )+
                    }
                }

                fn create_empty_handle(&self) -> TensorId {
                            match self {
                        Self::$DefaultBackend(runner) => runner.create_empty_handle(),
                        $(
                            Self::$OtherBackend(runner) => runner.create_empty_handle(),
                        )+
                    }
                }
            }

            impl<$DefaultBackend: BackendIr, $($OtherBackend: BackendIr),+, Br> RunnerChannel for DirectChannel<($DefaultBackend, $($OtherBackend),+), Br>
            where
                Br: MultiBackendBridge<TensorHandle = Handle<$DefaultBackend, $($OtherBackend),+>, Device = MultiDevice<$DefaultBackend, $($OtherBackend),+>>,
            {
                type Device = Br::Device;

                type Bridge = Br;

                type FloatElem = $DefaultBackend::FloatElem;
                type IntElem = $DefaultBackend::IntElem;
                type BoolElem = $DefaultBackend::BoolElem;

                type Client = MultiRunnerClient<$DefaultBackend, $($OtherBackend),+>;

                fn init_client(device: &Self::Device) -> Self::Client {
                    match device {
                        MultiDevice::$DefaultBackend(device) => MultiRunnerClient::$DefaultBackend(Runner::new(device.clone())),
                        $(
                            MultiDevice::$OtherBackend(device) => MultiRunnerClient::$OtherBackend(Runner::new(device.clone())),
                        )+
                    }
                }

                fn get_tensor_handle(
                    tensor: &TensorIr,
                    client: &Self::Client,
                ) -> <Self::Bridge as MultiBackendBridge>::TensorHandle {
                    match client {
                        MultiRunnerClient::$DefaultBackend(runner) => Handle::$DefaultBackend(runner.get_tensor_handle(tensor)),
                        $(
                            MultiRunnerClient::$OtherBackend(runner) => Handle::$OtherBackend(runner.get_tensor_handle(tensor)),
                        )+
                    }
                }

                fn register_tensor(
                    client: &Self::Client,
                    handle: <Self::Bridge as MultiBackendBridge>::TensorHandle,
                    shape: Shape,
                    dtype: DType,
                ) -> RouterTensor<Self::Client> {
                    match client {
                        MultiRunnerClient::$DefaultBackend(runner) => match handle {
                            Handle::$DefaultBackend(handle) => runner.register_tensor(handle, shape, dtype, client.clone()),
                            _ => unreachable!("Can't register tensor handle for another backend."),
                        },
                        $(
                            MultiRunnerClient::$OtherBackend(runner) =>  match handle {
                                Handle::$OtherBackend(handle) => runner.register_tensor(handle, shape, dtype, client.clone()),
                                _ => unreachable!("Can't register tensor handle for another backend."),
                            },
                        )+
                    }
                }

                fn name(_device: &Self::Device) -> String {
                    let mut name = format!("{}", $DefaultBackend::name(&<$DefaultBackend::Device as Default>::default()));
                    $(
                        name.push_str(&format!(", {}", $OtherBackend::name(&<$OtherBackend::Device as Default>::default())));
                    )+
                    format!("direct<({})>", name)
                }
            }

            impl<$DefaultBackend: BackendIr, $($OtherBackend: BackendIr),+> MultiBackendBridge for ByteBridge<($DefaultBackend, $($OtherBackend),+)> {
                type TensorHandle = Handle<$DefaultBackend, $($OtherBackend),+>;
                type Device = MultiDevice<$DefaultBackend, $($OtherBackend),+>;

                fn change_backend_float(
                    tensor: Self::TensorHandle,
                    shape: Shape,
                    target_device: &Self::Device,
                ) -> Self::TensorHandle {
                    multi_backend_match!(shape, (tensor, target_device) : $DefaultBackend, $($OtherBackend),+)
                }

                fn change_backend_int(
                    tensor: Self::TensorHandle,
                    shape: Shape,
                    target_device: &Self::Device,
                ) -> Self::TensorHandle {
                    multi_backend_match!(shape, (tensor, target_device) : $DefaultBackend, $($OtherBackend),+)
                }

                fn change_backend_bool(
                    tensor: Self::TensorHandle,
                    shape: Shape,
                    target_device: &Self::Device,
                ) -> Self::TensorHandle {
                    multi_backend_match!(shape, (tensor, target_device) : $DefaultBackend, $($OtherBackend),+)
                }

            }
        }
    };
}

macro_rules! bridge {
    ($Backend:ident, $handle:expr, $device:expr, $shape:expr) => {{
        // Bridge for the same backend
        let tensor = $Backend::float_tensor(TensorHandle {
            handle: $handle,
            shape: $shape,
        });
        let tensor = $Backend::float_to_device(tensor, $device);
        let handle = $Backend::float_tensor_handle(tensor);
        Handle::$Backend(handle)
    }};
    ($BackendA:ident, $BackendB:ident, $handle:expr, $device:expr, $shape:expr) => {{
        // Byte bridge between two backends
        let tensor = $BackendA::float_tensor(TensorHandle { handle: $handle, shape: $shape });
        let data = try_read_sync($BackendA::float_into_data(tensor)).expect(
            "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM."
        );
        let tensor = $BackendB::float_from_data(data, $device);
        let handle = $BackendB::float_tensor_handle(tensor);
        Handle::$BackendB(handle)
    }};
}

macro_rules! multi_backend_match {
    ($shape:expr, ($handle:expr, $device:expr) : $DefaultBackend:ident, $($OtherBackend:ident),+) => {
        multi_backend_match! (
            @step
            $shape,
            ($handle, $device);
            {
                (Handle::$DefaultBackend(handle), MultiDevice::$DefaultBackend(device)) => bridge!($DefaultBackend, handle, device, $shape),
                $(
                    (Handle::$DefaultBackend(handle), MultiDevice::$OtherBackend(device)) => bridge!($DefaultBackend, $OtherBackend, handle, device, $shape),
                    (Handle::$OtherBackend(handle), MultiDevice::$DefaultBackend(device)) => bridge!($OtherBackend, $DefaultBackend, handle, device, $shape),
                    (Handle::$OtherBackend(handle), MultiDevice::$OtherBackend(device)) => bridge!($OtherBackend, handle, device, $shape),
                )+
            };
            $($OtherBackend),+
        )
    };

    (@step
        $shape:expr,
        $pats:tt;
        { $($arms:tt)* };
        $BackendA:ident,
        $($OtherBackend:ident),+
    ) => {
        multi_backend_match! (
            @step
            $shape,
            $pats;
            {
                $($arms)*
                $(
                    (Handle::$BackendA(handle), MultiDevice::$OtherBackend(device)) => bridge!($BackendA, $OtherBackend, handle, device, $shape),
                    (Handle::$OtherBackend(handle), MultiDevice::$BackendA(device)) => bridge!($OtherBackend, $BackendA, handle, device, $shape),
                )*
            };
            $($OtherBackend),*
        )
    };

    (@step
        $shape:expr,
        ($handle:expr, $device:expr);
        { $($arms:tt)* };
        $($BackendA:ident)?
    ) => {
        match ($handle, $device) {
            $($arms)*
        }
    };
}

// Implement multi-backend types and byte bridge for up to 4 backends
impl_multi_backend_types!(duo, B1, B2);
impl_multi_backend_types!(trio, B1, B2, B3);
impl_multi_backend_types!(quad, B1, B2, B3, B4);

#[cfg(not(target_os = "windows"))] // cannot find a wgpu adapter on windows CI
#[cfg(test)]
mod tests {
    use burn_tensor::{Tensor, backend::Backend};

    use super::*;
    use crate::tests::{TestBackend, TestBackend1, TestBackend2};

    #[test]
    fn should_support_dual_byte_bridge() {
        let device1 = duo::MultiDevice::B1(<TestBackend1 as Backend>::Device::default());
        let device2 = duo::MultiDevice::B2(<TestBackend2 as Backend>::Device::default());
        let tensor1 = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0, 4.0], &device1);
        let tensor2 = Tensor::<TestBackend, 1>::from_floats([5.0, 6.0, 7.0, 8.0], &device2);

        let tensor1_2 = tensor1.clone().to_device(&device2);
        tensor1.into_data().assert_eq(&tensor1_2.into_data(), true);

        let tensor2_1 = tensor2.clone().to_device(&device1);
        tensor2.into_data().assert_eq(&tensor2_1.into_data(), true);
    }
}
