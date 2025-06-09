mod types {
    use alloc::sync::Arc;
    use burn_common::future::DynFut;
    use burn_ir::{BackendIr, OperationIr, TensorHandle, TensorId, TensorIr};
    use burn_tensor::{
        DType, Shape, TensorData, backend::{Backend, DeviceId, DeviceOps},
        try_read_sync,
    };
    use crate::{
        ByteBridge, DirectChannel, MultiBackendBridge, RouterTensor, Runner,
        RunnerChannel, RunnerClient,
    };
    /// Module containing the essential types for multi-backend operations.
    ///
    /// - `Handle`: the type used to point to a tensor (defined for all backends).
    /// - `MultiRunnerClient`: a client for multiple runners (each responsible to execute tensor operations on a given backend).
    /// - `DirectChannel`: a local channel with direct connection to the backend runner clients.
    /// - `ByteBridge`: a simple multi-backend bridge that transfers tensors via the underlying [tensor data](burn_tensor::TensorData).
    ///
    /// Each enum type is defined with backend identifiers as variant names (e.g., `B1` and `B2` for dual backends).
    pub mod duo {
        use super::*;
        /// The type that can be used to point to a tensor of any kind.
        /// Each backend has its own variant.
        pub enum Handle<B1: BackendIr, B2: BackendIr> {
            #[allow(missing_docs)]
            B1(B1::Handle),
            #[allow(missing_docs)]
            B2(B2::Handle),
        }
        /// The device type used by a backend.
        /// Each backend has its own variant.
        pub enum MultiDevice<B1: Backend, B2: Backend> {
            #[allow(missing_docs)]
            B1(B1::Device),
            #[allow(missing_docs)]
            B2(B2::Device),
        }
        #[automatically_derived]
        impl<
            B1: ::core::clone::Clone + Backend,
            B2: ::core::clone::Clone + Backend,
        > ::core::clone::Clone for MultiDevice<B1, B2>
        where
            B1::Device: ::core::clone::Clone,
            B2::Device: ::core::clone::Clone,
        {
            #[inline]
            fn clone(&self) -> MultiDevice<B1, B2> {
                match self {
                    MultiDevice::B1(__self_0) => {
                        MultiDevice::B1(::core::clone::Clone::clone(__self_0))
                    }
                    MultiDevice::B2(__self_0) => {
                        MultiDevice::B2(::core::clone::Clone::clone(__self_0))
                    }
                }
            }
        }
        #[automatically_derived]
        impl<
            B1: ::core::fmt::Debug + Backend,
            B2: ::core::fmt::Debug + Backend,
        > ::core::fmt::Debug for MultiDevice<B1, B2>
        where
            B1::Device: ::core::fmt::Debug,
            B2::Device: ::core::fmt::Debug,
        {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                match self {
                    MultiDevice::B1(__self_0) => {
                        ::core::fmt::Formatter::debug_tuple_field1_finish(
                            f,
                            "B1",
                            &__self_0,
                        )
                    }
                    MultiDevice::B2(__self_0) => {
                        ::core::fmt::Formatter::debug_tuple_field1_finish(
                            f,
                            "B2",
                            &__self_0,
                        )
                    }
                }
            }
        }
        impl<B1: Backend, B2: Backend> PartialEq for MultiDevice<B1, B2> {
            fn eq(&self, other: &Self) -> bool {
                match (self, other) {
                    (Self::B1(lhs), Self::B1(rhs)) => lhs == rhs,
                    (Self::B2(lhs), Self::B2(rhs)) => lhs == rhs,
                    _ => false,
                }
            }
        }
        impl<B1: Backend, B2: Backend> Default for MultiDevice<B1, B2> {
            fn default() -> Self {
                Self::B1(B1::Device::default())
            }
        }
        impl<B1: Backend, B2: Backend> DeviceOps for MultiDevice<B1, B2> {
            fn id(&self) -> DeviceId {
                match self {
                    Self::B1(device) => device.id(),
                    Self::B2(device) => device.id(),
                }
            }
        }
        /// A local client with multiple runners (each responsible to execute tensor operations on a given backend).
        pub enum MultiRunnerClient<B1: BackendIr, B2: BackendIr> {
            #[allow(missing_docs)]
            B1(Runner<B1>),
            #[allow(missing_docs)]
            B2(Runner<B2>),
        }
        #[automatically_derived]
        impl<
            B1: ::core::clone::Clone + BackendIr,
            B2: ::core::clone::Clone + BackendIr,
        > ::core::clone::Clone for MultiRunnerClient<B1, B2> {
            #[inline]
            fn clone(&self) -> MultiRunnerClient<B1, B2> {
                match self {
                    MultiRunnerClient::B1(__self_0) => {
                        MultiRunnerClient::B1(::core::clone::Clone::clone(__self_0))
                    }
                    MultiRunnerClient::B2(__self_0) => {
                        MultiRunnerClient::B2(::core::clone::Clone::clone(__self_0))
                    }
                }
            }
        }
        impl<B1: BackendIr, B2: BackendIr> RunnerClient for MultiRunnerClient<B1, B2> {
            type Device = MultiDevice<B1, B2>;
            fn register(&self, op: OperationIr) {
                match self {
                    Self::B1(runner) => runner.register(op),
                    Self::B2(runner) => runner.register(op),
                }
            }
            fn read_tensor(&self, tensor: TensorIr) -> DynFut<TensorData> {
                match self {
                    Self::B1(runner) => runner.read_tensor(tensor),
                    Self::B2(runner) => runner.read_tensor(tensor),
                }
            }
            fn register_tensor_data(&self, data: TensorData) -> RouterTensor<Self> {
                match self {
                    Self::B1(runner) => {
                        let desc = runner.register_tensor_data_desc(data);
                        RouterTensor::new(
                            Arc::new(desc.id),
                            desc.shape,
                            desc.dtype,
                            self.clone(),
                        )
                    }
                    Self::B2(runner) => {
                        let desc = runner.register_tensor_data_desc(data);
                        RouterTensor::new(
                            Arc::new(desc.id),
                            desc.shape,
                            desc.dtype,
                            self.clone(),
                        )
                    }
                }
            }
            fn register_empty_tensor(
                &self,
                shape: Vec<usize>,
                dtype: DType,
            ) -> RouterTensor<Self> {
                match self {
                    Self::B1(runner) => {
                        let desc = runner.register_empty_tensor_desc(shape, dtype);
                        RouterTensor::new(
                            Arc::new(desc.id),
                            desc.shape,
                            desc.dtype,
                            self.clone(),
                        )
                    }
                    Self::B2(runner) => {
                        let desc = runner.register_empty_tensor_desc(shape, dtype);
                        RouterTensor::new(
                            Arc::new(desc.id),
                            desc.shape,
                            desc.dtype,
                            self.clone(),
                        )
                    }
                }
            }
            fn register_float_tensor(
                &self,
                shape: Vec<usize>,
                dtype: burn_tensor::FloatDType,
            ) -> RouterTensor<Self> {
                match self {
                    Self::B1(runner) => {
                        let desc = runner.register_float_tensor_desc(shape, dtype);
                        RouterTensor::new(
                            Arc::new(desc.id),
                            desc.shape,
                            desc.dtype,
                            self.clone(),
                        )
                    }
                    Self::B2(runner) => {
                        let desc = runner.register_float_tensor_desc(shape, dtype);
                        RouterTensor::new(
                            Arc::new(desc.id),
                            desc.shape,
                            desc.dtype,
                            self.clone(),
                        )
                    }
                }
            }
            fn device(&self) -> Self::Device {
                match self {
                    Self::B1(runner) => MultiDevice::B1(runner.device()),
                    Self::B2(runner) => MultiDevice::B2(runner.device()),
                }
            }
            fn register_orphan(&self, id: &TensorId) {
                match self {
                    Self::B1(runner) => runner.register_orphan(id),
                    Self::B2(runner) => runner.register_orphan(id),
                }
            }
            fn sync(&self) {
                match self {
                    Self::B1(runner) => runner.sync(),
                    Self::B2(runner) => runner.sync(),
                };
            }
            fn seed(&self, seed: u64) {
                match self {
                    Self::B1(runner) => runner.seed(seed),
                    Self::B2(runner) => runner.seed(seed),
                }
            }
        }
        impl<B1: BackendIr, B2: BackendIr, Br> RunnerChannel
        for DirectChannel<(B1, B2), Br>
        where
            Br: MultiBackendBridge<
                TensorHandle = Handle<B1, B2>,
                Device = MultiDevice<B1, B2>,
            >,
        {
            type Device = Br::Device;
            type Bridge = Br;
            type FloatElem = B1::FloatElem;
            type IntElem = B1::IntElem;
            type BoolElem = B1::BoolElem;
            type Client = MultiRunnerClient<B1, B2>;
            fn init_client(device: &Self::Device) -> Self::Client {
                match device {
                    MultiDevice::B1(device) => {
                        MultiRunnerClient::B1(Runner::new(device.clone()))
                    }
                    MultiDevice::B2(device) => {
                        MultiRunnerClient::B2(Runner::new(device.clone()))
                    }
                }
            }
            fn get_tensor_handle(
                tensor: &TensorIr,
                client: &Self::Client,
            ) -> <Self::Bridge as MultiBackendBridge>::TensorHandle {
                match client {
                    MultiRunnerClient::B1(runner) => {
                        Handle::B1(runner.get_tensor_handle(tensor))
                    }
                    MultiRunnerClient::B2(runner) => {
                        Handle::B2(runner.get_tensor_handle(tensor))
                    }
                }
            }
            fn register_tensor(
                client: &Self::Client,
                handle: <Self::Bridge as MultiBackendBridge>::TensorHandle,
                shape: Vec<usize>,
                dtype: DType,
            ) -> RouterTensor<Self::Client> {
                match client {
                    MultiRunnerClient::B1(runner) => {
                        match handle {
                            Handle::B1(handle) => {
                                runner.register_tensor(handle, shape, dtype, client.clone())
                            }
                            _ => {
                                ::core::panicking::panic_fmt(
                                    format_args!(
                                        "internal error: entered unreachable code: {0}",
                                        format_args!(
                                            "Can\'t register tensor handle for another backend.",
                                        ),
                                    ),
                                );
                            }
                        }
                    }
                    MultiRunnerClient::B2(runner) => {
                        match handle {
                            Handle::B2(handle) => {
                                runner.register_tensor(handle, shape, dtype, client.clone())
                            }
                            _ => {
                                ::core::panicking::panic_fmt(
                                    format_args!(
                                        "internal error: entered unreachable code: {0}",
                                        format_args!(
                                            "Can\'t register tensor handle for another backend.",
                                        ),
                                    ),
                                );
                            }
                        }
                    }
                }
            }
            fn name(_device: &Self::Device) -> String {
                let mut name = ::alloc::__export::must_use({
                    let res = ::alloc::fmt::format(
                        format_args!(
                            "{0}",
                            B1::name(&<B1::Device as Default>::default()),
                        ),
                    );
                    res
                });
                name.push_str(
                    &::alloc::__export::must_use({
                        let res = ::alloc::fmt::format(
                            format_args!(
                                ", {0}",
                                B2::name(&<B2::Device as Default>::default()),
                            ),
                        );
                        res
                    }),
                );
                ::alloc::__export::must_use({
                    let res = ::alloc::fmt::format(format_args!("direct<({0})>", name));
                    res
                })
            }
        }
        impl<B1: BackendIr, B2: BackendIr> MultiBackendBridge for ByteBridge<(B1, B2)> {
            type TensorHandle = Handle<B1, B2>;
            type Device = MultiDevice<B1, B2>;
            fn change_backend_float(
                tensor: Self::TensorHandle,
                shape: Shape,
                target_device: &Self::Device,
            ) -> Self::TensorHandle {
                match (tensor, target_device) {
                    (Handle::B1(handle), MultiDevice::B1(device)) => {
                        let tensor = B1::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let tensor = B1::float_to_device(tensor, device);
                        let handle = B1::float_tensor_handle(tensor);
                        Handle::B1(handle)
                    }
                    (Handle::B1(handle), MultiDevice::B2(device)) => {
                        let tensor = B1::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B1::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B2::float_from_data(data, device);
                        let handle = B2::float_tensor_handle(tensor);
                        Handle::B2(handle)
                    }
                    (Handle::B2(handle), MultiDevice::B1(device)) => {
                        let tensor = B2::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B2::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B1::float_from_data(data, device);
                        let handle = B1::float_tensor_handle(tensor);
                        Handle::B1(handle)
                    }
                    (Handle::B2(handle), MultiDevice::B2(device)) => {
                        let tensor = B2::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let tensor = B2::float_to_device(tensor, device);
                        let handle = B2::float_tensor_handle(tensor);
                        Handle::B2(handle)
                    }
                }
            }
            fn change_backend_int(
                tensor: Self::TensorHandle,
                shape: Shape,
                target_device: &Self::Device,
            ) -> Self::TensorHandle {
                match (tensor, target_device) {
                    (Handle::B1(handle), MultiDevice::B1(device)) => {
                        let tensor = B1::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let tensor = B1::float_to_device(tensor, device);
                        let handle = B1::float_tensor_handle(tensor);
                        Handle::B1(handle)
                    }
                    (Handle::B1(handle), MultiDevice::B2(device)) => {
                        let tensor = B1::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B1::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B2::float_from_data(data, device);
                        let handle = B2::float_tensor_handle(tensor);
                        Handle::B2(handle)
                    }
                    (Handle::B2(handle), MultiDevice::B1(device)) => {
                        let tensor = B2::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B2::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B1::float_from_data(data, device);
                        let handle = B1::float_tensor_handle(tensor);
                        Handle::B1(handle)
                    }
                    (Handle::B2(handle), MultiDevice::B2(device)) => {
                        let tensor = B2::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let tensor = B2::float_to_device(tensor, device);
                        let handle = B2::float_tensor_handle(tensor);
                        Handle::B2(handle)
                    }
                }
            }
            fn change_backend_bool(
                tensor: Self::TensorHandle,
                shape: Shape,
                target_device: &Self::Device,
            ) -> Self::TensorHandle {
                match (tensor, target_device) {
                    (Handle::B1(handle), MultiDevice::B1(device)) => {
                        let tensor = B1::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let tensor = B1::float_to_device(tensor, device);
                        let handle = B1::float_tensor_handle(tensor);
                        Handle::B1(handle)
                    }
                    (Handle::B1(handle), MultiDevice::B2(device)) => {
                        let tensor = B1::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B1::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B2::float_from_data(data, device);
                        let handle = B2::float_tensor_handle(tensor);
                        Handle::B2(handle)
                    }
                    (Handle::B2(handle), MultiDevice::B1(device)) => {
                        let tensor = B2::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B2::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B1::float_from_data(data, device);
                        let handle = B1::float_tensor_handle(tensor);
                        Handle::B1(handle)
                    }
                    (Handle::B2(handle), MultiDevice::B2(device)) => {
                        let tensor = B2::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let tensor = B2::float_to_device(tensor, device);
                        let handle = B2::float_tensor_handle(tensor);
                        Handle::B2(handle)
                    }
                }
            }
        }
    }
    /// Module containing the essential types for multi-backend operations.
    ///
    /// - `Handle`: the type used to point to a tensor (defined for all backends).
    /// - `MultiRunnerClient`: a client for multiple runners (each responsible to execute tensor operations on a given backend).
    /// - `DirectChannel`: a local channel with direct connection to the backend runner clients.
    /// - `ByteBridge`: a simple multi-backend bridge that transfers tensors via the underlying [tensor data](burn_tensor::TensorData).
    ///
    /// Each enum type is defined with backend identifiers as variant names (e.g., `B1` and `B2` for dual backends).
    pub mod trio {
        use super::*;
        /// The type that can be used to point to a tensor of any kind.
        /// Each backend has its own variant.
        pub enum Handle<B1: BackendIr, B2: BackendIr, B3: BackendIr> {
            #[allow(missing_docs)]
            B1(B1::Handle),
            #[allow(missing_docs)]
            B2(B2::Handle),
            #[allow(missing_docs)]
            B3(B3::Handle),
        }
        /// The device type used by a backend.
        /// Each backend has its own variant.
        pub enum MultiDevice<B1: Backend, B2: Backend, B3: Backend> {
            #[allow(missing_docs)]
            B1(B1::Device),
            #[allow(missing_docs)]
            B2(B2::Device),
            #[allow(missing_docs)]
            B3(B3::Device),
        }
        #[automatically_derived]
        impl<
            B1: ::core::clone::Clone + Backend,
            B2: ::core::clone::Clone + Backend,
            B3: ::core::clone::Clone + Backend,
        > ::core::clone::Clone for MultiDevice<B1, B2, B3>
        where
            B1::Device: ::core::clone::Clone,
            B2::Device: ::core::clone::Clone,
            B3::Device: ::core::clone::Clone,
        {
            #[inline]
            fn clone(&self) -> MultiDevice<B1, B2, B3> {
                match self {
                    MultiDevice::B1(__self_0) => {
                        MultiDevice::B1(::core::clone::Clone::clone(__self_0))
                    }
                    MultiDevice::B2(__self_0) => {
                        MultiDevice::B2(::core::clone::Clone::clone(__self_0))
                    }
                    MultiDevice::B3(__self_0) => {
                        MultiDevice::B3(::core::clone::Clone::clone(__self_0))
                    }
                }
            }
        }
        #[automatically_derived]
        impl<
            B1: ::core::fmt::Debug + Backend,
            B2: ::core::fmt::Debug + Backend,
            B3: ::core::fmt::Debug + Backend,
        > ::core::fmt::Debug for MultiDevice<B1, B2, B3>
        where
            B1::Device: ::core::fmt::Debug,
            B2::Device: ::core::fmt::Debug,
            B3::Device: ::core::fmt::Debug,
        {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                match self {
                    MultiDevice::B1(__self_0) => {
                        ::core::fmt::Formatter::debug_tuple_field1_finish(
                            f,
                            "B1",
                            &__self_0,
                        )
                    }
                    MultiDevice::B2(__self_0) => {
                        ::core::fmt::Formatter::debug_tuple_field1_finish(
                            f,
                            "B2",
                            &__self_0,
                        )
                    }
                    MultiDevice::B3(__self_0) => {
                        ::core::fmt::Formatter::debug_tuple_field1_finish(
                            f,
                            "B3",
                            &__self_0,
                        )
                    }
                }
            }
        }
        impl<B1: Backend, B2: Backend, B3: Backend> PartialEq
        for MultiDevice<B1, B2, B3> {
            fn eq(&self, other: &Self) -> bool {
                match (self, other) {
                    (Self::B1(lhs), Self::B1(rhs)) => lhs == rhs,
                    (Self::B2(lhs), Self::B2(rhs)) => lhs == rhs,
                    (Self::B3(lhs), Self::B3(rhs)) => lhs == rhs,
                    _ => false,
                }
            }
        }
        impl<B1: Backend, B2: Backend, B3: Backend> Default for MultiDevice<B1, B2, B3> {
            fn default() -> Self {
                Self::B1(B1::Device::default())
            }
        }
        impl<B1: Backend, B2: Backend, B3: Backend> DeviceOps
        for MultiDevice<B1, B2, B3> {
            fn id(&self) -> DeviceId {
                match self {
                    Self::B1(device) => device.id(),
                    Self::B2(device) => device.id(),
                    Self::B3(device) => device.id(),
                }
            }
        }
        /// A local client with multiple runners (each responsible to execute tensor operations on a given backend).
        pub enum MultiRunnerClient<B1: BackendIr, B2: BackendIr, B3: BackendIr> {
            #[allow(missing_docs)]
            B1(Runner<B1>),
            #[allow(missing_docs)]
            B2(Runner<B2>),
            #[allow(missing_docs)]
            B3(Runner<B3>),
        }
        #[automatically_derived]
        impl<
            B1: ::core::clone::Clone + BackendIr,
            B2: ::core::clone::Clone + BackendIr,
            B3: ::core::clone::Clone + BackendIr,
        > ::core::clone::Clone for MultiRunnerClient<B1, B2, B3> {
            #[inline]
            fn clone(&self) -> MultiRunnerClient<B1, B2, B3> {
                match self {
                    MultiRunnerClient::B1(__self_0) => {
                        MultiRunnerClient::B1(::core::clone::Clone::clone(__self_0))
                    }
                    MultiRunnerClient::B2(__self_0) => {
                        MultiRunnerClient::B2(::core::clone::Clone::clone(__self_0))
                    }
                    MultiRunnerClient::B3(__self_0) => {
                        MultiRunnerClient::B3(::core::clone::Clone::clone(__self_0))
                    }
                }
            }
        }
        impl<B1: BackendIr, B2: BackendIr, B3: BackendIr> RunnerClient
        for MultiRunnerClient<B1, B2, B3> {
            type Device = MultiDevice<B1, B2, B3>;
            fn register(&self, op: OperationIr) {
                match self {
                    Self::B1(runner) => runner.register(op),
                    Self::B2(runner) => runner.register(op),
                    Self::B3(runner) => runner.register(op),
                }
            }
            fn read_tensor(&self, tensor: TensorIr) -> DynFut<TensorData> {
                match self {
                    Self::B1(runner) => runner.read_tensor(tensor),
                    Self::B2(runner) => runner.read_tensor(tensor),
                    Self::B3(runner) => runner.read_tensor(tensor),
                }
            }
            fn register_tensor_data(&self, data: TensorData) -> RouterTensor<Self> {
                match self {
                    Self::B1(runner) => {
                        let desc = runner.register_tensor_data_desc(data);
                        RouterTensor::new(
                            Arc::new(desc.id),
                            desc.shape,
                            desc.dtype,
                            self.clone(),
                        )
                    }
                    Self::B2(runner) => {
                        let desc = runner.register_tensor_data_desc(data);
                        RouterTensor::new(
                            Arc::new(desc.id),
                            desc.shape,
                            desc.dtype,
                            self.clone(),
                        )
                    }
                    Self::B3(runner) => {
                        let desc = runner.register_tensor_data_desc(data);
                        RouterTensor::new(
                            Arc::new(desc.id),
                            desc.shape,
                            desc.dtype,
                            self.clone(),
                        )
                    }
                }
            }
            fn register_empty_tensor(
                &self,
                shape: Vec<usize>,
                dtype: DType,
            ) -> RouterTensor<Self> {
                match self {
                    Self::B1(runner) => {
                        let desc = runner.register_empty_tensor_desc(shape, dtype);
                        RouterTensor::new(
                            Arc::new(desc.id),
                            desc.shape,
                            desc.dtype,
                            self.clone(),
                        )
                    }
                    Self::B2(runner) => {
                        let desc = runner.register_empty_tensor_desc(shape, dtype);
                        RouterTensor::new(
                            Arc::new(desc.id),
                            desc.shape,
                            desc.dtype,
                            self.clone(),
                        )
                    }
                    Self::B3(runner) => {
                        let desc = runner.register_empty_tensor_desc(shape, dtype);
                        RouterTensor::new(
                            Arc::new(desc.id),
                            desc.shape,
                            desc.dtype,
                            self.clone(),
                        )
                    }
                }
            }
            fn register_float_tensor(
                &self,
                shape: Vec<usize>,
                dtype: burn_tensor::FloatDType,
            ) -> RouterTensor<Self> {
                match self {
                    Self::B1(runner) => {
                        let desc = runner.register_float_tensor_desc(shape, dtype);
                        RouterTensor::new(
                            Arc::new(desc.id),
                            desc.shape,
                            desc.dtype,
                            self.clone(),
                        )
                    }
                    Self::B2(runner) => {
                        let desc = runner.register_float_tensor_desc(shape, dtype);
                        RouterTensor::new(
                            Arc::new(desc.id),
                            desc.shape,
                            desc.dtype,
                            self.clone(),
                        )
                    }
                    Self::B3(runner) => {
                        let desc = runner.register_float_tensor_desc(shape, dtype);
                        RouterTensor::new(
                            Arc::new(desc.id),
                            desc.shape,
                            desc.dtype,
                            self.clone(),
                        )
                    }
                }
            }
            fn device(&self) -> Self::Device {
                match self {
                    Self::B1(runner) => MultiDevice::B1(runner.device()),
                    Self::B2(runner) => MultiDevice::B2(runner.device()),
                    Self::B3(runner) => MultiDevice::B3(runner.device()),
                }
            }
            fn register_orphan(&self, id: &TensorId) {
                match self {
                    Self::B1(runner) => runner.register_orphan(id),
                    Self::B2(runner) => runner.register_orphan(id),
                    Self::B3(runner) => runner.register_orphan(id),
                }
            }
            fn sync(&self) {
                match self {
                    Self::B1(runner) => runner.sync(),
                    Self::B2(runner) => runner.sync(),
                    Self::B3(runner) => runner.sync(),
                };
            }
            fn seed(&self, seed: u64) {
                match self {
                    Self::B1(runner) => runner.seed(seed),
                    Self::B2(runner) => runner.seed(seed),
                    Self::B3(runner) => runner.seed(seed),
                }
            }
        }
        impl<B1: BackendIr, B2: BackendIr, B3: BackendIr, Br> RunnerChannel
        for DirectChannel<(B1, B2, B3), Br>
        where
            Br: MultiBackendBridge<
                TensorHandle = Handle<B1, B2, B3>,
                Device = MultiDevice<B1, B2, B3>,
            >,
        {
            type Device = Br::Device;
            type Bridge = Br;
            type FloatElem = B1::FloatElem;
            type IntElem = B1::IntElem;
            type BoolElem = B1::BoolElem;
            type Client = MultiRunnerClient<B1, B2, B3>;
            fn init_client(device: &Self::Device) -> Self::Client {
                match device {
                    MultiDevice::B1(device) => {
                        MultiRunnerClient::B1(Runner::new(device.clone()))
                    }
                    MultiDevice::B2(device) => {
                        MultiRunnerClient::B2(Runner::new(device.clone()))
                    }
                    MultiDevice::B3(device) => {
                        MultiRunnerClient::B3(Runner::new(device.clone()))
                    }
                }
            }
            fn get_tensor_handle(
                tensor: &TensorIr,
                client: &Self::Client,
            ) -> <Self::Bridge as MultiBackendBridge>::TensorHandle {
                match client {
                    MultiRunnerClient::B1(runner) => {
                        Handle::B1(runner.get_tensor_handle(tensor))
                    }
                    MultiRunnerClient::B2(runner) => {
                        Handle::B2(runner.get_tensor_handle(tensor))
                    }
                    MultiRunnerClient::B3(runner) => {
                        Handle::B3(runner.get_tensor_handle(tensor))
                    }
                }
            }
            fn register_tensor(
                client: &Self::Client,
                handle: <Self::Bridge as MultiBackendBridge>::TensorHandle,
                shape: Vec<usize>,
                dtype: DType,
            ) -> RouterTensor<Self::Client> {
                match client {
                    MultiRunnerClient::B1(runner) => {
                        match handle {
                            Handle::B1(handle) => {
                                runner.register_tensor(handle, shape, dtype, client.clone())
                            }
                            _ => {
                                ::core::panicking::panic_fmt(
                                    format_args!(
                                        "internal error: entered unreachable code: {0}",
                                        format_args!(
                                            "Can\'t register tensor handle for another backend.",
                                        ),
                                    ),
                                );
                            }
                        }
                    }
                    MultiRunnerClient::B2(runner) => {
                        match handle {
                            Handle::B2(handle) => {
                                runner.register_tensor(handle, shape, dtype, client.clone())
                            }
                            _ => {
                                ::core::panicking::panic_fmt(
                                    format_args!(
                                        "internal error: entered unreachable code: {0}",
                                        format_args!(
                                            "Can\'t register tensor handle for another backend.",
                                        ),
                                    ),
                                );
                            }
                        }
                    }
                    MultiRunnerClient::B3(runner) => {
                        match handle {
                            Handle::B3(handle) => {
                                runner.register_tensor(handle, shape, dtype, client.clone())
                            }
                            _ => {
                                ::core::panicking::panic_fmt(
                                    format_args!(
                                        "internal error: entered unreachable code: {0}",
                                        format_args!(
                                            "Can\'t register tensor handle for another backend.",
                                        ),
                                    ),
                                );
                            }
                        }
                    }
                }
            }
            fn name(_device: &Self::Device) -> String {
                let mut name = ::alloc::__export::must_use({
                    let res = ::alloc::fmt::format(
                        format_args!(
                            "{0}",
                            B1::name(&<B1::Device as Default>::default()),
                        ),
                    );
                    res
                });
                name.push_str(
                    &::alloc::__export::must_use({
                        let res = ::alloc::fmt::format(
                            format_args!(
                                ", {0}",
                                B2::name(&<B2::Device as Default>::default()),
                            ),
                        );
                        res
                    }),
                );
                name.push_str(
                    &::alloc::__export::must_use({
                        let res = ::alloc::fmt::format(
                            format_args!(
                                ", {0}",
                                B3::name(&<B3::Device as Default>::default()),
                            ),
                        );
                        res
                    }),
                );
                ::alloc::__export::must_use({
                    let res = ::alloc::fmt::format(format_args!("direct<({0})>", name));
                    res
                })
            }
        }
        impl<B1: BackendIr, B2: BackendIr, B3: BackendIr> MultiBackendBridge
        for ByteBridge<(B1, B2, B3)> {
            type TensorHandle = Handle<B1, B2, B3>;
            type Device = MultiDevice<B1, B2, B3>;
            fn change_backend_float(
                tensor: Self::TensorHandle,
                shape: Shape,
                target_device: &Self::Device,
            ) -> Self::TensorHandle {
                match (tensor, target_device) {
                    (Handle::B1(handle), MultiDevice::B1(device)) => {
                        let tensor = B1::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let tensor = B1::float_to_device(tensor, device);
                        let handle = B1::float_tensor_handle(tensor);
                        Handle::B1(handle)
                    }
                    (Handle::B1(handle), MultiDevice::B2(device)) => {
                        let tensor = B1::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B1::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B2::float_from_data(data, device);
                        let handle = B2::float_tensor_handle(tensor);
                        Handle::B2(handle)
                    }
                    (Handle::B2(handle), MultiDevice::B1(device)) => {
                        let tensor = B2::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B2::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B1::float_from_data(data, device);
                        let handle = B1::float_tensor_handle(tensor);
                        Handle::B1(handle)
                    }
                    (Handle::B2(handle), MultiDevice::B2(device)) => {
                        let tensor = B2::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let tensor = B2::float_to_device(tensor, device);
                        let handle = B2::float_tensor_handle(tensor);
                        Handle::B2(handle)
                    }
                    (Handle::B1(handle), MultiDevice::B3(device)) => {
                        let tensor = B1::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B1::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B3::float_from_data(data, device);
                        let handle = B3::float_tensor_handle(tensor);
                        Handle::B3(handle)
                    }
                    (Handle::B3(handle), MultiDevice::B1(device)) => {
                        let tensor = B3::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B3::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B1::float_from_data(data, device);
                        let handle = B1::float_tensor_handle(tensor);
                        Handle::B1(handle)
                    }
                    (Handle::B3(handle), MultiDevice::B3(device)) => {
                        let tensor = B3::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let tensor = B3::float_to_device(tensor, device);
                        let handle = B3::float_tensor_handle(tensor);
                        Handle::B3(handle)
                    }
                    (Handle::B2(handle), MultiDevice::B3(device)) => {
                        let tensor = B2::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B2::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B3::float_from_data(data, device);
                        let handle = B3::float_tensor_handle(tensor);
                        Handle::B3(handle)
                    }
                    (Handle::B3(handle), MultiDevice::B2(device)) => {
                        let tensor = B3::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B3::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B2::float_from_data(data, device);
                        let handle = B2::float_tensor_handle(tensor);
                        Handle::B2(handle)
                    }
                }
            }
            fn change_backend_int(
                tensor: Self::TensorHandle,
                shape: Shape,
                target_device: &Self::Device,
            ) -> Self::TensorHandle {
                match (tensor, target_device) {
                    (Handle::B1(handle), MultiDevice::B1(device)) => {
                        let tensor = B1::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let tensor = B1::float_to_device(tensor, device);
                        let handle = B1::float_tensor_handle(tensor);
                        Handle::B1(handle)
                    }
                    (Handle::B1(handle), MultiDevice::B2(device)) => {
                        let tensor = B1::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B1::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B2::float_from_data(data, device);
                        let handle = B2::float_tensor_handle(tensor);
                        Handle::B2(handle)
                    }
                    (Handle::B2(handle), MultiDevice::B1(device)) => {
                        let tensor = B2::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B2::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B1::float_from_data(data, device);
                        let handle = B1::float_tensor_handle(tensor);
                        Handle::B1(handle)
                    }
                    (Handle::B2(handle), MultiDevice::B2(device)) => {
                        let tensor = B2::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let tensor = B2::float_to_device(tensor, device);
                        let handle = B2::float_tensor_handle(tensor);
                        Handle::B2(handle)
                    }
                    (Handle::B1(handle), MultiDevice::B3(device)) => {
                        let tensor = B1::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B1::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B3::float_from_data(data, device);
                        let handle = B3::float_tensor_handle(tensor);
                        Handle::B3(handle)
                    }
                    (Handle::B3(handle), MultiDevice::B1(device)) => {
                        let tensor = B3::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B3::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B1::float_from_data(data, device);
                        let handle = B1::float_tensor_handle(tensor);
                        Handle::B1(handle)
                    }
                    (Handle::B3(handle), MultiDevice::B3(device)) => {
                        let tensor = B3::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let tensor = B3::float_to_device(tensor, device);
                        let handle = B3::float_tensor_handle(tensor);
                        Handle::B3(handle)
                    }
                    (Handle::B2(handle), MultiDevice::B3(device)) => {
                        let tensor = B2::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B2::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B3::float_from_data(data, device);
                        let handle = B3::float_tensor_handle(tensor);
                        Handle::B3(handle)
                    }
                    (Handle::B3(handle), MultiDevice::B2(device)) => {
                        let tensor = B3::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B3::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B2::float_from_data(data, device);
                        let handle = B2::float_tensor_handle(tensor);
                        Handle::B2(handle)
                    }
                }
            }
            fn change_backend_bool(
                tensor: Self::TensorHandle,
                shape: Shape,
                target_device: &Self::Device,
            ) -> Self::TensorHandle {
                match (tensor, target_device) {
                    (Handle::B1(handle), MultiDevice::B1(device)) => {
                        let tensor = B1::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let tensor = B1::float_to_device(tensor, device);
                        let handle = B1::float_tensor_handle(tensor);
                        Handle::B1(handle)
                    }
                    (Handle::B1(handle), MultiDevice::B2(device)) => {
                        let tensor = B1::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B1::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B2::float_from_data(data, device);
                        let handle = B2::float_tensor_handle(tensor);
                        Handle::B2(handle)
                    }
                    (Handle::B2(handle), MultiDevice::B1(device)) => {
                        let tensor = B2::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B2::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B1::float_from_data(data, device);
                        let handle = B1::float_tensor_handle(tensor);
                        Handle::B1(handle)
                    }
                    (Handle::B2(handle), MultiDevice::B2(device)) => {
                        let tensor = B2::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let tensor = B2::float_to_device(tensor, device);
                        let handle = B2::float_tensor_handle(tensor);
                        Handle::B2(handle)
                    }
                    (Handle::B1(handle), MultiDevice::B3(device)) => {
                        let tensor = B1::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B1::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B3::float_from_data(data, device);
                        let handle = B3::float_tensor_handle(tensor);
                        Handle::B3(handle)
                    }
                    (Handle::B3(handle), MultiDevice::B1(device)) => {
                        let tensor = B3::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B3::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B1::float_from_data(data, device);
                        let handle = B1::float_tensor_handle(tensor);
                        Handle::B1(handle)
                    }
                    (Handle::B3(handle), MultiDevice::B3(device)) => {
                        let tensor = B3::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let tensor = B3::float_to_device(tensor, device);
                        let handle = B3::float_tensor_handle(tensor);
                        Handle::B3(handle)
                    }
                    (Handle::B2(handle), MultiDevice::B3(device)) => {
                        let tensor = B2::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B2::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B3::float_from_data(data, device);
                        let handle = B3::float_tensor_handle(tensor);
                        Handle::B3(handle)
                    }
                    (Handle::B3(handle), MultiDevice::B2(device)) => {
                        let tensor = B3::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B3::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B2::float_from_data(data, device);
                        let handle = B2::float_tensor_handle(tensor);
                        Handle::B2(handle)
                    }
                }
            }
        }
    }
    /// Module containing the essential types for multi-backend operations.
    ///
    /// - `Handle`: the type used to point to a tensor (defined for all backends).
    /// - `MultiRunnerClient`: a client for multiple runners (each responsible to execute tensor operations on a given backend).
    /// - `DirectChannel`: a local channel with direct connection to the backend runner clients.
    /// - `ByteBridge`: a simple multi-backend bridge that transfers tensors via the underlying [tensor data](burn_tensor::TensorData).
    ///
    /// Each enum type is defined with backend identifiers as variant names (e.g., `B1` and `B2` for dual backends).
    pub mod quad {
        use super::*;
        /// The type that can be used to point to a tensor of any kind.
        /// Each backend has its own variant.
        pub enum Handle<B1: BackendIr, B2: BackendIr, B3: BackendIr, B4: BackendIr> {
            #[allow(missing_docs)]
            B1(B1::Handle),
            #[allow(missing_docs)]
            B2(B2::Handle),
            #[allow(missing_docs)]
            B3(B3::Handle),
            #[allow(missing_docs)]
            B4(B4::Handle),
        }
        /// The device type used by a backend.
        /// Each backend has its own variant.
        pub enum MultiDevice<B1: Backend, B2: Backend, B3: Backend, B4: Backend> {
            #[allow(missing_docs)]
            B1(B1::Device),
            #[allow(missing_docs)]
            B2(B2::Device),
            #[allow(missing_docs)]
            B3(B3::Device),
            #[allow(missing_docs)]
            B4(B4::Device),
        }
        #[automatically_derived]
        impl<
            B1: ::core::clone::Clone + Backend,
            B2: ::core::clone::Clone + Backend,
            B3: ::core::clone::Clone + Backend,
            B4: ::core::clone::Clone + Backend,
        > ::core::clone::Clone for MultiDevice<B1, B2, B3, B4>
        where
            B1::Device: ::core::clone::Clone,
            B2::Device: ::core::clone::Clone,
            B3::Device: ::core::clone::Clone,
            B4::Device: ::core::clone::Clone,
        {
            #[inline]
            fn clone(&self) -> MultiDevice<B1, B2, B3, B4> {
                match self {
                    MultiDevice::B1(__self_0) => {
                        MultiDevice::B1(::core::clone::Clone::clone(__self_0))
                    }
                    MultiDevice::B2(__self_0) => {
                        MultiDevice::B2(::core::clone::Clone::clone(__self_0))
                    }
                    MultiDevice::B3(__self_0) => {
                        MultiDevice::B3(::core::clone::Clone::clone(__self_0))
                    }
                    MultiDevice::B4(__self_0) => {
                        MultiDevice::B4(::core::clone::Clone::clone(__self_0))
                    }
                }
            }
        }
        #[automatically_derived]
        impl<
            B1: ::core::fmt::Debug + Backend,
            B2: ::core::fmt::Debug + Backend,
            B3: ::core::fmt::Debug + Backend,
            B4: ::core::fmt::Debug + Backend,
        > ::core::fmt::Debug for MultiDevice<B1, B2, B3, B4>
        where
            B1::Device: ::core::fmt::Debug,
            B2::Device: ::core::fmt::Debug,
            B3::Device: ::core::fmt::Debug,
            B4::Device: ::core::fmt::Debug,
        {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                match self {
                    MultiDevice::B1(__self_0) => {
                        ::core::fmt::Formatter::debug_tuple_field1_finish(
                            f,
                            "B1",
                            &__self_0,
                        )
                    }
                    MultiDevice::B2(__self_0) => {
                        ::core::fmt::Formatter::debug_tuple_field1_finish(
                            f,
                            "B2",
                            &__self_0,
                        )
                    }
                    MultiDevice::B3(__self_0) => {
                        ::core::fmt::Formatter::debug_tuple_field1_finish(
                            f,
                            "B3",
                            &__self_0,
                        )
                    }
                    MultiDevice::B4(__self_0) => {
                        ::core::fmt::Formatter::debug_tuple_field1_finish(
                            f,
                            "B4",
                            &__self_0,
                        )
                    }
                }
            }
        }
        impl<B1: Backend, B2: Backend, B3: Backend, B4: Backend> PartialEq
        for MultiDevice<B1, B2, B3, B4> {
            fn eq(&self, other: &Self) -> bool {
                match (self, other) {
                    (Self::B1(lhs), Self::B1(rhs)) => lhs == rhs,
                    (Self::B2(lhs), Self::B2(rhs)) => lhs == rhs,
                    (Self::B3(lhs), Self::B3(rhs)) => lhs == rhs,
                    (Self::B4(lhs), Self::B4(rhs)) => lhs == rhs,
                    _ => false,
                }
            }
        }
        impl<B1: Backend, B2: Backend, B3: Backend, B4: Backend> Default
        for MultiDevice<B1, B2, B3, B4> {
            fn default() -> Self {
                Self::B1(B1::Device::default())
            }
        }
        impl<B1: Backend, B2: Backend, B3: Backend, B4: Backend> DeviceOps
        for MultiDevice<B1, B2, B3, B4> {
            fn id(&self) -> DeviceId {
                match self {
                    Self::B1(device) => device.id(),
                    Self::B2(device) => device.id(),
                    Self::B3(device) => device.id(),
                    Self::B4(device) => device.id(),
                }
            }
        }
        /// A local client with multiple runners (each responsible to execute tensor operations on a given backend).
        pub enum MultiRunnerClient<
            B1: BackendIr,
            B2: BackendIr,
            B3: BackendIr,
            B4: BackendIr,
        > {
            #[allow(missing_docs)]
            B1(Runner<B1>),
            #[allow(missing_docs)]
            B2(Runner<B2>),
            #[allow(missing_docs)]
            B3(Runner<B3>),
            #[allow(missing_docs)]
            B4(Runner<B4>),
        }
        #[automatically_derived]
        impl<
            B1: ::core::clone::Clone + BackendIr,
            B2: ::core::clone::Clone + BackendIr,
            B3: ::core::clone::Clone + BackendIr,
            B4: ::core::clone::Clone + BackendIr,
        > ::core::clone::Clone for MultiRunnerClient<B1, B2, B3, B4> {
            #[inline]
            fn clone(&self) -> MultiRunnerClient<B1, B2, B3, B4> {
                match self {
                    MultiRunnerClient::B1(__self_0) => {
                        MultiRunnerClient::B1(::core::clone::Clone::clone(__self_0))
                    }
                    MultiRunnerClient::B2(__self_0) => {
                        MultiRunnerClient::B2(::core::clone::Clone::clone(__self_0))
                    }
                    MultiRunnerClient::B3(__self_0) => {
                        MultiRunnerClient::B3(::core::clone::Clone::clone(__self_0))
                    }
                    MultiRunnerClient::B4(__self_0) => {
                        MultiRunnerClient::B4(::core::clone::Clone::clone(__self_0))
                    }
                }
            }
        }
        impl<B1: BackendIr, B2: BackendIr, B3: BackendIr, B4: BackendIr> RunnerClient
        for MultiRunnerClient<B1, B2, B3, B4> {
            type Device = MultiDevice<B1, B2, B3, B4>;
            fn register(&self, op: OperationIr) {
                match self {
                    Self::B1(runner) => runner.register(op),
                    Self::B2(runner) => runner.register(op),
                    Self::B3(runner) => runner.register(op),
                    Self::B4(runner) => runner.register(op),
                }
            }
            fn read_tensor(&self, tensor: TensorIr) -> DynFut<TensorData> {
                match self {
                    Self::B1(runner) => runner.read_tensor(tensor),
                    Self::B2(runner) => runner.read_tensor(tensor),
                    Self::B3(runner) => runner.read_tensor(tensor),
                    Self::B4(runner) => runner.read_tensor(tensor),
                }
            }
            fn register_tensor_data(&self, data: TensorData) -> RouterTensor<Self> {
                match self {
                    Self::B1(runner) => {
                        let desc = runner.register_tensor_data_desc(data);
                        RouterTensor::new(
                            Arc::new(desc.id),
                            desc.shape,
                            desc.dtype,
                            self.clone(),
                        )
                    }
                    Self::B2(runner) => {
                        let desc = runner.register_tensor_data_desc(data);
                        RouterTensor::new(
                            Arc::new(desc.id),
                            desc.shape,
                            desc.dtype,
                            self.clone(),
                        )
                    }
                    Self::B3(runner) => {
                        let desc = runner.register_tensor_data_desc(data);
                        RouterTensor::new(
                            Arc::new(desc.id),
                            desc.shape,
                            desc.dtype,
                            self.clone(),
                        )
                    }
                    Self::B4(runner) => {
                        let desc = runner.register_tensor_data_desc(data);
                        RouterTensor::new(
                            Arc::new(desc.id),
                            desc.shape,
                            desc.dtype,
                            self.clone(),
                        )
                    }
                }
            }
            fn register_empty_tensor(
                &self,
                shape: Vec<usize>,
                dtype: DType,
            ) -> RouterTensor<Self> {
                match self {
                    Self::B1(runner) => {
                        let desc = runner.register_empty_tensor_desc(shape, dtype);
                        RouterTensor::new(
                            Arc::new(desc.id),
                            desc.shape,
                            desc.dtype,
                            self.clone(),
                        )
                    }
                    Self::B2(runner) => {
                        let desc = runner.register_empty_tensor_desc(shape, dtype);
                        RouterTensor::new(
                            Arc::new(desc.id),
                            desc.shape,
                            desc.dtype,
                            self.clone(),
                        )
                    }
                    Self::B3(runner) => {
                        let desc = runner.register_empty_tensor_desc(shape, dtype);
                        RouterTensor::new(
                            Arc::new(desc.id),
                            desc.shape,
                            desc.dtype,
                            self.clone(),
                        )
                    }
                    Self::B4(runner) => {
                        let desc = runner.register_empty_tensor_desc(shape, dtype);
                        RouterTensor::new(
                            Arc::new(desc.id),
                            desc.shape,
                            desc.dtype,
                            self.clone(),
                        )
                    }
                }
            }
            fn register_float_tensor(
                &self,
                shape: Vec<usize>,
                dtype: burn_tensor::FloatDType,
            ) -> RouterTensor<Self> {
                match self {
                    Self::B1(runner) => {
                        let desc = runner.register_float_tensor_desc(shape, dtype);
                        RouterTensor::new(
                            Arc::new(desc.id),
                            desc.shape,
                            desc.dtype,
                            self.clone(),
                        )
                    }
                    Self::B2(runner) => {
                        let desc = runner.register_float_tensor_desc(shape, dtype);
                        RouterTensor::new(
                            Arc::new(desc.id),
                            desc.shape,
                            desc.dtype,
                            self.clone(),
                        )
                    }
                    Self::B3(runner) => {
                        let desc = runner.register_float_tensor_desc(shape, dtype);
                        RouterTensor::new(
                            Arc::new(desc.id),
                            desc.shape,
                            desc.dtype,
                            self.clone(),
                        )
                    }
                    Self::B4(runner) => {
                        let desc = runner.register_float_tensor_desc(shape, dtype);
                        RouterTensor::new(
                            Arc::new(desc.id),
                            desc.shape,
                            desc.dtype,
                            self.clone(),
                        )
                    }
                }
            }
            fn device(&self) -> Self::Device {
                match self {
                    Self::B1(runner) => MultiDevice::B1(runner.device()),
                    Self::B2(runner) => MultiDevice::B2(runner.device()),
                    Self::B3(runner) => MultiDevice::B3(runner.device()),
                    Self::B4(runner) => MultiDevice::B4(runner.device()),
                }
            }
            fn register_orphan(&self, id: &TensorId) {
                match self {
                    Self::B1(runner) => runner.register_orphan(id),
                    Self::B2(runner) => runner.register_orphan(id),
                    Self::B3(runner) => runner.register_orphan(id),
                    Self::B4(runner) => runner.register_orphan(id),
                }
            }
            fn sync(&self) {
                match self {
                    Self::B1(runner) => runner.sync(),
                    Self::B2(runner) => runner.sync(),
                    Self::B3(runner) => runner.sync(),
                    Self::B4(runner) => runner.sync(),
                };
            }
            fn seed(&self, seed: u64) {
                match self {
                    Self::B1(runner) => runner.seed(seed),
                    Self::B2(runner) => runner.seed(seed),
                    Self::B3(runner) => runner.seed(seed),
                    Self::B4(runner) => runner.seed(seed),
                }
            }
        }
        impl<
            B1: BackendIr,
            B2: BackendIr,
            B3: BackendIr,
            B4: BackendIr,
            Br,
        > RunnerChannel for DirectChannel<(B1, B2, B3, B4), Br>
        where
            Br: MultiBackendBridge<
                TensorHandle = Handle<B1, B2, B3, B4>,
                Device = MultiDevice<B1, B2, B3, B4>,
            >,
        {
            type Device = Br::Device;
            type Bridge = Br;
            type FloatElem = B1::FloatElem;
            type IntElem = B1::IntElem;
            type BoolElem = B1::BoolElem;
            type Client = MultiRunnerClient<B1, B2, B3, B4>;
            fn init_client(device: &Self::Device) -> Self::Client {
                match device {
                    MultiDevice::B1(device) => {
                        MultiRunnerClient::B1(Runner::new(device.clone()))
                    }
                    MultiDevice::B2(device) => {
                        MultiRunnerClient::B2(Runner::new(device.clone()))
                    }
                    MultiDevice::B3(device) => {
                        MultiRunnerClient::B3(Runner::new(device.clone()))
                    }
                    MultiDevice::B4(device) => {
                        MultiRunnerClient::B4(Runner::new(device.clone()))
                    }
                }
            }
            fn get_tensor_handle(
                tensor: &TensorIr,
                client: &Self::Client,
            ) -> <Self::Bridge as MultiBackendBridge>::TensorHandle {
                match client {
                    MultiRunnerClient::B1(runner) => {
                        Handle::B1(runner.get_tensor_handle(tensor))
                    }
                    MultiRunnerClient::B2(runner) => {
                        Handle::B2(runner.get_tensor_handle(tensor))
                    }
                    MultiRunnerClient::B3(runner) => {
                        Handle::B3(runner.get_tensor_handle(tensor))
                    }
                    MultiRunnerClient::B4(runner) => {
                        Handle::B4(runner.get_tensor_handle(tensor))
                    }
                }
            }
            fn register_tensor(
                client: &Self::Client,
                handle: <Self::Bridge as MultiBackendBridge>::TensorHandle,
                shape: Vec<usize>,
                dtype: DType,
            ) -> RouterTensor<Self::Client> {
                match client {
                    MultiRunnerClient::B1(runner) => {
                        match handle {
                            Handle::B1(handle) => {
                                runner.register_tensor(handle, shape, dtype, client.clone())
                            }
                            _ => {
                                ::core::panicking::panic_fmt(
                                    format_args!(
                                        "internal error: entered unreachable code: {0}",
                                        format_args!(
                                            "Can\'t register tensor handle for another backend.",
                                        ),
                                    ),
                                );
                            }
                        }
                    }
                    MultiRunnerClient::B2(runner) => {
                        match handle {
                            Handle::B2(handle) => {
                                runner.register_tensor(handle, shape, dtype, client.clone())
                            }
                            _ => {
                                ::core::panicking::panic_fmt(
                                    format_args!(
                                        "internal error: entered unreachable code: {0}",
                                        format_args!(
                                            "Can\'t register tensor handle for another backend.",
                                        ),
                                    ),
                                );
                            }
                        }
                    }
                    MultiRunnerClient::B3(runner) => {
                        match handle {
                            Handle::B3(handle) => {
                                runner.register_tensor(handle, shape, dtype, client.clone())
                            }
                            _ => {
                                ::core::panicking::panic_fmt(
                                    format_args!(
                                        "internal error: entered unreachable code: {0}",
                                        format_args!(
                                            "Can\'t register tensor handle for another backend.",
                                        ),
                                    ),
                                );
                            }
                        }
                    }
                    MultiRunnerClient::B4(runner) => {
                        match handle {
                            Handle::B4(handle) => {
                                runner.register_tensor(handle, shape, dtype, client.clone())
                            }
                            _ => {
                                ::core::panicking::panic_fmt(
                                    format_args!(
                                        "internal error: entered unreachable code: {0}",
                                        format_args!(
                                            "Can\'t register tensor handle for another backend.",
                                        ),
                                    ),
                                );
                            }
                        }
                    }
                }
            }
            fn name(_device: &Self::Device) -> String {
                let mut name = ::alloc::__export::must_use({
                    let res = ::alloc::fmt::format(
                        format_args!(
                            "{0}",
                            B1::name(&<B1::Device as Default>::default()),
                        ),
                    );
                    res
                });
                name.push_str(
                    &::alloc::__export::must_use({
                        let res = ::alloc::fmt::format(
                            format_args!(
                                ", {0}",
                                B2::name(&<B2::Device as Default>::default()),
                            ),
                        );
                        res
                    }),
                );
                name.push_str(
                    &::alloc::__export::must_use({
                        let res = ::alloc::fmt::format(
                            format_args!(
                                ", {0}",
                                B3::name(&<B3::Device as Default>::default()),
                            ),
                        );
                        res
                    }),
                );
                name.push_str(
                    &::alloc::__export::must_use({
                        let res = ::alloc::fmt::format(
                            format_args!(
                                ", {0}",
                                B4::name(&<B4::Device as Default>::default()),
                            ),
                        );
                        res
                    }),
                );
                ::alloc::__export::must_use({
                    let res = ::alloc::fmt::format(format_args!("direct<({0})>", name));
                    res
                })
            }
        }
        impl<
            B1: BackendIr,
            B2: BackendIr,
            B3: BackendIr,
            B4: BackendIr,
        > MultiBackendBridge for ByteBridge<(B1, B2, B3, B4)> {
            type TensorHandle = Handle<B1, B2, B3, B4>;
            type Device = MultiDevice<B1, B2, B3, B4>;
            fn change_backend_float(
                tensor: Self::TensorHandle,
                shape: Shape,
                target_device: &Self::Device,
            ) -> Self::TensorHandle {
                match (tensor, target_device) {
                    (Handle::B1(handle), MultiDevice::B1(device)) => {
                        let tensor = B1::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let tensor = B1::float_to_device(tensor, device);
                        let handle = B1::float_tensor_handle(tensor);
                        Handle::B1(handle)
                    }
                    (Handle::B1(handle), MultiDevice::B2(device)) => {
                        let tensor = B1::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B1::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B2::float_from_data(data, device);
                        let handle = B2::float_tensor_handle(tensor);
                        Handle::B2(handle)
                    }
                    (Handle::B2(handle), MultiDevice::B1(device)) => {
                        let tensor = B2::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B2::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B1::float_from_data(data, device);
                        let handle = B1::float_tensor_handle(tensor);
                        Handle::B1(handle)
                    }
                    (Handle::B2(handle), MultiDevice::B2(device)) => {
                        let tensor = B2::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let tensor = B2::float_to_device(tensor, device);
                        let handle = B2::float_tensor_handle(tensor);
                        Handle::B2(handle)
                    }
                    (Handle::B1(handle), MultiDevice::B3(device)) => {
                        let tensor = B1::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B1::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B3::float_from_data(data, device);
                        let handle = B3::float_tensor_handle(tensor);
                        Handle::B3(handle)
                    }
                    (Handle::B3(handle), MultiDevice::B1(device)) => {
                        let tensor = B3::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B3::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B1::float_from_data(data, device);
                        let handle = B1::float_tensor_handle(tensor);
                        Handle::B1(handle)
                    }
                    (Handle::B3(handle), MultiDevice::B3(device)) => {
                        let tensor = B3::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let tensor = B3::float_to_device(tensor, device);
                        let handle = B3::float_tensor_handle(tensor);
                        Handle::B3(handle)
                    }
                    (Handle::B1(handle), MultiDevice::B4(device)) => {
                        let tensor = B1::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B1::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B4::float_from_data(data, device);
                        let handle = B4::float_tensor_handle(tensor);
                        Handle::B4(handle)
                    }
                    (Handle::B4(handle), MultiDevice::B1(device)) => {
                        let tensor = B4::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B4::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B1::float_from_data(data, device);
                        let handle = B1::float_tensor_handle(tensor);
                        Handle::B1(handle)
                    }
                    (Handle::B4(handle), MultiDevice::B4(device)) => {
                        let tensor = B4::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let tensor = B4::float_to_device(tensor, device);
                        let handle = B4::float_tensor_handle(tensor);
                        Handle::B4(handle)
                    }
                    (Handle::B2(handle), MultiDevice::B3(device)) => {
                        let tensor = B2::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B2::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B3::float_from_data(data, device);
                        let handle = B3::float_tensor_handle(tensor);
                        Handle::B3(handle)
                    }
                    (Handle::B3(handle), MultiDevice::B2(device)) => {
                        let tensor = B3::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B3::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B2::float_from_data(data, device);
                        let handle = B2::float_tensor_handle(tensor);
                        Handle::B2(handle)
                    }
                    (Handle::B2(handle), MultiDevice::B4(device)) => {
                        let tensor = B2::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B2::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B4::float_from_data(data, device);
                        let handle = B4::float_tensor_handle(tensor);
                        Handle::B4(handle)
                    }
                    (Handle::B4(handle), MultiDevice::B2(device)) => {
                        let tensor = B4::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B4::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B2::float_from_data(data, device);
                        let handle = B2::float_tensor_handle(tensor);
                        Handle::B2(handle)
                    }
                    (Handle::B3(handle), MultiDevice::B4(device)) => {
                        let tensor = B3::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B3::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B4::float_from_data(data, device);
                        let handle = B4::float_tensor_handle(tensor);
                        Handle::B4(handle)
                    }
                    (Handle::B4(handle), MultiDevice::B3(device)) => {
                        let tensor = B4::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B4::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B3::float_from_data(data, device);
                        let handle = B3::float_tensor_handle(tensor);
                        Handle::B3(handle)
                    }
                }
            }
            fn change_backend_int(
                tensor: Self::TensorHandle,
                shape: Shape,
                target_device: &Self::Device,
            ) -> Self::TensorHandle {
                match (tensor, target_device) {
                    (Handle::B1(handle), MultiDevice::B1(device)) => {
                        let tensor = B1::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let tensor = B1::float_to_device(tensor, device);
                        let handle = B1::float_tensor_handle(tensor);
                        Handle::B1(handle)
                    }
                    (Handle::B1(handle), MultiDevice::B2(device)) => {
                        let tensor = B1::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B1::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B2::float_from_data(data, device);
                        let handle = B2::float_tensor_handle(tensor);
                        Handle::B2(handle)
                    }
                    (Handle::B2(handle), MultiDevice::B1(device)) => {
                        let tensor = B2::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B2::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B1::float_from_data(data, device);
                        let handle = B1::float_tensor_handle(tensor);
                        Handle::B1(handle)
                    }
                    (Handle::B2(handle), MultiDevice::B2(device)) => {
                        let tensor = B2::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let tensor = B2::float_to_device(tensor, device);
                        let handle = B2::float_tensor_handle(tensor);
                        Handle::B2(handle)
                    }
                    (Handle::B1(handle), MultiDevice::B3(device)) => {
                        let tensor = B1::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B1::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B3::float_from_data(data, device);
                        let handle = B3::float_tensor_handle(tensor);
                        Handle::B3(handle)
                    }
                    (Handle::B3(handle), MultiDevice::B1(device)) => {
                        let tensor = B3::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B3::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B1::float_from_data(data, device);
                        let handle = B1::float_tensor_handle(tensor);
                        Handle::B1(handle)
                    }
                    (Handle::B3(handle), MultiDevice::B3(device)) => {
                        let tensor = B3::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let tensor = B3::float_to_device(tensor, device);
                        let handle = B3::float_tensor_handle(tensor);
                        Handle::B3(handle)
                    }
                    (Handle::B1(handle), MultiDevice::B4(device)) => {
                        let tensor = B1::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B1::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B4::float_from_data(data, device);
                        let handle = B4::float_tensor_handle(tensor);
                        Handle::B4(handle)
                    }
                    (Handle::B4(handle), MultiDevice::B1(device)) => {
                        let tensor = B4::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B4::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B1::float_from_data(data, device);
                        let handle = B1::float_tensor_handle(tensor);
                        Handle::B1(handle)
                    }
                    (Handle::B4(handle), MultiDevice::B4(device)) => {
                        let tensor = B4::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let tensor = B4::float_to_device(tensor, device);
                        let handle = B4::float_tensor_handle(tensor);
                        Handle::B4(handle)
                    }
                    (Handle::B2(handle), MultiDevice::B3(device)) => {
                        let tensor = B2::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B2::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B3::float_from_data(data, device);
                        let handle = B3::float_tensor_handle(tensor);
                        Handle::B3(handle)
                    }
                    (Handle::B3(handle), MultiDevice::B2(device)) => {
                        let tensor = B3::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B3::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B2::float_from_data(data, device);
                        let handle = B2::float_tensor_handle(tensor);
                        Handle::B2(handle)
                    }
                    (Handle::B2(handle), MultiDevice::B4(device)) => {
                        let tensor = B2::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B2::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B4::float_from_data(data, device);
                        let handle = B4::float_tensor_handle(tensor);
                        Handle::B4(handle)
                    }
                    (Handle::B4(handle), MultiDevice::B2(device)) => {
                        let tensor = B4::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B4::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B2::float_from_data(data, device);
                        let handle = B2::float_tensor_handle(tensor);
                        Handle::B2(handle)
                    }
                    (Handle::B3(handle), MultiDevice::B4(device)) => {
                        let tensor = B3::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B3::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B4::float_from_data(data, device);
                        let handle = B4::float_tensor_handle(tensor);
                        Handle::B4(handle)
                    }
                    (Handle::B4(handle), MultiDevice::B3(device)) => {
                        let tensor = B4::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B4::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B3::float_from_data(data, device);
                        let handle = B3::float_tensor_handle(tensor);
                        Handle::B3(handle)
                    }
                }
            }
            fn change_backend_bool(
                tensor: Self::TensorHandle,
                shape: Shape,
                target_device: &Self::Device,
            ) -> Self::TensorHandle {
                match (tensor, target_device) {
                    (Handle::B1(handle), MultiDevice::B1(device)) => {
                        let tensor = B1::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let tensor = B1::float_to_device(tensor, device);
                        let handle = B1::float_tensor_handle(tensor);
                        Handle::B1(handle)
                    }
                    (Handle::B1(handle), MultiDevice::B2(device)) => {
                        let tensor = B1::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B1::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B2::float_from_data(data, device);
                        let handle = B2::float_tensor_handle(tensor);
                        Handle::B2(handle)
                    }
                    (Handle::B2(handle), MultiDevice::B1(device)) => {
                        let tensor = B2::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B2::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B1::float_from_data(data, device);
                        let handle = B1::float_tensor_handle(tensor);
                        Handle::B1(handle)
                    }
                    (Handle::B2(handle), MultiDevice::B2(device)) => {
                        let tensor = B2::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let tensor = B2::float_to_device(tensor, device);
                        let handle = B2::float_tensor_handle(tensor);
                        Handle::B2(handle)
                    }
                    (Handle::B1(handle), MultiDevice::B3(device)) => {
                        let tensor = B1::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B1::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B3::float_from_data(data, device);
                        let handle = B3::float_tensor_handle(tensor);
                        Handle::B3(handle)
                    }
                    (Handle::B3(handle), MultiDevice::B1(device)) => {
                        let tensor = B3::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B3::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B1::float_from_data(data, device);
                        let handle = B1::float_tensor_handle(tensor);
                        Handle::B1(handle)
                    }
                    (Handle::B3(handle), MultiDevice::B3(device)) => {
                        let tensor = B3::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let tensor = B3::float_to_device(tensor, device);
                        let handle = B3::float_tensor_handle(tensor);
                        Handle::B3(handle)
                    }
                    (Handle::B1(handle), MultiDevice::B4(device)) => {
                        let tensor = B1::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B1::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B4::float_from_data(data, device);
                        let handle = B4::float_tensor_handle(tensor);
                        Handle::B4(handle)
                    }
                    (Handle::B4(handle), MultiDevice::B1(device)) => {
                        let tensor = B4::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B4::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B1::float_from_data(data, device);
                        let handle = B1::float_tensor_handle(tensor);
                        Handle::B1(handle)
                    }
                    (Handle::B4(handle), MultiDevice::B4(device)) => {
                        let tensor = B4::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let tensor = B4::float_to_device(tensor, device);
                        let handle = B4::float_tensor_handle(tensor);
                        Handle::B4(handle)
                    }
                    (Handle::B2(handle), MultiDevice::B3(device)) => {
                        let tensor = B2::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B2::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B3::float_from_data(data, device);
                        let handle = B3::float_tensor_handle(tensor);
                        Handle::B3(handle)
                    }
                    (Handle::B3(handle), MultiDevice::B2(device)) => {
                        let tensor = B3::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B3::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B2::float_from_data(data, device);
                        let handle = B2::float_tensor_handle(tensor);
                        Handle::B2(handle)
                    }
                    (Handle::B2(handle), MultiDevice::B4(device)) => {
                        let tensor = B2::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B2::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B4::float_from_data(data, device);
                        let handle = B4::float_tensor_handle(tensor);
                        Handle::B4(handle)
                    }
                    (Handle::B4(handle), MultiDevice::B2(device)) => {
                        let tensor = B4::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B4::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B2::float_from_data(data, device);
                        let handle = B2::float_tensor_handle(tensor);
                        Handle::B2(handle)
                    }
                    (Handle::B3(handle), MultiDevice::B4(device)) => {
                        let tensor = B3::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B3::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B4::float_from_data(data, device);
                        let handle = B4::float_tensor_handle(tensor);
                        Handle::B4(handle)
                    }
                    (Handle::B4(handle), MultiDevice::B3(device)) => {
                        let tensor = B4::float_tensor(TensorHandle {
                            handle: handle,
                            shape: shape,
                        });
                        let data = try_read_sync(B4::float_into_data(tensor))
                            .expect(
                                "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM.",
                            );
                        let tensor = B3::float_from_data(data, device);
                        let handle = B3::float_tensor_handle(tensor);
                        Handle::B3(handle)
                    }
                }
            }
        }
    }
}
