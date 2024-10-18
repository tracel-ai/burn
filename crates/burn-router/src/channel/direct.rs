use alloc::{format, string::String, sync::Arc, vec::Vec};
use core::marker::PhantomData;

use burn_tensor::{
    backend::{Backend, BackendBridge, DeviceId, DeviceOps},
    repr::{OperationDescription, ReprBackend, TensorDescription, TensorId},
    DType, TensorData,
};

use super::{RunnerChannel, TensorHandle};
use crate::{MultiBackendBridge, RouterTensor, Runner, RunnerClient};

/// A local channel with direct connection to the backend runner clients.
pub struct DirectChannel<Backends, Bridge> {
    backends: PhantomData<Backends>,
    bridge: PhantomData<Bridge>,
}

impl<Backends, Bridge> Clone for DirectChannel<Backends, Bridge> {
    fn clone(&self) -> Self {
        Self {
            backends: self.backends,
            bridge: self.bridge,
        }
    }
}

impl<B1, B2, Br> RunnerChannel for DirectChannel<(B1, B2), Br>
where
    B1: ReprBackend,
    B2: ReprBackend<FloatElem = B1::FloatElem, IntElem = B1::IntElem>,
    Br: MultiBackendBridge<TensorHandle = TensorHandle2<B1, B2>, Device = MultiDevice2<B1, B2>>,
    // Restrict full precision backend handle to be the same
    <<B1 as Backend>::FullPrecisionBridge as BackendBridge<B1>>::Target:
        ReprBackend<Handle = B1::Handle>,
    <<B2 as Backend>::FullPrecisionBridge as BackendBridge<B2>>::Target:
        ReprBackend<Handle = B2::Handle>,
{
    type Device = Br::Device;

    type Bridge = Br;

    type FloatElem = B1::FloatElem;
    type IntElem = B1::IntElem;

    type Client = MultiRunnerClient2<B1, B2>;

    fn init_client(device: &Self::Device) -> Self::Client {
        match device {
            MultiDevice2::Device1(device) => {
                MultiRunnerClient2::RunnerClient1(Runner::new(device.clone()))
            }
            MultiDevice2::Device2(device) => {
                MultiRunnerClient2::RunnerClient2(Runner::new(device.clone()))
            }
        }
    }

    fn get_tensor_handle(
        tensor: &TensorDescription,
        client: &Self::Client,
    ) -> TensorHandle<Self::Bridge> {
        match client {
            MultiRunnerClient2::RunnerClient1(runner) => {
                TensorHandle2::Handle1(runner.get_tensor_handle(tensor))
            }
            MultiRunnerClient2::RunnerClient2(runner) => {
                TensorHandle2::Handle2(runner.get_tensor_handle(tensor))
            }
        }
    }

    fn register_tensor(
        client: &Self::Client,
        handle: TensorHandle<Self::Bridge>,
        shape: Vec<usize>,
        dtype: DType,
    ) -> RouterTensor<Self::Client> {
        match client {
            MultiRunnerClient2::RunnerClient1(runner) => match handle {
                TensorHandle2::Handle1(handle) => {
                    runner.register_tensor(handle, shape, dtype, client.clone())
                }
                TensorHandle2::Handle2(_) => {
                    unreachable!("Can't register tensor handle for another backend.")
                }
            },
            MultiRunnerClient2::RunnerClient2(runner) => match handle {
                TensorHandle2::Handle1(_) => {
                    unreachable!("Can't register tensor handle for another backend.")
                }
                TensorHandle2::Handle2(handle) => {
                    runner.register_tensor(handle, shape, dtype, client.clone())
                }
            },
        }
    }

    fn name() -> String {
        format!("direct<({}, {})>", B1::name(), B2::name())
    }
}

// TODO: generate this for different number of backends (up to 4?)

/// Handle type to interact with two backends.
pub enum TensorHandle2<B1: ReprBackend, B2: ReprBackend> {
    /// Handle for the first backend.
    Handle1(B1::Handle),
    /// Handle for the second backend.
    Handle2(B2::Handle),
}

/// Device type to interact with two backends.
#[derive(Clone, Debug)]
pub enum MultiDevice2<B1: Backend, B2: Backend> {
    /// Device for the first backend.
    Device1(B1::Device),
    /// Device for the second backend.
    Device2(B2::Device),
}

impl<B1: Backend, B2: Backend> PartialEq for MultiDevice2<B1, B2> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Device1(lhs), Self::Device1(rhs)) => lhs == rhs,
            (Self::Device2(lhs), Self::Device2(rhs)) => lhs == rhs,
            _ => false,
        }
    }
}

impl<B1: Backend, B2: Backend> Eq for MultiDevice2<B1, B2> {}

impl<B1: Backend, B2: Backend> Default for MultiDevice2<B1, B2> {
    fn default() -> Self {
        Self::Device1(B1::Device::default())
    }
}

impl<B1: Backend, B2: Backend> DeviceOps for MultiDevice2<B1, B2> {
    fn id(&self) -> DeviceId {
        match self {
            MultiDevice2::Device1(device) => device.id(),
            MultiDevice2::Device2(device) => device.id(),
        }
    }
}

/// Local [`RunnerClient`] with two backends.
#[derive(Clone)]
pub enum MultiRunnerClient2<B1: ReprBackend, B2: ReprBackend> {
    /// Client for the first backend runner.
    RunnerClient1(Runner<B1>),
    /// Client for the second backend runner.
    RunnerClient2(Runner<B2>),
}

impl<B1: ReprBackend, B2: ReprBackend> RunnerClient for MultiRunnerClient2<B1, B2>
where
    <<B1 as Backend>::FullPrecisionBridge as BackendBridge<B1>>::Target:
        ReprBackend<Handle = B1::Handle>,
    <<B2 as Backend>::FullPrecisionBridge as BackendBridge<B2>>::Target:
        ReprBackend<Handle = B2::Handle>,
{
    type Device = MultiDevice2<B1, B2>;

    fn register(&self, op: OperationDescription) {
        match self {
            MultiRunnerClient2::RunnerClient1(runner) => runner.register(op),
            MultiRunnerClient2::RunnerClient2(runner) => runner.register(op),
        }
    }

    async fn read_tensor(&self, tensor: TensorDescription) -> TensorData {
        match self {
            MultiRunnerClient2::RunnerClient1(runner) => runner.read_tensor(tensor).await,
            MultiRunnerClient2::RunnerClient2(runner) => runner.read_tensor(tensor).await,
        }
    }

    fn register_tensor_data(&self, data: TensorData) -> RouterTensor<Self> {
        match self {
            MultiRunnerClient2::RunnerClient1(runner) => {
                let desc = runner.register_tensor_data_desc(data);
                RouterTensor::new(Arc::new(desc.id), desc.shape, desc.dtype, self.clone())
            }
            MultiRunnerClient2::RunnerClient2(runner) => {
                let desc = runner.register_tensor_data_desc(data);
                RouterTensor::new(Arc::new(desc.id), desc.shape, desc.dtype, self.clone())
            }
        }
    }

    fn register_empty_tensor(&self, shape: Vec<usize>, dtype: DType) -> RouterTensor<Self> {
        match self {
            MultiRunnerClient2::RunnerClient1(runner) => {
                let desc = runner.register_empty_tensor_desc(shape, dtype);
                RouterTensor::new(Arc::new(desc.id), desc.shape, desc.dtype, self.clone())
            }
            MultiRunnerClient2::RunnerClient2(runner) => {
                let desc = runner.register_empty_tensor_desc(shape, dtype);
                RouterTensor::new(Arc::new(desc.id), desc.shape, desc.dtype, self.clone())
            }
        }
    }

    fn register_float_tensor(&self, shape: Vec<usize>, full_precision: bool) -> RouterTensor<Self> {
        match self {
            MultiRunnerClient2::RunnerClient1(runner) => {
                let desc = runner.register_float_tensor_desc(shape, full_precision);
                RouterTensor::new(Arc::new(desc.id), desc.shape, desc.dtype, self.clone())
            }
            MultiRunnerClient2::RunnerClient2(runner) => {
                let desc = runner.register_float_tensor_desc(shape, full_precision);
                RouterTensor::new(Arc::new(desc.id), desc.shape, desc.dtype, self.clone())
            }
        }
    }

    fn device(&self) -> Self::Device {
        match self {
            MultiRunnerClient2::RunnerClient1(runner) => MultiDevice2::Device1(runner.device()),
            MultiRunnerClient2::RunnerClient2(runner) => MultiDevice2::Device2(runner.device()),
        }
    }

    fn register_orphan(&self, id: &TensorId) {
        match self {
            MultiRunnerClient2::RunnerClient1(runner) => runner.register_orphan(id),
            MultiRunnerClient2::RunnerClient2(runner) => runner.register_orphan(id),
        }
    }

    fn sync(&self) {
        match self {
            MultiRunnerClient2::RunnerClient1(runner) => runner.sync(),
            MultiRunnerClient2::RunnerClient2(runner) => runner.sync(),
        }
    }

    fn seed(&self, seed: u64) {
        match self {
            MultiRunnerClient2::RunnerClient1(runner) => runner.seed(seed),
            MultiRunnerClient2::RunnerClient2(runner) => runner.seed(seed),
        }
    }
}

// NOTE: conflicting implementations because B1 and B2 cannot be differentiated (could be the same type)
// impl<B1: ReprBackend, B2: ReprBackend> From<RouterTensor<Runner<B1>>>
//     for RouterTensor<MultiRunnerClient2<B1, B2>>
// {
//     fn from(value: RouterTensor<Runner<B1>>) -> Self {
//         RouterTensor {
//             desc: value.desc,
//             client: MultiRunnerClient2::RunnerClient1(value.client),
//         }
//     }
// }

// impl<B1: ReprBackend, B2: ReprBackend> From<RouterTensor<Runner<B2>>>
//     for RouterTensor<MultiRunnerClient2<B1, B2>>
// {
//     fn from(value: RouterTensor<Runner<B2>>) -> Self {
//         RouterTensor {
//             desc: value.desc,
//             client: MultiRunnerClient2::RunnerClient2(value.client),
//         }
//     }
// }
