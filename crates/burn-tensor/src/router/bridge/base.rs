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

    fn change_backend_float(
        tensor: Self::TensorHandle,
        shape: Shape,
        device: &Self::Device,
    ) -> Self::TensorHandle;
}

// TODO: generate this for different number of backends (up to 4?)

/// [`MultiBackendBridge`] handle type for two backends.
pub enum TensorHandle2<B1: ReprBackend, B2: ReprBackend> {
    Handle1(B1::Handle),
    Handle2(B2::Handle),
}

/// [`MultiBackendBridge`] device type for two backends.
#[derive(Clone, Debug)]
pub enum MultiDevice2<B1: Backend, B2: Backend> {
    Device1(B1::Device),
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
    fn id(&self) -> crate::backend::DeviceId {
        match self {
            MultiDevice2::Device1(device) => device.id(),
            MultiDevice2::Device2(device) => device.id(),
        }
    }
}

/// Local [`RunnerClient`] with two backends.
#[derive(Clone)]
pub enum MultiRunnerClient2<B1: ReprBackend, B2: ReprBackend> {
    RunnerClient1(Runner<B1>),
    RunnerClient2(Runner<B2>),
}

impl<B1: ReprBackend, B2: ReprBackend> RunnerClient for MultiRunnerClient2<B1, B2> {
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

    fn write_tensor(&self, data: TensorData) -> RouterTensor<Self> {
        match self {
            MultiRunnerClient2::RunnerClient1(runner) => {
                let tensor = runner.write_tensor(data);
                RouterTensor {
                    id: tensor.id,
                    shape: tensor.shape,
                    dtype: tensor.dtype,
                    client: MultiRunnerClient2::RunnerClient1(tensor.client),
                }
            }
            MultiRunnerClient2::RunnerClient2(runner) => {
                let tensor = runner.write_tensor(data);
                RouterTensor {
                    id: tensor.id,
                    shape: tensor.shape,
                    dtype: tensor.dtype,
                    client: MultiRunnerClient2::RunnerClient2(tensor.client),
                }
            }
        }
    }

    fn register_new_tensor(&self, shape: Vec<usize>, dtype: DType) -> RouterTensor<Self> {
        match self {
            MultiRunnerClient2::RunnerClient1(runner) => {
                let tensor = runner.register_new_tensor(shape, dtype);
                RouterTensor {
                    id: tensor.id,
                    shape: tensor.shape,
                    dtype: tensor.dtype,
                    client: MultiRunnerClient2::RunnerClient1(tensor.client),
                }
            }
            MultiRunnerClient2::RunnerClient2(runner) => {
                let tensor = runner.register_new_tensor(shape, dtype);
                RouterTensor {
                    id: tensor.id,
                    shape: tensor.shape,
                    dtype: tensor.dtype,
                    client: MultiRunnerClient2::RunnerClient2(tensor.client),
                }
            }
        }
    }

    fn device(&self) -> Self::Device {
        match self {
            MultiRunnerClient2::RunnerClient1(runner) => {
                MultiDevice2::Device1(runner.device().clone())
            }
            MultiRunnerClient2::RunnerClient2(runner) => {
                MultiDevice2::Device2(runner.device().clone())
            }
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
