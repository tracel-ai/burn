use core::marker::PhantomData;

use crate::{
    repr::{QuantizedTensorDescription, ReprBackend, TensorDescription},
    router::{
        MultiBackendBridge, MultiDevice2, MultiRunnerClient2, RouterTensor, Runner, TensorHandle2,
    },
};

use super::{RunnerChannel, TensorHandle};

pub struct DirectChannel<Backends, Bridge> {
    backends: PhantomData<Backends>,
    bridge: PhantomData<Bridge>,
}

impl<Backends, Bridge> Clone for DirectChannel<Backends, Bridge> {
    fn clone(&self) -> Self {
        Self {
            backends: self.backends.clone(),
            bridge: self.bridge.clone(),
        }
    }
}

impl<B1, B2, Br> RunnerChannel for DirectChannel<(B1, B2), Br>
where
    B1: ReprBackend,
    B2: ReprBackend,
    Br: MultiBackendBridge<TensorHandle = TensorHandle2<B1, B2>, Device = MultiDevice2<B1, B2>>,
{
    type Device = Br::Device;

    type Bridge = Br;

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

    fn get_float_tensor(
        tensor: &TensorDescription,
        client: &Self::Client,
    ) -> TensorHandle<Self::Bridge> {
        match client {
            MultiRunnerClient2::RunnerClient1(runner) => {
                TensorHandle2::Handle1(runner.get_float_tensor(tensor))
            }
            MultiRunnerClient2::RunnerClient2(runner) => {
                TensorHandle2::Handle2(runner.get_float_tensor(tensor))
            }
        }
    }

    fn get_int_tensor(
        tensor: &TensorDescription,
        client: &Self::Client,
    ) -> TensorHandle<Self::Bridge> {
        todo!()
    }

    fn get_bool_tensor(
        tensor: &TensorDescription,
        client: &Self::Client,
    ) -> TensorHandle<Self::Bridge> {
        todo!()
    }

    fn get_quantized_tensor(
        tensor: &QuantizedTensorDescription,
        client: &Self::Client,
    ) -> TensorHandle<Self::Bridge> {
        todo!()
    }

    fn register_tensor(
        client: &Self::Client,
        handle: TensorHandle<Self::Bridge>,
        shape: Vec<usize>,
        dtype: crate::DType,
    ) -> RouterTensor<Self::Client> {
        match client {
            MultiRunnerClient2::RunnerClient1(runner) => match handle {
                TensorHandle2::Handle1(handle) => {
                    runner.register_tensor(handle, shape, dtype, client.clone())
                }
                TensorHandle2::Handle2(_) => unreachable!(),
            },
            MultiRunnerClient2::RunnerClient2(runner) => match handle {
                TensorHandle2::Handle1(_) => unreachable!(),
                TensorHandle2::Handle2(handle) => {
                    runner.register_tensor(handle, shape, dtype, client.clone())
                }
            },
        }
    }
}
