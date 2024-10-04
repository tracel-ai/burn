use core::marker::PhantomData;

use crate::{
    repr::{ReprBackend, TensorDescription},
    router::{
        Handle2, MultiBackendBridge, MultiDevice2, MultiRunnerClient2, RouterTensor, Runner,
        TensorHandle2,
    },
};

use super::RunnerChannel;

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

impl<
        B1: ReprBackend,
        B2: ReprBackend,
        // Br: MultiBackendBridge<TensorType = Handle2<B1, B2>, Device = MultiDevice2<B1, B2>>,
        Br: MultiBackendBridge<TensorType = TensorHandle2<B1, B2>, Device = MultiDevice2<B1, B2>>,
    > RunnerChannel for DirectChannel<(B1, B2), Br>
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
        client: Self::Client,
    ) -> <Self::Bridge as MultiBackendBridge>::TensorType {
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
        client: Self::Client,
    ) -> <Self::Bridge as MultiBackendBridge>::TensorType {
        todo!()
    }

    fn get_bool_tensor(
        tensor: &TensorDescription,
        client: Self::Client,
    ) -> <Self::Bridge as MultiBackendBridge>::TensorType {
        todo!()
    }

    fn get_quantized_tensor(
        tensor: &TensorDescription,
        client: Self::Client,
    ) -> <Self::Bridge as MultiBackendBridge>::TensorType {
        todo!()
    }

    fn register_tensor(
        client: Self::Client,
        handle: <Self::Bridge as MultiBackendBridge>::TensorType,
        shape: Vec<usize>,
        dtype: crate::DType,
    ) -> RouterTensor<Self::Client> {
        let desc = match &client {
            MultiRunnerClient2::RunnerClient1(runner) => match handle {
                TensorHandle2::Handle1(handle) => runner.register_tensor(handle, shape, dtype),
                TensorHandle2::Handle2(_) => unreachable!(),
            },
            MultiRunnerClient2::RunnerClient2(runner) => match handle {
                TensorHandle2::Handle1(_) => unreachable!(),
                TensorHandle2::Handle2(handle) => runner.register_tensor(handle, shape, dtype),
            },
        };
        RouterTensor { desc, client }
    }
}
