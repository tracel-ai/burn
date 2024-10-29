use core::marker::PhantomData;

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
