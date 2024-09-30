use burn_common::stream::StreamId;

use super::{RunnerChannel, RunnerClient};
use crate::{repr::TensorDescription, TensorData};

// #[derive(Clone, Debug)]
// pub struct RouterTensor<R: MultiBackendRuntime> {
pub struct RouterTensor<C: RunnerChannel> {
    pub(crate) desc: TensorDescription,
    pub(crate) client: C::Client,
    pub(crate) stream: StreamId,
}

impl<C: RunnerChannel> RouterTensor<C> {
    pub(crate) async fn into_data(self) -> TensorData {
        let desc = self.desc;

        self.client.read_tensor(desc, self.stream).await
    }

    pub fn into_description(self) -> TensorDescription {
        self.desc
    }
}

impl<C: RunnerChannel> core::fmt::Debug for RouterTensor<C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("tensor"))
    }
}

impl<C: RunnerChannel> Clone for RouterTensor<C> {
    fn clone(&self) -> Self {
        Self {
            desc: self.desc.clone(),
            client: self.client.clone(),
            stream: self.stream,
        }
    }
}
