use burn_common::stream::StreamId;

use super::{RunnerClient, RunnerRuntime};
use crate::{repr::TensorDescription, TensorData};

pub struct RunnerTensor<R: RunnerRuntime> {
    pub(crate) desc: TensorDescription,
    pub(crate) client: R::Client,
    pub(crate) stream: StreamId,
}

impl<R: RunnerRuntime> RunnerTensor<R> {
    pub(crate) async fn into_data(self) -> TensorData {
        let id = self.stream;
        let desc = self.desc;

        self.client.read_tensor(desc, id).await
    }

    pub fn into_description(self) -> TensorDescription {
        self.desc
    }
}

impl<R: RunnerRuntime> core::fmt::Debug for RunnerTensor<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("tensor"))
    }
}

impl<R: RunnerRuntime> Clone for RunnerTensor<R> {
    fn clone(&self) -> Self {
        Self {
            desc: self.desc.clone(),
            client: self.client.clone(),
            stream: self.stream,
        }
    }
}
