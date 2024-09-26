use burn_common::stream::StreamId;

use super::{RouteInfo, RunnerClient};
use crate::{repr::TensorDescription, TensorData};

pub struct RouterTensor<C: RunnerClient> {
    pub(crate) desc: TensorDescription,
    pub(crate) client: C,
    pub(crate) stream: StreamId,
    pub(crate) runner_id: usize,
}

impl<C: RunnerClient> RouterTensor<C> {
    pub(crate) async fn into_data(self) -> TensorData {
        let desc = self.desc;
        let info = RouteInfo {
            stream: self.stream,
            runner_id: self.runner_id,
        };

        self.client.read_tensor(desc, info).await
    }

    pub fn into_description(self) -> TensorDescription {
        self.desc
    }
}

impl<C: RunnerClient> core::fmt::Debug for RouterTensor<C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("tensor"))
    }
}

impl<C: RunnerClient> Clone for RouterTensor<C> {
    fn clone(&self) -> Self {
        Self {
            desc: self.desc.clone(),
            client: self.client.clone(),
            stream: self.stream,
            runner_id: self.runner_id,
        }
    }
}
