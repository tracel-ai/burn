use burn_common::stream::StreamId;

use super::{ServerClient, ServerRuntime};
use crate::{repr::TensorDescription, TensorData};

pub struct ServerTensor<R: ServerRuntime> {
    pub(crate) desc: TensorDescription,
    pub(crate) client: R::Client,
    pub(crate) stream: StreamId,
}

impl<R: ServerRuntime> ServerTensor<R> {
    pub(crate) async fn into_data(self) -> TensorData {
        let id = self.stream;
        let desc = self.desc;

        self.client.read_tensor(desc, id).await
    }

    pub fn into_description(self) -> TensorDescription {
        self.desc
    }
}

impl<R: ServerRuntime> core::fmt::Debug for ServerTensor<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("tensor"))
    }
}

impl<R: ServerRuntime> Clone for ServerTensor<R> {
    fn clone(&self) -> Self {
        Self {
            desc: self.desc.clone(),
            client: self.client.clone(),
            stream: self.stream,
        }
    }
}
