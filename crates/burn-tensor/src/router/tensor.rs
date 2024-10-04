use super::RunnerClient;
use crate::{repr::TensorDescription, TensorData};

// #[derive(Clone, Debug)]
pub struct RouterTensor<C: RunnerClient> {
    pub(crate) desc: TensorDescription,
    pub(crate) client: C,
    // pub(crate) stream: StreamId,
}

impl<C: RunnerClient> RouterTensor<C> {
    pub(crate) async fn into_data(self) -> TensorData {
        self.client.read_tensor(self.desc).await
    }

    pub fn into_description(self) -> TensorDescription {
        self.desc
    }

    pub fn to_description(&self) -> TensorDescription {
        self.desc.clone()
    }

    // TODO: should hold same fields as TensorDescription but hold Arc<TensorId>
    // for refcount
    //     pub(crate) fn status(&self) -> TensorStatus {
    //         if Arc::strong_count(&self.desc.id) <= 1 {
    //             TensorStatus::ReadWrite
    //         } else {
    //             TensorStatus::ReadOnly
    //         }
    //     }
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
        }
    }
}
