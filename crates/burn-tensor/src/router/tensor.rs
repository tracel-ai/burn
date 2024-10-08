use alloc::sync::Arc;

use super::RunnerClient;
use crate::{
    repr::{TensorDescription, TensorId, TensorStatus},
    DType, TensorData,
};

pub struct RouterTensor<C: RunnerClient> {
    pub(crate) id: Arc<TensorId>,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: DType,
    pub(crate) client: C,
}

impl<C: RunnerClient> RouterTensor<C> {
    pub(crate) async fn into_data(self) -> TensorData {
        self.client
            .clone()
            .read_tensor(self.into_description())
            .await
    }

    pub(crate) fn into_description(mut self) -> TensorDescription {
        let status = self.status();
        let mut shape_out = Vec::new();
        core::mem::swap(&mut self.shape, &mut shape_out);

        TensorDescription {
            status,
            shape: shape_out,
            id: *self.id.as_ref(),
            dtype: self.dtype,
        }
    }

    pub(crate) fn to_description_out(&self) -> TensorDescription {
        TensorDescription {
            status: TensorStatus::NotInit,
            shape: self.shape.clone(),
            id: *self.id.as_ref(),
            dtype: self.dtype,
        }
    }

    pub(crate) fn status(&self) -> TensorStatus {
        if Arc::strong_count(&self.id) <= 1 {
            TensorStatus::ReadWrite
        } else {
            TensorStatus::ReadOnly
        }
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
            id: self.id.clone(),
            shape: self.shape.clone(),
            client: self.client.clone(),
            dtype: self.dtype,
        }
    }
}
