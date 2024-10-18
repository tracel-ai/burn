use alloc::{sync::Arc, vec::Vec};

use super::RunnerClient;
use burn_tensor::{
    repr::{TensorDescription, TensorId, TensorStatus},
    DType, Shape, TensorData,
};

/// Tensor primitive for the [router backend](crate::BackendRouter).
pub struct RouterTensor<C: RunnerClient> {
    pub(crate) id: Arc<TensorId>,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: DType,
    pub(crate) client: C,

    // Orphan means that a tensor is never converted into a description when it becomes `ReadWrite`.
    //
    // When a tensor is dropped and is still an orphan, we need to register it as such to avoid
    // memory leak.
    pub(crate) is_orphan: bool,
}

impl<C: RunnerClient> RouterTensor<C> {
    pub(crate) fn new(id: Arc<TensorId>, shape: Vec<usize>, dtype: DType, client: C) -> Self {
        Self {
            id,
            shape,
            dtype,
            client,
            is_orphan: true,
        }
    }

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

        if let TensorStatus::ReadWrite = status {
            self.is_orphan = false;
        }

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

    pub(crate) fn shape(&self) -> Shape {
        Shape::from(self.shape.clone())
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
        f.write_str(
            format!(
                "{{ id: {:?}, shape: {:?}, dtype: {:?}, should_drop: {:?}, device: {:?} }}",
                self.id,
                self.shape,
                self.dtype,
                self.is_orphan,
                self.client.device().clone(),
            )
            .as_str(),
        )
    }
}

impl<C: RunnerClient> Clone for RouterTensor<C> {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            shape: self.shape.clone(),
            client: self.client.clone(),
            dtype: self.dtype,
            is_orphan: self.is_orphan,
        }
    }
}

impl<C: RunnerClient> Drop for RouterTensor<C> {
    fn drop(&mut self) {
        if !self.is_orphan {
            return;
        }

        match self.status() {
            TensorStatus::ReadWrite => {
                self.client.register_orphan(&self.id);
            }
            TensorStatus::ReadOnly => {}
            TensorStatus::NotInit => {}
        }
    }
}
