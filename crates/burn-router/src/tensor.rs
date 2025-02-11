use alloc::{sync::Arc, vec::Vec};

use super::RunnerClient;
use burn_ir::{TensorId, TensorIr, TensorStatus};
use burn_tensor::{DType, Shape, TensorData, TensorMetadata};

/// Tensor primitive for the [router backend](crate::BackendRouter).
pub struct RouterTensor<C: RunnerClient> {
    pub(crate) id: Arc<TensorId>,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: DType,
    pub(crate) client: C,

    // Orphan means that a tensor is never converted into a representation when it becomes `ReadWrite`.
    //
    // When a tensor is dropped and is still an orphan, we need to register it as such to avoid
    // memory leak.
    pub(crate) is_orphan: bool,
}

impl<C: RunnerClient> TensorMetadata for RouterTensor<C> {
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn shape(&self) -> Shape {
        Shape::from(self.shape.clone())
    }
}

impl<C: RunnerClient> RouterTensor<C> {
    /// Create a new router tensor.
    pub fn new(id: Arc<TensorId>, shape: Vec<usize>, dtype: DType, client: C) -> Self {
        Self {
            id,
            shape,
            dtype,
            client,
            is_orphan: true,
        }
    }

    pub(crate) async fn into_data(self) -> TensorData {
        self.client.clone().read_tensor(self.into_ir()).await
    }

    pub(crate) fn into_ir(mut self) -> TensorIr {
        let status = self.status();
        let mut shape_out = Vec::new();
        core::mem::swap(&mut self.shape, &mut shape_out);

        if let TensorStatus::ReadWrite = status {
            self.is_orphan = false;
        }

        TensorIr {
            status,
            shape: shape_out,
            id: *self.id.as_ref(),
            dtype: self.dtype,
        }
    }

    pub(crate) fn to_ir_out(&self) -> TensorIr {
        TensorIr {
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
