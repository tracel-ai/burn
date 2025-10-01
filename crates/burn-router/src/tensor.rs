use core::sync::atomic::{AtomicU32, Ordering};

use alloc::{sync::Arc, vec::Vec};

use super::RunnerClient;
use burn_ir::{TensorId, TensorIr, TensorStatus};
use burn_tensor::{DType, Shape, TensorData, TensorMetadata};

/// Tensor primitive for the [router backend](crate::BackendRouter).
pub struct RouterTensor<C: RunnerClient> {
    pub(crate) id: TensorId,
    pub(crate) shape: Shape,
    pub(crate) dtype: DType,
    /// The client that has this tensor
    pub client: C,
    pub(crate) count: Arc<AtomicU32>,
}

impl<C: RunnerClient> TensorMetadata for RouterTensor<C> {
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn shape(&self) -> Shape {
        self.shape.clone()
    }

    fn rank(&self) -> usize {
        self.shape.num_dims()
    }
}

impl<C: RunnerClient> RouterTensor<C> {
    /// Create a new router tensor.
    pub fn new(id: TensorId, shape: Shape, dtype: DType, client: C) -> Self {
        Self {
            id,
            shape,
            dtype,
            client,
            count: Arc::new(AtomicU32::new(1)),
        }
    }

    pub(crate) async fn into_data(self) -> TensorData {
        self.client.clone().read_tensor(self.into_ir()).await
    }

    /// Get the ir for this tensor
    pub fn into_ir(mut self) -> TensorIr {
        let count = self.count.load(Ordering::Relaxed);
        let status = self.status(count);
        let mut shape_out = Shape::from(Vec::<usize>::new());
        core::mem::swap(&mut self.shape, &mut shape_out);

        if let TensorStatus::ReadWrite = status {
            // Avoids an unwanted drop on the same thread.
            //
            // Since `drop` is called after `into_ir`, we must not register a drop if the tensor
            // was consumed with a `ReadWrite` status.
            self.count.fetch_add(1, Ordering::Relaxed);
        }

        TensorIr {
            status,
            shape: shape_out,
            id: self.id,
            dtype: self.dtype,
        }
    }

    pub(crate) fn to_ir_out(&self) -> TensorIr {
        TensorIr {
            status: TensorStatus::NotInit,
            shape: self.shape.clone(),
            id: self.id,
            dtype: self.dtype,
        }
    }

    pub(crate) fn status(&self, count: u32) -> TensorStatus {
        if count <= 1 {
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
                "{{ id: {:?}, shape: {:?}, dtype: {:?}, device: {:?} }}",
                self.id,
                self.shape,
                self.dtype,
                self.client.device().clone(),
            )
            .as_str(),
        )
    }
}

impl<C: RunnerClient> Clone for RouterTensor<C> {
    fn clone(&self) -> Self {
        self.count.fetch_add(1, Ordering::Relaxed);

        Self {
            id: self.id,
            shape: self.shape.clone(),
            client: self.client.clone(),
            dtype: self.dtype,
            count: self.count.clone(),
        }
    }
}

impl<C: RunnerClient> Drop for RouterTensor<C> {
    fn drop(&mut self) {
        let count = self.count.fetch_sub(1, Ordering::Relaxed);

        match self.status(count) {
            TensorStatus::ReadWrite => {
                let id = self.id;
                let mut shape = Shape::from(Vec::<usize>::new());
                core::mem::swap(&mut shape, &mut self.shape);

                let ir = TensorIr {
                    id,
                    shape,
                    status: TensorStatus::ReadWrite,
                    dtype: self.dtype,
                };
                self.client.register(burn_ir::OperationIr::Drop(ir));
            }
            TensorStatus::ReadOnly => {}
            TensorStatus::NotInit => {}
        }
    }
}
