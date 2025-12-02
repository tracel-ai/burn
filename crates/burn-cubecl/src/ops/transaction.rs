use burn_tensor::{
    DType, TensorData,
    backend::ExecutionError,
    ops::{TransactionOps, TransactionPrimitiveData},
};
use cubecl::server::{Binding, CopyDescriptor};

use crate::{CubeBackend, CubeRuntime, FloatElement, IntElement, element::BoolElement};

impl<R, F, I, BT> TransactionOps<Self> for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    async fn tr_execute(
        transaction: burn_tensor::ops::TransactionPrimitive<Self>,
    ) -> Result<burn_tensor::ops::TransactionPrimitiveData, ExecutionError> {
        let mut client = None;

        enum Kind {
            Float,
            Int,
            Bool,
        }

        #[derive(new)]
        struct BindingData {
            index: usize,
            kind: Kind,
            handle: Option<Binding>,
            shape: Vec<usize>,
            strides: Vec<usize>,
            dtype: DType,
        }

        let mut num_bindings = 0;

        let mut kinds = Vec::new();

        for t in transaction.read_floats.into_iter() {
            if client.is_none() {
                client = Some(t.client.clone());
            }

            let t = crate::kernel::into_contiguous_aligned(t);
            let binding = BindingData::new(
                num_bindings,
                Kind::Float,
                Some(t.handle.binding()),
                t.shape.into(),
                t.strides,
                t.dtype,
            );

            kinds.push(binding);
            num_bindings += 1;
        }
        for t in transaction.read_ints.into_iter() {
            if client.is_none() {
                client = Some(t.client.clone());
            }

            let t = crate::kernel::into_contiguous_aligned(t);
            let binding = BindingData::new(
                num_bindings,
                Kind::Int,
                Some(t.handle.binding()),
                t.shape.into(),
                t.strides,
                t.dtype,
            );

            kinds.push(binding);
            num_bindings += 1;
        }
        for t in transaction.read_bools.into_iter() {
            if client.is_none() {
                client = Some(t.client.clone());
            }

            let t = crate::kernel::into_contiguous_aligned(t);
            let binding = BindingData::new(
                num_bindings,
                Kind::Bool,
                Some(t.handle.binding()),
                t.shape.into(),
                t.strides,
                t.dtype,
            );

            kinds.push(binding);
            num_bindings += 1;
        }

        let client = client.unwrap();

        let bindings = kinds
            .iter_mut()
            .map(|b| {
                CopyDescriptor::new(
                    b.handle.take().unwrap(),
                    &b.shape,
                    &b.strides,
                    b.dtype.size(),
                )
            })
            .collect();

        let mut data: Vec<Option<_>> = client
            .read_tensor_async(bindings)
            .await
            .map_err(|err| ExecutionError::WithContext {
                reason: format!("{err:?}"),
            })?
            .into_iter()
            .map(Some)
            .collect::<Vec<Option<_>>>();

        let mut result = TransactionPrimitiveData::default();

        for binding in kinds {
            let bytes = data.get_mut(binding.index).unwrap().take().unwrap();
            let t_data = TensorData::from_bytes(bytes, binding.shape, binding.dtype);

            match binding.kind {
                Kind::Float => {
                    result.read_floats.push(t_data);
                }
                Kind::Int => {
                    result.read_ints.push(t_data);
                }
                Kind::Bool => {
                    result.read_bools.push(t_data);
                }
            }
        }

        Ok(result)
    }
}
