use burn_tensor::{
    ops::{TransactionOps, TransactionPrimitiveResult},
    TensorData,
};

use crate::{FloatElement, IntElement, JitBackend, JitRuntime};
use cubecl::CubeElement;

impl<R, F, I> TransactionOps<Self> for JitBackend<R, F, I>
where
    R: JitRuntime,
    F: FloatElement,
    I: IntElement,
{
    fn tr_execute(
        transaction: burn_tensor::ops::TransactionPrimitive<Self>,
    ) -> impl std::future::Future<Output = burn_tensor::ops::TransactionPrimitiveResult> + 'static + Send
    {
        let mut bindings = Vec::new();
        let mut client = None;

        enum Kind {
            Float(usize, Vec<usize>),
            Int(usize, Vec<usize>),
            Bool(usize, Vec<usize>),
        }

        let mut num_bindings = 0;

        let mut kinds = Vec::new();

        transaction.read_floats.into_iter().for_each(|t| {
            if client.is_none() {
                client = Some(t.client.clone());
            }

            kinds.push(Kind::Float(num_bindings, t.shape.into()));
            num_bindings += 1;
            bindings.push(t.handle.binding())
        });
        transaction.read_ints.into_iter().for_each(|t| {
            if client.is_none() {
                client = Some(t.client.clone());
            }

            kinds.push(Kind::Int(num_bindings, t.shape.into()));
            num_bindings += 1;
            bindings.push(t.handle.binding())
        });
        transaction.read_bools.into_iter().for_each(|t| {
            if client.is_none() {
                client = Some(t.client.clone());
            }

            kinds.push(Kind::Bool(num_bindings, t.shape.into()));
            num_bindings += 1;
            bindings.push(t.handle.binding())
        });

        let client = client.unwrap();

        async move {
            let mut data = client
                .read_async(bindings)
                .await
                .into_iter()
                .map(Some)
                .collect::<Vec<_>>();

            let mut result = TransactionPrimitiveResult::default();

            for kind in kinds {
                match kind {
                    Kind::Float(index, shape) => {
                        let bytes = data.get_mut(index).unwrap().take().unwrap();

                        result
                            .read_floats
                            .push(TensorData::new(F::from_elem_data(bytes), shape));
                    }
                    Kind::Int(index, shape) => {
                        let bytes = data.get_mut(index).unwrap().take().unwrap();
                        result
                            .read_ints
                            .push(TensorData::new(I::from_elem_data(bytes).to_vec(), shape));
                    }
                    Kind::Bool(index, shape) => {
                        let bytes = data.get_mut(index).unwrap().take().unwrap();
                        result
                            .read_bools
                            .push(TensorData::new(bool::from_elem_data(bytes).to_vec(), shape));
                    }
                }
            }

            result
        }
    }
}
