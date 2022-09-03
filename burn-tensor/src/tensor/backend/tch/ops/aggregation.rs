use crate::{
    backend::Backend,
    tensor::{
        backend::tch::{TchBackend, TchTensor},
        ops::*,
        Shape,
    },
    TchElement,
};

impl<E: TchElement, const D: usize> TensorOpsAggregation<TchBackend<E>, D> for TchTensor<E, D> {
    fn mean(&self) -> <TchBackend<E> as Backend>::TensorPrimitive<1> {
        let kind = self.kind.clone();
        let tensor = self.tensor.mean(kind.kind());
        let shape = Shape::new([1]);

        TchTensor {
            tensor,
            kind,
            shape,
        }
    }

    fn sum(&self) -> <TchBackend<E> as Backend>::TensorPrimitive<1> {
        let kind = self.kind.clone();
        let tensor = self.tensor.sum(kind.kind());
        let shape = Shape::new([1]);

        TchTensor {
            tensor,
            kind,
            shape,
        }
    }

    fn mean_dim(&self, dim: usize) -> <TchBackend<E> as Backend>::TensorPrimitive<D> {
        let kind = self.kind.clone();
        let tensor = self.tensor.mean_dim(&[dim as i64], true, kind.kind());
        let shape = Shape::from(tensor.size());

        TchTensor {
            tensor,
            kind,
            shape,
        }
    }

    fn sum_dim(&self, dim: usize) -> <TchBackend<E> as Backend>::TensorPrimitive<D> {
        let kind = self.kind.clone();
        let tensor = self
            .tensor
            .sum_dim_intlist(&[dim as i64], true, kind.kind());
        let shape = Shape::from(tensor.size());

        TchTensor {
            tensor,
            kind,
            shape,
        }
    }
}
