use crate::backend::tch::TchBackend;
use crate::tensor::TchElement;
use crate::tensor::{
    backend::tch::{TchKind, TchTensor},
    ops::*,
};

impl<E, const D: usize> TensorOpsMapComparison<TchBackend<E>, D> for TchTensor<E, D>
where
    E: TchElement,
{
    fn equal(
        &self,
        other: &Self,
    ) -> <TchBackend<E> as crate::back::Backend>::BoolTensorPrimitive<D> {
        let tensor = self.tensor.eq_tensor(&other.tensor);

        TchTensor {
            shape: self.shape,
            tensor,
            kind: TchKind::<bool>::new(),
        }
    }

    fn equal_scalar(
        &self,
        other: &<TchBackend<E> as crate::back::Backend>::Elem,
    ) -> <TchBackend<E> as crate::back::Backend>::BoolTensorPrimitive<D> {
        let other: f64 = (*other).into();
        let tensor = self.tensor.eq(other);

        TchTensor {
            shape: self.shape,
            tensor,
            kind: TchKind::<bool>::new(),
        }
    }

    fn greater(
        &self,
        other: &Self,
    ) -> <TchBackend<E> as crate::back::Backend>::BoolTensorPrimitive<D> {
        let tensor = self.tensor.greater_tensor(&other.tensor);

        TchTensor {
            shape: self.shape,
            tensor,
            kind: TchKind::<bool>::new(),
        }
    }

    fn greater_scalar(
        &self,
        other: &<TchBackend<E> as crate::back::Backend>::Elem,
    ) -> <TchBackend<E> as crate::back::Backend>::BoolTensorPrimitive<D> {
        let other: f64 = (*other).into();
        let tensor = self.tensor.greater(other);

        TchTensor {
            shape: self.shape,
            tensor,
            kind: TchKind::<bool>::new(),
        }
    }

    fn greater_equal(
        &self,
        other: &Self,
    ) -> <TchBackend<E> as crate::back::Backend>::BoolTensorPrimitive<D> {
        let tensor = self.tensor.greater_equal_tensor(&other.tensor);

        TchTensor {
            shape: self.shape,
            tensor,
            kind: TchKind::<bool>::new(),
        }
    }

    fn greater_equal_scalar(
        &self,
        other: &<TchBackend<E> as crate::back::Backend>::Elem,
    ) -> <TchBackend<E> as crate::back::Backend>::BoolTensorPrimitive<D> {
        let other: f64 = (*other).into();
        let tensor = self.tensor.greater_equal(other);

        TchTensor {
            shape: self.shape,
            tensor,
            kind: TchKind::<bool>::new(),
        }
    }

    fn lower(
        &self,
        other: &Self,
    ) -> <TchBackend<E> as crate::back::Backend>::BoolTensorPrimitive<D> {
        let tensor = self.tensor.less_tensor(&other.tensor);

        TchTensor {
            shape: self.shape,
            tensor,
            kind: TchKind::<bool>::new(),
        }
    }

    fn lower_scalar(
        &self,
        other: &<TchBackend<E> as crate::back::Backend>::Elem,
    ) -> <TchBackend<E> as crate::back::Backend>::BoolTensorPrimitive<D> {
        let other: f64 = (*other).into();
        let tensor = self.tensor.less(other);

        TchTensor {
            shape: self.shape,
            tensor,
            kind: TchKind::<bool>::new(),
        }
    }

    fn lower_equal(
        &self,
        other: &Self,
    ) -> <TchBackend<E> as crate::back::Backend>::BoolTensorPrimitive<D> {
        let tensor = self.tensor.less_equal_tensor(&other.tensor);

        TchTensor {
            shape: self.shape,
            tensor,
            kind: TchKind::<bool>::new(),
        }
    }

    fn lower_equal_scalar(
        &self,
        other: &<TchBackend<E> as crate::back::Backend>::Elem,
    ) -> <TchBackend<E> as crate::back::Backend>::BoolTensorPrimitive<D> {
        let other: f64 = (*other).into();
        let tensor = self.tensor.less_equal(other);

        TchTensor {
            shape: self.shape,
            tensor,
            kind: TchKind::<bool>::new(),
        }
    }
}
