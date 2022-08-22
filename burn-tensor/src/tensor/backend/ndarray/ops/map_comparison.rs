use crate::backend::ndarray::NdArrayBackend;
use crate::tensor::{backend::ndarray::NdArrayTensor, ops::*};
use crate::NdArrayElement;

impl<E, const D: usize> TensorOpsMapComparison<NdArrayBackend<E>, D> for NdArrayTensor<E, D>
where
    E: NdArrayElement,
{
    fn greater(
        &self,
        other: &Self,
    ) -> <NdArrayBackend<E> as crate::back::Backend>::BoolTensorPrimitive<D> {
        let tensor = self.sub(other);
        let zero = E::zeros(&E::default());
        tensor.greater_scalar(&zero)
    }

    fn greater_scalar(
        &self,
        other: &<NdArrayBackend<E> as crate::back::Backend>::Elem,
    ) -> <NdArrayBackend<E> as crate::back::Backend>::BoolTensorPrimitive<D> {
        let array = self.array.mapv(|a| a > *other).into_shared();

        NdArrayTensor {
            shape: self.shape,
            array,
        }
    }

    fn greater_equal(
        &self,
        other: &Self,
    ) -> <NdArrayBackend<E> as crate::back::Backend>::BoolTensorPrimitive<D> {
        let tensor = self.sub(other);
        let zero = E::zeros(&E::default());
        tensor.greater_equal_scalar(&zero)
    }

    fn greater_equal_scalar(
        &self,
        other: &<NdArrayBackend<E> as crate::back::Backend>::Elem,
    ) -> <NdArrayBackend<E> as crate::back::Backend>::BoolTensorPrimitive<D> {
        let array = self.array.mapv(|a| a >= *other).into_shared();

        NdArrayTensor {
            shape: self.shape,
            array,
        }
    }

    fn lower(
        &self,
        other: &Self,
    ) -> <NdArrayBackend<E> as crate::back::Backend>::BoolTensorPrimitive<D> {
        let tensor = self.sub(other);
        let zero = E::zeros(&E::default());
        tensor.lower_scalar(&zero)
    }

    fn lower_scalar(
        &self,
        other: &<NdArrayBackend<E> as crate::back::Backend>::Elem,
    ) -> <NdArrayBackend<E> as crate::back::Backend>::BoolTensorPrimitive<D> {
        let array = self.array.mapv(|a| a < *other).into_shared();

        NdArrayTensor {
            shape: self.shape,
            array,
        }
    }

    fn lower_equal(
        &self,
        other: &Self,
    ) -> <NdArrayBackend<E> as crate::back::Backend>::BoolTensorPrimitive<D> {
        let tensor = self.sub(other);
        let zero = E::zeros(&E::default());
        tensor.lower_equal_scalar(&zero)
    }

    fn lower_equal_scalar(
        &self,
        other: &<NdArrayBackend<E> as crate::back::Backend>::Elem,
    ) -> <NdArrayBackend<E> as crate::back::Backend>::BoolTensorPrimitive<D> {
        let array = self.array.mapv(|a| a <= *other).into_shared();

        NdArrayTensor {
            shape: self.shape,
            array,
        }
    }
}

#[cfg(tests)]
mod tests {
    use super::*;

    #[test]
    fn test_greater() {}
}
