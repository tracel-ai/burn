use alloc::{string::String, vec::Vec};
use burn_backend::{Backend, tensor::FloatTensor};
use core::marker::PhantomData;

use crate::tensor::FuncTensor;

/// A structure whose float tensor leaves can be traversed, rebound to another backend, and zipped
/// with another tree of identical structure.
///
/// The trait is the parameter model for the functional transformations. `WithBackend` is the
/// central operation: it produces the identical structure with the backend type parameter
/// rebound, which is how `grad` lifts a parameter tree to `Autodiff<B>`, how `jvp` lifts it to
/// `ForwardDiff<B>`, and how `vmap` lifts it to `Vmap<B>`.
///
/// # Limitations
///
/// Int/Bool fields must be closed over, not carried in the tree. Fields marked `#[pytree(skip)]`
/// must not reference the backend type parameter `B`. Differentiable leaves are float tensors only;
/// close over int/bool tensors (indices, masks) outside the tree.
pub trait PyTree<B: Backend>: Clone + Send {
    /// The identical structural type with its backend rebound to `B2`.
    type WithBackend<B2: Backend>: PyTree<B2, WithBackend<B> = Self>;

    /// Number of differentiable float tensor leaves.
    fn num_leaves(&self) -> usize;

    /// Maps each float primitive leaf from `B` to `B2`.
    fn map_float<B2: Backend, F>(&self, f: &mut F) -> Self::WithBackend<B2>
    where
        F: FnMut(&FloatTensor<B>) -> FloatTensor<B2>;

    /// Visits each float primitive leaf in deterministic order.
    fn visit_float<F>(&self, f: &mut F)
    where
        F: FnMut(&FloatTensor<B>);

    /// Zips two trees of identical structure across `B` and `B2` into `B3`.
    fn zip_float<B2: Backend, B3: Backend, F>(
        &self,
        other: &Self::WithBackend<B2>,
        f: &mut F,
    ) -> Self::WithBackend<B3>
    where
        F: FnMut(&FloatTensor<B>, &FloatTensor<B2>) -> FloatTensor<B3>;
}

/// Rebinds a non-differentiable field to another backend.
///
/// This is used by the `PyTree` derive for fields that carry backend data but are not
/// differentiable float leaves, such as index and mask tensors. Implementations that cannot
/// transfer data between arbitrary backends should return a type that preserves the source
/// backend, and callers should keep such fields out of the differentiated tree.
pub trait BackendRebind<B: Backend>: Clone + Send {
    /// The rebound type.
    type WithBackend<B2: Backend>;

    /// Rebinds the value to `B2`.
    fn rebind<B2: Backend>(&self, device: &B2::Device) -> Self::WithBackend<B2>;
}

impl<B: Backend, const D: usize> PyTree<B> for FuncTensor<B, D> {
    type WithBackend<B2: Backend> = FuncTensor<B2, D>;

    fn num_leaves(&self) -> usize {
        1
    }

    fn map_float<B2: Backend, F>(&self, f: &mut F) -> Self::WithBackend<B2>
    where
        F: FnMut(&FloatTensor<B>) -> FloatTensor<B2>,
    {
        FuncTensor::from_primitive(f(self.primitive()))
    }

    fn visit_float<F>(&self, f: &mut F)
    where
        F: FnMut(&FloatTensor<B>),
    {
        f(self.primitive());
    }

    fn zip_float<B2: Backend, B3: Backend, F>(
        &self,
        other: &Self::WithBackend<B2>,
        f: &mut F,
    ) -> Self::WithBackend<B3>
    where
        F: FnMut(&FloatTensor<B>, &FloatTensor<B2>) -> FloatTensor<B3>,
    {
        FuncTensor::from_primitive(f(self.primitive(), other.primitive()))
    }
}

impl<B: Backend> PyTree<B> for () {
    type WithBackend<B2: Backend> = ();

    fn num_leaves(&self) -> usize {
        0
    }

    fn map_float<B2: Backend, F>(&self, _f: &mut F) -> Self::WithBackend<B2>
    where
        F: FnMut(&FloatTensor<B>) -> FloatTensor<B2>,
    {
    }

    fn visit_float<F>(&self, _f: &mut F)
    where
        F: FnMut(&FloatTensor<B>),
    {
    }

    fn zip_float<B2: Backend, B3: Backend, F>(
        &self,
        _other: &Self::WithBackend<B2>,
        _f: &mut F,
    ) -> Self::WithBackend<B3>
    where
        F: FnMut(&FloatTensor<B>, &FloatTensor<B2>) -> FloatTensor<B3>,
    {
    }
}

macro_rules! impl_pytree_tuple {
    ($($T:ident => $idx:tt),+) => {
        impl<B: Backend, $($T: PyTree<B>),+> PyTree<B> for ($($T,)+) {
            type WithBackend<B2: Backend> = ($($T::WithBackend<B2>,)+);

            fn num_leaves(&self) -> usize {
                0 $(+ self.$idx.num_leaves())+
            }

            fn map_float<B2: Backend, F>(&self, f: &mut F) -> Self::WithBackend<B2>
            where
                F: FnMut(&FloatTensor<B>) -> FloatTensor<B2>,
            {
                ($(self.$idx.map_float(f),)+)
            }

            fn visit_float<F>(&self, f: &mut F)
            where
                F: FnMut(&FloatTensor<B>),
            {
                $(self.$idx.visit_float(f);)+
            }

            fn zip_float<B2: Backend, B3: Backend, F>(
                &self,
                other: &Self::WithBackend<B2>,
                f: &mut F,
            ) -> Self::WithBackend<B3>
            where
                F: FnMut(&FloatTensor<B>, &FloatTensor<B2>) -> FloatTensor<B3>,
            {
                ($(self.$idx.zip_float(&other.$idx, f),)+)
            }
        }
    };
}

impl_pytree_tuple!(T0 => 0);
impl_pytree_tuple!(T0 => 0, T1 => 1);
impl_pytree_tuple!(T0 => 0, T1 => 1, T2 => 2);
impl_pytree_tuple!(T0 => 0, T1 => 1, T2 => 2, T3 => 3);
impl_pytree_tuple!(T0 => 0, T1 => 1, T2 => 2, T3 => 3, T4 => 4);
impl_pytree_tuple!(T0 => 0, T1 => 1, T2 => 2, T3 => 3, T4 => 4, T5 => 5);
impl_pytree_tuple!(T0 => 0, T1 => 1, T2 => 2, T3 => 3, T4 => 4, T5 => 5, T6 => 6);
impl_pytree_tuple!(
    T0 => 0,
    T1 => 1,
    T2 => 2,
    T3 => 3,
    T4 => 4,
    T5 => 5,
    T6 => 6,
    T7 => 7
);

impl<B: Backend, T: PyTree<B>> PyTree<B> for Vec<T> {
    type WithBackend<B2: Backend> = Vec<T::WithBackend<B2>>;

    fn num_leaves(&self) -> usize {
        self.iter().map(PyTree::num_leaves).sum()
    }

    fn map_float<B2: Backend, F>(&self, f: &mut F) -> Self::WithBackend<B2>
    where
        F: FnMut(&FloatTensor<B>) -> FloatTensor<B2>,
    {
        self.iter().map(|value| value.map_float(f)).collect()
    }

    fn visit_float<F>(&self, f: &mut F)
    where
        F: FnMut(&FloatTensor<B>),
    {
        for value in self {
            value.visit_float(f);
        }
    }

    fn zip_float<B2: Backend, B3: Backend, F>(
        &self,
        other: &Self::WithBackend<B2>,
        f: &mut F,
    ) -> Self::WithBackend<B3>
    where
        F: FnMut(&FloatTensor<B>, &FloatTensor<B2>) -> FloatTensor<B3>,
    {
        assert_eq!(
            self.len(),
            other.len(),
            "PyTree Vec length mismatch while zipping"
        );
        self.iter()
            .zip(other.iter())
            .map(|(lhs, rhs)| lhs.zip_float(rhs, f))
            .collect()
    }
}

impl<B: Backend, T: PyTree<B>> PyTree<B> for Option<T> {
    type WithBackend<B2: Backend> = Option<T::WithBackend<B2>>;

    fn num_leaves(&self) -> usize {
        self.as_ref().map_or(0, PyTree::num_leaves)
    }

    fn map_float<B2: Backend, F>(&self, f: &mut F) -> Self::WithBackend<B2>
    where
        F: FnMut(&FloatTensor<B>) -> FloatTensor<B2>,
    {
        self.as_ref().map(|value| value.map_float(f))
    }

    fn visit_float<F>(&self, f: &mut F)
    where
        F: FnMut(&FloatTensor<B>),
    {
        if let Some(value) = self {
            value.visit_float(f);
        }
    }

    fn zip_float<B2: Backend, B3: Backend, F>(
        &self,
        other: &Self::WithBackend<B2>,
        f: &mut F,
    ) -> Self::WithBackend<B3>
    where
        F: FnMut(&FloatTensor<B>, &FloatTensor<B2>) -> FloatTensor<B3>,
    {
        match (self, other) {
            (Some(lhs), Some(rhs)) => Some(lhs.zip_float(rhs, f)),
            (None, None) => None,
            _ => panic!("PyTree Option structure mismatch while zipping"),
        }
    }
}

impl<B: Backend, T: PyTree<B>, const N: usize> PyTree<B> for [T; N] {
    type WithBackend<B2: Backend> = [T::WithBackend<B2>; N];

    fn num_leaves(&self) -> usize {
        self.iter().map(PyTree::num_leaves).sum()
    }

    fn map_float<B2: Backend, F>(&self, f: &mut F) -> Self::WithBackend<B2>
    where
        F: FnMut(&FloatTensor<B>) -> FloatTensor<B2>,
    {
        core::array::from_fn(|index| self[index].map_float(f))
    }

    fn visit_float<F>(&self, f: &mut F)
    where
        F: FnMut(&FloatTensor<B>),
    {
        for value in self {
            value.visit_float(f);
        }
    }

    fn zip_float<B2: Backend, B3: Backend, F>(
        &self,
        other: &Self::WithBackend<B2>,
        f: &mut F,
    ) -> Self::WithBackend<B3>
    where
        F: FnMut(&FloatTensor<B>, &FloatTensor<B2>) -> FloatTensor<B3>,
    {
        core::array::from_fn(|index| self[index].zip_float(&other[index], f))
    }
}

macro_rules! impl_pytree_scalar {
    ($($ty:ty),* $(,)?) => {
        $(
            impl<B: Backend> PyTree<B> for $ty {
                type WithBackend<B2: Backend> = Self;

                fn num_leaves(&self) -> usize {
                    0
                }

                fn map_float<B2: Backend, F>(&self, _f: &mut F) -> Self::WithBackend<B2>
                where
                    F: FnMut(&FloatTensor<B>) -> FloatTensor<B2>,
                {
                    self.clone()
                }

                fn visit_float<F>(&self, _f: &mut F)
                where
                    F: FnMut(&FloatTensor<B>),
                {
                }

                fn zip_float<B2: Backend, B3: Backend, F>(
                    &self,
                    _other: &Self::WithBackend<B2>,
                    _f: &mut F,
                ) -> Self::WithBackend<B3>
                where
                    F: FnMut(&FloatTensor<B>, &FloatTensor<B2>) -> FloatTensor<B3>,
                {
                    self.clone()
                }
            }
        )*
    };
}

impl_pytree_scalar!(f32, f64, i32, i64, u32, u64, usize, isize, bool, String);

impl<B: Backend, T: Send + 'static> PyTree<B> for PhantomData<T> {
    type WithBackend<B2: Backend> = PhantomData<T>;

    fn num_leaves(&self) -> usize {
        0
    }

    fn map_float<B2: Backend, F>(&self, _f: &mut F) -> Self::WithBackend<B2>
    where
        F: FnMut(&FloatTensor<B>) -> FloatTensor<B2>,
    {
        PhantomData
    }

    fn visit_float<F>(&self, _f: &mut F)
    where
        F: FnMut(&FloatTensor<B>),
    {
    }

    fn zip_float<B2: Backend, B3: Backend, F>(
        &self,
        _other: &Self::WithBackend<B2>,
        _f: &mut F,
    ) -> Self::WithBackend<B3>
    where
        F: FnMut(&FloatTensor<B>, &FloatTensor<B2>) -> FloatTensor<B3>,
    {
        PhantomData
    }
}

macro_rules! impl_backend_rebind_scalar {
    ($($ty:ty),* $(,)?) => {
        $(
            impl<B: Backend> BackendRebind<B> for $ty {
                type WithBackend<B2: Backend> = Self;

                fn rebind<B2: Backend>(&self, _device: &B2::Device) -> Self::WithBackend<B2> {
                    self.clone()
                }
            }
        )*
    };
}

impl_backend_rebind_scalar!(f32, f64, i32, i64, u32, u64, usize, isize, bool, String);

impl<B: Backend, T: Send + 'static> BackendRebind<B> for PhantomData<T> {
    type WithBackend<B2: Backend> = PhantomData<T>;

    fn rebind<B2: Backend>(&self, _device: &B2::Device) -> Self::WithBackend<B2> {
        PhantomData
    }
}

#[cfg(test)]
mod tests {
    use crate::{FuncTensor, PyTree, grad::grad};
    use alloc::string::{String, ToString};
    use alloc::vec;
    use alloc::vec::Vec;
    use burn_autodiff::Autodiff;
    use burn_backend::{TensorData, TensorMetadata, ops::FloatTensorOps};
    use burn_flex::{Flex, FlexDevice};

    fn tensor(values: &[f32]) -> FuncTensor<Flex, 1> {
        FuncTensor::from_data(TensorData::from(values), &FlexDevice)
    }

    #[derive(Clone, Debug, PyTree)]
    struct Derived<B: burn_backend::Backend, const N: usize> {
        weights: FuncTensor<B, 1>,
        bias: FuncTensor<B, 1>,
        #[pytree(skip)]
        metadata: usize,
        #[pytree(skip)]
        label: String,
    }

    #[derive(Clone, Debug, PyTree)]
    struct Params<B: burn_backend::Backend> {
        w: FuncTensor<B, 1>,
        b: FuncTensor<B, 1>,
        #[pytree(skip)]
        metadata: usize,
    }

    #[test]
    fn tuple_and_array_traversal() {
        let tuple = (tensor(&[1.0]), tensor(&[2.0, 3.0]));
        assert_eq!(tuple.num_leaves(), 2);

        let array = [tensor(&[1.0]), tensor(&[2.0])];
        assert_eq!(array.num_leaves(), 2);

        let mut visited = Vec::new();
        tuple.visit_float(&mut |leaf| visited.push(leaf.shape().num_elements()));
        assert_eq!(visited, vec![1, 2]);
    }

    #[test]
    fn vec_and_option_traversal() {
        let values = vec![tensor(&[1.0]), tensor(&[2.0]), tensor(&[3.0])];
        assert_eq!(values.num_leaves(), 3);

        let some = Some(tensor(&[1.0]));
        assert_eq!(some.num_leaves(), 1);
        let none: Option<FuncTensor<Flex, 1>> = None;
        assert_eq!(none.num_leaves(), 0);
    }

    #[test]
    fn derived_struct_traversal_and_skipped_fields() {
        let params = Derived::<Flex, 1> {
            weights: tensor(&[1.0, 2.0]),
            bias: tensor(&[3.0]),
            metadata: 42,
            label: "linear".to_string(),
        };
        assert_eq!(params.num_leaves(), 2);

        let rebound = params.map_float::<Flex, _>(&mut |leaf| leaf.clone());
        assert_eq!(rebound.metadata, 42);
        assert_eq!(rebound.label, "linear");
        assert_eq!(rebound.weights.shape().num_elements(), 2);
    }

    #[test]
    fn zip_float_pairs_leaves() {
        let lhs = (tensor(&[1.0, 2.0]), tensor(&[3.0]));
        let rhs = (tensor(&[10.0, 20.0]), tensor(&[30.0]));

        let sum =
            lhs.zip_float::<Flex, Flex, _>(&rhs, &mut |a, b| Flex::float_add(a.clone(), b.clone()));
        let values = sum.0.into_data_sync().try_to_vec::<f32>().unwrap();
        assert_eq!(values, vec![11.0, 22.0]);
    }

    #[test]
    fn primitive_scalars_have_no_leaves() {
        assert_eq!(PyTree::<Flex>::num_leaves(&42usize), 0);
        assert_eq!(PyTree::<Flex>::num_leaves(&3.5f64), 0);
        assert_eq!(PyTree::<Flex>::num_leaves(&true), 0);
        assert_eq!(PyTree::<Flex>::num_leaves(&"metadata".to_string()), 0);
    }

    #[test]
    fn skipped_field_requires_b_independence() {
        #[derive(Clone, PyTree)]
        struct WithMetadata<B: burn_backend::Backend> {
            w: FuncTensor<B, 1>,
            #[pytree(skip)]
            step: usize,
        }
        let _ = WithMetadata::<Flex> {
            w: tensor(&[1.0]),
            step: 5,
        };
    }

    #[test]
    fn derive_grad_struct() {
        let device = FlexDevice;
        let params = Params::<Flex> {
            w: FuncTensor::from_data(TensorData::from([2.0f32]), &device),
            b: FuncTensor::from_data(TensorData::from([1.0f32]), &device),
            metadata: 7,
        };

        let f = |p: &Params<Autodiff<Flex>>| {
            let x =
                FuncTensor::<Autodiff<Flex>, 1>::from_data(TensorData::from([3.0f32]), &FlexDevice);
            (p.w.clone() * x + p.b.clone()).sum()
        };

        let grads = grad::<Flex, _, _>(f)(&params);

        assert_eq!(
            grads.w.into_data_sync().try_to_vec::<f32>().unwrap(),
            vec![3.0]
        );
        assert_eq!(
            grads.b.into_data_sync().try_to_vec::<f32>().unwrap(),
            vec![1.0]
        );
        assert_eq!(grads.metadata, 7);
    }

    #[test]
    fn tuple_tree() {
        let device = FlexDevice;
        let params = (
            FuncTensor::<Flex, 1>::from_data(TensorData::from([2.0f32]), &device),
            FuncTensor::<Flex, 1>::from_data(TensorData::from([5.0f32]), &device),
        );

        let f = |p: &(FuncTensor<Autodiff<Flex>, 1>, FuncTensor<Autodiff<Flex>, 1>)| {
            (p.0.clone() * 2.0f64 + p.1.clone() * 4.0f64).sum()
        };

        let grads = grad::<Flex, _, _>(f)(&params);

        assert_eq!(
            grads.0.into_data_sync().try_to_vec::<f32>().unwrap(),
            vec![2.0]
        );
        assert_eq!(
            grads.1.into_data_sync().try_to_vec::<f32>().unwrap(),
            vec![4.0]
        );
    }

    mod macro_hygiene {
        // Intentionally DO NOT import `use super::*;` or `use burn_func::PyTree;`
        // Verifies that #[derive(PyTree)] expands cleanly using ::burn_func::PyTree
        #[derive(Clone, burn_derive::PyTree)]
        struct IsolatedHygieneStruct<B: burn_func::burn_backend::Backend> {
            val: burn_func::FuncTensor<B, 1>,
        }

        #[test]
        fn macro_hygiene_without_imported_trait() {
            use burn_flex::{Flex, FlexDevice};
            use burn_func::{FuncTensor, PyTree};
            let device = FlexDevice;
            let s = IsolatedHygieneStruct::<Flex> {
                val: FuncTensor::from_data([1.0f32, 2.0], &device),
            };
            assert_eq!(s.num_leaves(), 1);
        }
    }
}
