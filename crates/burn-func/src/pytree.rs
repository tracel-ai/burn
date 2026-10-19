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
