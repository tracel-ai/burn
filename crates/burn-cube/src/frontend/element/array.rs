use std::marker::PhantomData;

use crate::{
    frontend::{indexation::Index, CubeContext, CubeElem, CubeType},
    ir::{Item, Variable},
    prelude::{assign, index, index_assign, Comptime},
    unexpanded,
};

use super::{ExpandElement, Init, UInt};

#[derive(Clone, Copy)]
pub struct Array<T: CubeType> {
    _val: PhantomData<T>,
}

#[derive(Clone)]
pub struct ArrayExpand<T: CubeElem> {
    pub val: <T as CubeType>::ExpandType,
}

impl<T: CubeElem> From<ArrayExpand<T>> for ExpandElement {
    fn from(array_expand: ArrayExpand<T>) -> Self {
        array_expand.val
    }
}

impl<T: CubeElem> From<ArrayExpand<T>> for Variable {
    fn from(array_expand: ArrayExpand<T>) -> Self {
        *array_expand.val
    }
}

impl<T: CubeElem> Init for ArrayExpand<T> {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

impl<T: CubeElem> CubeType for Array<T> {
    type ExpandType = ArrayExpand<T>;
}

impl<T: CubeElem + Clone> Array<T> {
    pub fn new<S: Index>(_size: S) -> Self {
        Array { _val: PhantomData }
    }

    pub fn new_expand<S: Index>(
        context: &mut CubeContext,
        size: S,
    ) -> <Self as CubeType>::ExpandType {
        let size = size.value();
        let size = match size {
            crate::ir::Variable::ConstantScalar(val, _) => val as u32,
            _ => panic!("Array need constant initialization value"),
        };
        context.create_local_array(Item::new(T::as_elem()), size)
    }

    pub fn vectorized<S: Index>(_size: S, _vectorization_factor: UInt) -> Self {
        Array { _val: PhantomData }
    }

    pub fn vectorized_expand<S: Index>(
        context: &mut CubeContext,
        size: S,
        vectorization_factor: UInt,
    ) -> <Self as CubeType>::ExpandType {
        let size = size.value();
        let size = match size {
            crate::ir::Variable::ConstantScalar(val, _) => val as u32,
            _ => panic!("Shared memory need constant initialization value"),
        };
        context.create_local_array(
            Item::vectorized(T::as_elem(), vectorization_factor.val as u8),
            size,
        )
    }

    pub fn to_vectorized(self, _vectorization_factor: Comptime<UInt>) -> T {
        unexpanded!()
    }
}

impl<T: CubeElem> ArrayExpand<T> {
    pub fn to_vectorized_expand(
        self,
        context: &mut CubeContext,
        vectorization_factor: UInt,
    ) -> <T as CubeType>::ExpandType {
        let factor = vectorization_factor.val;
        let mut new_var = context.create_local(Item::vectorized(
            T::as_elem(),
            vectorization_factor.val as u8,
        ));
        if vectorization_factor.val == 1 {
            let element = index::expand(context, self.val.clone(), 0);
            assign::expand(context, element, new_var.clone());
        } else {
            for i in 0..factor {
                let element = index::expand(context, self.val.clone(), i);
                new_var = index_assign::expand(context, new_var, i, element);
            }
        }
        new_var
    }
}
