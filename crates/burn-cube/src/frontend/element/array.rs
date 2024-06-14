use std::marker::PhantomData;

use crate::{
    frontend::{indexation::Index, CubeContext, CubeElem, CubeType},
    ir::{Item, Variable},
};

use super::{ExpandElement, Init};

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
}
