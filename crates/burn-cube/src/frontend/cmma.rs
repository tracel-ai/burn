use std::marker::PhantomData;

use crate::{
    ir::{self, Operation},
    unexpanded,
};

use super::{Array, CubeContext, CubeElem, CubeType, ExpandElement, UInt};

pub use ir::{MatrixIdent, MatrixLayout};

pub struct Matrix<C: CubeType> {
    _c: PhantomData<C>,
}

impl<C: CubeType> CubeType for Matrix<C> {
    type ExpandType = ExpandElement;
}

impl<C: CubeElem> Matrix<C> {
    #[allow(unused_variables)]
    pub fn new(ident: MatrixIdent, m: u8, n: u8, k: u8, layout: Option<MatrixLayout>) -> Self {
        Matrix { _c: PhantomData }
    }

    pub fn new_expand(
        context: &mut CubeContext,
        ident: MatrixIdent,
        m: u8,
        n: u8,
        k: u8,
        layout: Option<MatrixLayout>,
    ) -> ExpandElement {
        context.create_matrix(ir::Matrix {
            ident,
            m,
            n,
            k,
            elem: C::as_elem(),
            layout,
        })
    }
}

/// Fill the matrix with the provided value.
#[allow(unused_variables)]
pub fn fill<C: CubeType>(mat: Matrix<C>, value: C) {
    unexpanded!()
}

/// Expand method of [fill].
pub fn fill_expand<C: CubeType>(
    context: &mut CubeContext,
    mat: ExpandElement,
    value: ExpandElement,
) {
    context.register(Operation::CoopMma(ir::CoopMma::Fill {
        mat: *mat,
        value: *value,
    }));
}

/// Load the matrix with the provided array using the stride.
#[allow(unused_variables)]
pub fn load<C: CubeType>(mat: Matrix<C>, value: &Array<C>, stride: UInt) {
    unexpanded!()
}

/// Expand method of [load].
#[allow(unused_variables)]
pub fn load_expand<C: CubeType>(
    context: &mut CubeContext,
    mat: ExpandElement,
    value: ExpandElement,
    stride: ExpandElement,
) {
    context.register(Operation::CoopMma(ir::CoopMma::Load {
        mat: *mat,
        value: *value,
        stride: *value,
    }));
}
