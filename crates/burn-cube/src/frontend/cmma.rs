use std::marker::PhantomData;

use crate::{ir, unexpanded};

use super::{CubeContext, CubeElem, CubeType, ExpandElement};

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

/// Fill the matric with the provided value.
pub fn fill<C: CubeType>(_mat: Matrix<C>, _value: C) {
    unexpanded!()
}
