use std::marker::PhantomData;

use crate::{
    ir::{self, Operation},
    unexpanded,
};

use super::{Array, CubeContext, CubePrimitive, CubeType, ExpandElement, Init, UInt};

pub use ir::{MatrixIdent, MatrixLayout};

pub struct Matrix<C: CubeType> {
    _c: PhantomData<C>,
}

#[derive(Clone)]
pub struct MatrixExpand {
    elem: ExpandElement,
}

impl<C: CubeType> CubeType for Matrix<C> {
    type ExpandType = MatrixExpand;
}

impl Init for MatrixExpand {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

impl<C: CubePrimitive> Matrix<C> {
    #[allow(unused_variables)]
    pub fn new(ident: MatrixIdent, m: u8, n: u8, k: u8, layout: MatrixLayout) -> Self {
        Matrix { _c: PhantomData }
    }

    pub fn new_expand(
        context: &mut CubeContext,
        ident: MatrixIdent,
        m: u8,
        n: u8,
        k: u8,
        layout: MatrixLayout,
    ) -> MatrixExpand {
        let elem = context.create_matrix(ir::Matrix {
            ident,
            m,
            n,
            k,
            elem: C::as_elem(),
            layout,
        });
        MatrixExpand { elem }
    }
}

/// Fill the matrix with the provided value.
#[allow(unused_variables)]
pub fn fill<C: CubeType>(mat: &mut Matrix<C>, value: C) {
    unexpanded!()
}

/// Expand method of [fill].
pub fn fill_expand<C: CubeType>(
    context: &mut CubeContext,
    mat: MatrixExpand,
    value: ExpandElement,
) {
    context.register(Operation::CoopMma(ir::CoopMma::Fill {
        mat: *mat.elem,
        value: *value,
    }));
}

/// Load the matrix with the provided array using the stride.
#[allow(unused_variables)]
pub fn load<C: CubeType>(mat: &mut Matrix<C>, value: &Array<C>, stride: UInt) {
    unexpanded!()
}

/// Expand method of [load].
#[allow(unused_variables)]
pub fn load_expand<C: CubeType>(
    context: &mut CubeContext,
    mat: MatrixExpand,
    value: ExpandElement,
    stride: ExpandElement,
) {
    context.register(Operation::CoopMma(ir::CoopMma::Load {
        mat: *mat.elem,
        value: *value,
        stride: *stride,
    }));
}

#[allow(unused_variables)]
pub fn store<C: CubePrimitive>(
    output: &mut Array<C>,
    mat: &Matrix<C>,
    stride: UInt,
    layout: MatrixLayout,
) {
    unexpanded!()
}

#[allow(unused_variables)]
pub fn store_expand<C: CubePrimitive>(
    context: &mut CubeContext,
    output: ExpandElement,
    mat: MatrixExpand,
    stride: ExpandElement,
    layout: MatrixLayout,
) {
    context.register(Operation::CoopMma(ir::CoopMma::Store {
        output: *output,
        mat: *mat.elem,
        stride: *stride,
        layout,
    }));
}

#[allow(unused_variables)]
pub fn execute<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive, D: CubePrimitive>(
    mat_a: &Matrix<A>,
    mat_b: &Matrix<B>,
    mat_c: &Matrix<C>,
    mat_d: &Matrix<D>,
) {
    unexpanded!()
}

pub fn execute_expand<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive, D: CubePrimitive>(
    context: &mut CubeContext,
    a: MatrixExpand,
    b: MatrixExpand,
    c: MatrixExpand,
    d: MatrixExpand,
) {
    context.register(Operation::CoopMma(ir::CoopMma::Execute {
        mat_a: *a.elem,
        mat_b: *b.elem,
        mat_c: *c.elem,
        mat_d: *d.elem,
    }));
}
