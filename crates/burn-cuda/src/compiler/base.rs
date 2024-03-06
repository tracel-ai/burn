use std::marker::PhantomData;

use crate::element::{FloatElement, IntElement};

#[derive(new, Clone, Debug, Default)]
pub struct CudaCompiler<F: FloatElement, I: IntElement> {
    _f: PhantomData<F>,
    _i: PhantomData<I>,
}

impl<F: FloatElement, I: IntElement> burn_jit::Compiler for CudaCompiler<F, I> {
    type Representation = String;
    type Float = f32;
    type Int = i32;

    type FullPrecisionCompiler = CudaCompiler<f32, i32>;

    fn compile(shader: burn_jit::gpu::ComputeShader) -> Self::Representation {
        todo!()
    }

    fn elem_size(elem: burn_jit::gpu::Elem) -> usize {
        todo!()
    }
}
