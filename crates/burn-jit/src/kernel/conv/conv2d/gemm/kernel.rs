use std::marker::PhantomData;

use cubecl::{
    ir::MatrixLayout,
    prelude::{CubePrimitive, CubeType},
    CubeDim,
};

use super::{epilogue::Epilogue, swizzle::CubeSwizzle};

pub struct GemmShape {
    m: u32,
    n: u32,
    k: u32,
}

impl GemmShape {
    pub const fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }
}

pub trait GemmKernel {
    type ElementA: CubePrimitive;
    const LAYOUT_A: MatrixLayout;
    type ElementB: CubePrimitive;
    const LAYOUT_B: MatrixLayout;
    type ElementC: CubePrimitive;
    const LAYOUT_C: MatrixLayout;
    type ElementAccumulator: CubePrimitive;
    type ElementCompute: CubePrimitive;
    const OPERATOR_CLASS: ();
    const CUBE_SHAPE: GemmShape;
    const PLANE_SHAPE: GemmShape;
    const INSTRUCTION_SHAPE: GemmShape;
    type CubeSwizzle: CubeSwizzle;
    type EpilogueOutputOp: Epilogue;
    const STAGES: u32;
    const CONV_DIM: u32;

    type Arguments: CubeType;
}

pub struct Conv2dFPropOptimized {}

pub trait ConvAlgorithm<K: GemmKernel> {
    const PLANE_COUNT: u32;

    fn workspace_size(args: &K::Arguments) -> usize {
        let grid_tiled_shape = K::CubeSwizzle::get_tiled_shape();
    }
}

pub struct ImplicitGemmConvolution<K: GemmKernel> {
    _k: PhantomData<K>,
}

impl<K: GemmKernel> ConvAlgorithm for ImplicitGemmConvolution<K> {
    const PLANE_COUNT: u32 = (K::CUBE_SHAPE.m / K::PLANE_SHAPE.m)
        * (K::CUBE_SHAPE.n / K::PLANE_SHAPE.n)
        * (K::CUBE_SHAPE.k / K::PLANE_SHAPE.k);
}
