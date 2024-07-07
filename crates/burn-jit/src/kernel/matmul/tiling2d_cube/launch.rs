use core::panic;
use std::cmp::max;

use burn_compute::client::ComputeClient;
use burn_cube::{compute::CubeCount, frontend::TensorArg, ir::CubeDim, Compiler, Runtime};

use crate::{
    kernel::{
        into_contiguous,
        matmul::{
            config::{tiling2d_cube_count, tiling2d_cube_dim, CubeTiling2dConfig},
            tiling2d_cube::direct::{
                base::{BlockCheck, DirectLoader, VectorReaderEnum},
                horizontal_block_check::HorizontalBlockCheckLoad,
                unchecked_block::UncheckedBlockLoad,
                vector_reader::{MatchingVectorReader, UnmatchingVectorReader},
                vertical_block_check::VerticalBlockCheckLoad,
                whole_block_check::WholeBlockCheckLoad,
            },
            Tiling2dConfig,
        },
    },
    tensor::{JitTensor, MemoryLayout},
    FloatElement, JitRuntime,
};

use super::base::tiling2d_cube_launch;

/// Matrix multiplication using tiling 2d algorithm
pub fn matmul_tiling_2d_cube<'a, R: JitRuntime, E: FloatElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    out: JitTensor<R, E, D>,
    config: Tiling2dConfig,
) -> JitTensor<R, E, D> {
    assert!(
        config.block_size_k * max(config.block_size_m, config.block_size_n)
            <= <R::Compiler as Compiler>::max_shared_memory_size(),
        "Shared memory limit will be busted. "
    );

    let m = lhs.shape.dims[D - 2];
    let k = lhs.shape.dims[D - 1];
    let n = rhs.shape.dims[D - 1];

    let client = lhs.client.clone();

    let check_layout = |tensor: JitTensor<R, E, D>| match tensor.memory_layout() {
        MemoryLayout::Contiguous => (tensor, false),
        MemoryLayout::MildlyPermuted {
            transposed,
            batch_swap: _,
        } => (tensor, transposed),
        MemoryLayout::HighlyPermuted => (into_contiguous(tensor), false),
    };
    let (lhs, lhs_transposed) = check_layout(lhs);
    let (rhs, rhs_transposed) = check_layout(rhs);

    let vectorization = |shape: usize| {
        [4, 2]
            .into_iter()
            .filter(|v| shape % v == 0)
            .map(|v| v as u8)
            .next()
            .unwrap_or(1)
    };

    let out_vectorization = vectorization(n);

    let cube_count = tiling2d_cube_count::<R, D>(&out.shape, &config);
    let cube_dim = tiling2d_cube_dim(&config);
    let cube_config = CubeTiling2dConfig::new(&config, m, k, n, lhs_transposed, rhs_transposed);

    let direct = true;
    if direct {
        let (lhs_check_vertical, lhs_check_horizontal, lhs_vectorization) = if lhs_transposed {
            let vectorization = vectorization(m);
            (
                cube_config.check_k_bounds,
                cube_config.check_m_bounds,
                vectorization,
            )
        } else {
            (cube_config.check_m_bounds, cube_config.check_k_bounds, 1)
        };

        let lhs_vector_reader = match lhs_vectorization {
            4 => VectorReaderEnum::Matching,
            _ => VectorReaderEnum::Unmatching,
        };
        let lhs_block_check = match (lhs_check_vertical, lhs_check_horizontal) {
            (true, true) => BlockCheck::Whole,
            (true, false) => BlockCheck::Vertical,
            (false, true) => BlockCheck::Horizontal,
            (false, false) => BlockCheck::Unchecked,
        };

        let (rhs_check_vertical, rhs_check_horizontal, rhs_vectorization) = if rhs_transposed {
            (cube_config.check_n_bounds, cube_config.check_k_bounds, 1)
        } else {
            let vectorization = vectorization(n);
            (
                cube_config.check_k_bounds,
                cube_config.check_n_bounds,
                vectorization,
            )
        };
        let rhs_vector_reader = match rhs_vectorization {
            4 => VectorReaderEnum::Matching,
            _ => VectorReaderEnum::Unmatching,
        };

        let rhs_block_check = match (rhs_check_vertical, rhs_check_horizontal) {
            (true, true) => BlockCheck::Whole,
            (true, false) => BlockCheck::Vertical,
            (false, true) => BlockCheck::Horizontal,
            (false, false) => BlockCheck::Unchecked,
        };

        direct_dispatch::<R, E, D>(
            client,
            cube_count,
            cube_dim,
            &lhs,
            lhs_vectorization,
            &rhs,
            rhs_vectorization,
            &out,
            out_vectorization,
            cube_config,
            lhs_block_check,
            lhs_vector_reader,
            rhs_block_check,
            rhs_vector_reader,
        );
    } else {
        // let lhs_vectorization = match lhs_transposed {
        //     true => vectorization(m),
        //     false => vectorization(k),
        // };
        // let mut rhs_vectorization = match rhs_transposed {
        //     true => vectorization(k),
        //     false => vectorization(n),
        // };

        // tiling2d_cube_launch_macro!(
        //     E::FloatPrimitive,
        //     TileLoader,
        //     R,
        //     client,
        //     cube_count,
        //     cube_dim,
        //     TensorArg::<'a, R>::vectorized(
        //         lhs_vectorization,
        //         &lhs.handle,
        //         &lhs.strides,
        //         &lhs.shape.dims
        //     ),
        //     TensorArg::<'a, R>::vectorized(
        //         rhs_vectorization,
        //         &rhs.handle,
        //         &rhs.strides,
        //         &rhs.shape.dims
        //     ),
        //     TensorArg::<'a, R>::vectorized(
        //         out_vectorization,
        //         &out.handle,
        //         &out.strides,
        //         &out.shape.dims
        //     ),
        //     cube_config
        // );
    }
    out
}

fn direct_dispatch<'a, R: JitRuntime, E: FloatElement, const D: usize>(
    client: ComputeClient<<R as JitRuntime>::JitServer, <R as Runtime>::Channel>,
    cube_count: CubeCount<<R as JitRuntime>::JitServer>,
    cube_dim: CubeDim,
    lhs: &'a JitTensor<R, E, D>,
    lhs_vectorization: u8,
    rhs: &'a JitTensor<R, E, D>,
    rhs_vectorization: u8,
    out: &'a JitTensor<R, E, D>,
    out_vectorization: u8,
    cube_config: CubeTiling2dConfig,
    lhs_block_check: BlockCheck,
    lhs_vector_reader: VectorReaderEnum,
    rhs_block_check: BlockCheck,
    rhs_vector_reader: VectorReaderEnum,
) {
    let lhs_arg: TensorArg<'a, R> = TensorArg::vectorized(
        lhs_vectorization,
        &lhs.handle,
        &lhs.strides,
        &lhs.shape.dims,
    );
    let rhs_arg: TensorArg<'a, R> = TensorArg::vectorized(
        rhs_vectorization,
        &rhs.handle,
        &rhs.strides,
        &rhs.shape.dims,
    );
    let out_arg: TensorArg<'a, R> = TensorArg::vectorized(
        out_vectorization,
        &out.handle,
        &out.strides,
        &out.shape.dims,
    );

    macro_rules! tiling2d_cube_launch_macro {
        (
            $l:ty,
            $r:ty,
        ) => {
            tiling2d_cube_launch::<E::FloatPrimitive, DirectLoader<E::FloatPrimitive, $l, $r>, R>(
                client,
                cube_count,
                cube_dim,
                lhs_arg,
                rhs_arg,
                out_arg,
                cube_config,
            )
        };
    }

    match (
        lhs_block_check,
        lhs_vector_reader,
        rhs_block_check,
        rhs_vector_reader,
    ) {
        (
            BlockCheck::Unchecked,
            VectorReaderEnum::Unmatching,
            BlockCheck::Unchecked,
            VectorReaderEnum::Matching,
        ) => {
            //OUI
            tiling2d_cube_launch_macro!(
                UncheckedBlockLoad<UnmatchingVectorReader>,
                UncheckedBlockLoad<MatchingVectorReader>,
            )
        }
        (
            BlockCheck::Horizontal,
            VectorReaderEnum::Unmatching,
            BlockCheck::Whole,
            VectorReaderEnum::Matching,
        ) => {
            //OUI
            tiling2d_cube_launch_macro!(
                HorizontalBlockCheckLoad<UnmatchingVectorReader>,
                WholeBlockCheckLoad<MatchingVectorReader>,
            )
        }
        (
            BlockCheck::Whole,
            VectorReaderEnum::Unmatching,
            BlockCheck::Whole,
            VectorReaderEnum::Matching,
        ) => {
            //OUI
            tiling2d_cube_launch_macro!(
                WholeBlockCheckLoad<UnmatchingVectorReader>,
                WholeBlockCheckLoad<MatchingVectorReader>,
            )
        }
        (
            BlockCheck::Unchecked,
            VectorReaderEnum::Unmatching,
            BlockCheck::Horizontal,
            VectorReaderEnum::Matching,
        ) => {
            // OUI
            tiling2d_cube_launch_macro!(
                UncheckedBlockLoad<UnmatchingVectorReader>,
                HorizontalBlockCheckLoad<MatchingVectorReader>,
            )
        }
        (
            BlockCheck::Whole,
            VectorReaderEnum::Unmatching,
            BlockCheck::Whole,
            VectorReaderEnum::Unmatching,
        ) => {
            // NON
            tiling2d_cube_launch_macro!(
                WholeBlockCheckLoad<UnmatchingVectorReader>,
                WholeBlockCheckLoad<UnmatchingVectorReader>,
            )
        }
        (
            BlockCheck::Horizontal,
            VectorReaderEnum::Matching,
            BlockCheck::Unchecked,
            VectorReaderEnum::Matching,
        ) => {
            //NON
            tiling2d_cube_launch_macro!(
                HorizontalBlockCheckLoad<MatchingVectorReader>,
                UncheckedBlockLoad<MatchingVectorReader>,
            )
        }
        (
            BlockCheck::Unchecked,
            VectorReaderEnum::Unmatching,
            BlockCheck::Vertical,
            VectorReaderEnum::Unmatching,
        ) => {
            //NON
            tiling2d_cube_launch_macro!(
                UncheckedBlockLoad<UnmatchingVectorReader>,
                VerticalBlockCheckLoad<UnmatchingVectorReader>,
            )
        }
        (
            BlockCheck::Vertical,
            VectorReaderEnum::Matching,
            BlockCheck::Horizontal,
            VectorReaderEnum::Unmatching,
        ) => {
            tiling2d_cube_launch_macro!(
                VerticalBlockCheckLoad<MatchingVectorReader>,
                HorizontalBlockCheckLoad<UnmatchingVectorReader>,
            )
        }
        (
            BlockCheck::Vertical,
            VectorReaderEnum::Unmatching,
            BlockCheck::Unchecked,
            VectorReaderEnum::Unmatching,
        ) => {
            // NON
            tiling2d_cube_launch_macro!(
                VerticalBlockCheckLoad<UnmatchingVectorReader>,
                UncheckedBlockLoad<UnmatchingVectorReader>,
            )
        }
        _ => todo!(
            "{:?},{:?},{:?},{:?}",
            lhs_block_check,
            lhs_vector_reader,
            rhs_block_check,
            rhs_vector_reader
        ),
    }
}
