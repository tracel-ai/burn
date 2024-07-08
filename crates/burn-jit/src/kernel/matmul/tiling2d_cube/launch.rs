use std::cmp::max;

use burn_compute::client::ComputeClient;
use burn_cube::{compute::CubeCount, frontend::TensorArg, ir::CubeDim, Compiler, Runtime};

use crate::{
    kernel::{
        into_contiguous,
        matmul::{
            config::{tiling2d_cube_count, tiling2d_cube_dim, CubeTiling2dConfig},
            tiling2d_cube::direct::{
                horizontal_block_check::HorizontalBlockCheck,
                loader::{BlockCheck, DirectLoader, VectorReaderEnum},
                unchecked_block::UncheckedBlockCheck,
                vector_reader::{MatchingVectorization, UnmatchingVectorization},
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
            (
                cube_config.check_k_bounds,
                cube_config.check_m_bounds,
                vectorization(m),
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
            (
                cube_config.check_k_bounds,
                cube_config.check_n_bounds,
                vectorization(n),
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
        panic!("not available now")
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
            tiling2d_cube_launch::<
                E::FloatPrimitive,
                DirectLoader<E::FloatPrimitive, $l, $r>,
                UncheckedBlockCheck<MatchingVectorization>,
                R,
            >(
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

    println!("{:?}", lhs_block_check);
    println!("{:?}", lhs_vector_reader);
    println!("{:?}", rhs_block_check);
    println!("{:?}", rhs_vector_reader);
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
            tiling2d_cube_launch_macro!(
                UncheckedBlockCheck<UnmatchingVectorization>,
                UncheckedBlockCheck<MatchingVectorization>,
            )
        }
        (
            BlockCheck::Horizontal,
            VectorReaderEnum::Unmatching,
            BlockCheck::Whole,
            VectorReaderEnum::Matching,
        ) => {
            tiling2d_cube_launch_macro!(
                HorizontalBlockCheck<UnmatchingVectorization>,
                WholeBlockCheckLoad<MatchingVectorization>,
            )
        }
        (
            BlockCheck::Whole,
            VectorReaderEnum::Unmatching,
            BlockCheck::Whole,
            VectorReaderEnum::Matching,
        ) => {
            tiling2d_cube_launch_macro!(
                WholeBlockCheckLoad<UnmatchingVectorization>,
                WholeBlockCheckLoad<MatchingVectorization>,
            )
        }
        (
            BlockCheck::Unchecked,
            VectorReaderEnum::Unmatching,
            BlockCheck::Horizontal,
            VectorReaderEnum::Matching,
        ) => {
            tiling2d_cube_launch_macro!(
                UncheckedBlockCheck<UnmatchingVectorization>,
                HorizontalBlockCheck<MatchingVectorization>,
            )
        }
        (
            BlockCheck::Whole,
            VectorReaderEnum::Unmatching,
            BlockCheck::Whole,
            VectorReaderEnum::Unmatching,
        ) => {
            tiling2d_cube_launch_macro!(
                WholeBlockCheckLoad<UnmatchingVectorization>,
                WholeBlockCheckLoad<UnmatchingVectorization>,
            )
        }
        (
            BlockCheck::Horizontal,
            VectorReaderEnum::Matching,
            BlockCheck::Unchecked,
            VectorReaderEnum::Matching,
        ) => {
            tiling2d_cube_launch_macro!(
                HorizontalBlockCheck<MatchingVectorization>,
                UncheckedBlockCheck<MatchingVectorization>,
            )
        }
        (
            BlockCheck::Unchecked,
            VectorReaderEnum::Unmatching,
            BlockCheck::Vertical,
            VectorReaderEnum::Unmatching,
        ) => {
            tiling2d_cube_launch_macro!(
                UncheckedBlockCheck<UnmatchingVectorization>,
                VerticalBlockCheckLoad<UnmatchingVectorization>,
            )
        }
        (
            BlockCheck::Unchecked,
            VectorReaderEnum::Unmatching,
            BlockCheck::Horizontal,
            VectorReaderEnum::Unmatching,
        ) => {
            tiling2d_cube_launch_macro!(
                UncheckedBlockCheck<UnmatchingVectorization>,
                HorizontalBlockCheck<UnmatchingVectorization>,
            )
        }
        (
            BlockCheck::Vertical,
            VectorReaderEnum::Matching,
            BlockCheck::Horizontal,
            VectorReaderEnum::Unmatching,
        ) => {
            tiling2d_cube_launch_macro!(
                VerticalBlockCheckLoad<MatchingVectorization>,
                HorizontalBlockCheck<UnmatchingVectorization>,
            )
        }
        (
            BlockCheck::Vertical,
            VectorReaderEnum::Unmatching,
            BlockCheck::Unchecked,
            VectorReaderEnum::Unmatching,
        ) => {
            tiling2d_cube_launch_macro!(
                VerticalBlockCheckLoad<UnmatchingVectorization>,
                UncheckedBlockCheck<UnmatchingVectorization>,
            )
        }
        (
            BlockCheck::Vertical,
            VectorReaderEnum::Unmatching,
            BlockCheck::Unchecked,
            VectorReaderEnum::Matching,
        ) => {
            tiling2d_cube_launch_macro!(
                VerticalBlockCheckLoad<UnmatchingVectorization>,
                UncheckedBlockCheck<MatchingVectorization>,
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
