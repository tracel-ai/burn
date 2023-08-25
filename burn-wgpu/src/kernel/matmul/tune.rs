use burn_tensor::{Distribution, Shape, Tensor};
use mem_coalescing::matmul_mem_coalescing;

use crate::{
    benchmark::Benchmark,
    element::{FloatElement, WgpuElement},
    kernel,
    tensor::{WgpuTensor, WgpuTensorDyn},
    tune::{AutoTuneFunction, AutoTuneKey, Execution, KernelFunction, Tunable},
    GraphicsApi, WgpuBackend, WgpuDevice,
};
use std::{marker::PhantomData, sync::Arc};

use super::mem_coalescing;

const TILING_2D_BLOCK_SIZES: [usize; 2] = [64, 128];
const TILING_2D_BLOCK_SIZES_K: [usize; 2] = [16, 32];
const TILING_2D_TILE_SIZES: [usize; 2] = [4, 16];
const MEMORY_COALESCING_WORKGROUP_SIZES: [usize; 3] = [8, 16, 32];

macro_rules! call_dim {
    ($func:expr, $dim:expr, $( $x:expr ),*) => {
        match $dim {
            1 => {
                let tensor: WgpuTensor<E, 1> = $func($($x,)*);
                tensor.into()
            },
            2 => {
                let tensor: WgpuTensor<E, 2> = $func($($x,)*);
                tensor.into()
            },
            3 => {
                let tensor: WgpuTensor<E, 3> = $func($($x,)*);
                tensor.into()
            },
            4 => {
                let tensor: WgpuTensor<E, 4> = $func($($x,)*);
                tensor.into()
            },
            5 => {
                let tensor: WgpuTensor<E, 5> = $func($($x,)*);
                tensor.into()
            },
            6 => {
                let tensor: WgpuTensor<E, 6> = $func($($x,)*);
                tensor.into()
            },
            _ => panic!("Tensors of rank 7 and more can't be autotuned."),
        }
    };
}

macro_rules! tiling2d_tunable {
    ($name:ident, $func:expr) => {
        #[derive(new, Default)]
        struct $name<E: WgpuElement> {
            b_m: usize,
            b_n: usize,
            b_k: usize,
            t_m: usize,
            t_n: usize,
            workgroup_size_x: usize,
            workgroup_size_y: usize,
            _elem: PhantomData<E>,
        }

        impl<E: WgpuElement> KernelFunction for $name<E> {
            type Input = (WgpuTensorDyn<E>, WgpuTensorDyn<E>);
            type Output = WgpuTensorDyn<E>;

            fn call(&self, (lhs, rhs): Self::Input) -> Self::Output {

                #[allow(clippy::too_many_arguments)]
                fn call_dyn<E: WgpuElement, const D: usize>(
                    lhs: WgpuTensorDyn<E>,
                    rhs: WgpuTensorDyn<E>,
                    b_m: usize,
                    b_n: usize,
                    b_k: usize,
                    t_m: usize,
                    t_n: usize,
                    workgroup_size_x: usize,
                    workgroup_size_y: usize,
                ) -> WgpuTensor<E, D> {
                    $func(
                        WgpuTensor::<E, D>::from(lhs),
                        WgpuTensor::<E, D>::from(rhs),
                        b_m,
                        b_n,
                        b_k,
                        t_m,
                        t_n,
                        workgroup_size_x,
                        workgroup_size_y,
                    )
                }

                return call_dim!(
                    call_dyn,
                    lhs.shape.len(),
                    lhs,
                    rhs,
                    self.b_m,
                    self.b_n,
                    self.b_k,
                    self.t_m,
                    self.t_n,
                    self.workgroup_size_x,
                    self.workgroup_size_y
                );
            }

            fn description(&self) -> String {
                format!(
                    "Tiling 2D matmul ({}) - B_M {}, B_N {}, B_K {}, T_M {}, T_N {}, W_X {}, W_X {}",
                    stringify!($name),
                    self.b_m,
                    self.b_n,
                    self.b_k,
                    self.t_m,
                    self.t_n,
                    self.workgroup_size_x,
                    self.workgroup_size_y
                )
            }
        }
    };
}

tiling2d_tunable!(
    Tiling2DContiguousLoad,
    kernel::matmul::contiguous::matmul_tiling_2d
);

tiling2d_tunable!(Tiling2DTileLoad, kernel::matmul::tile::matmul_tiling_2d);
tiling2d_tunable!(
    Tiling2DContiguousLoadVectorized,
    kernel::matmul::contiguous_vectorized::matmul_tiling_2d
);

tiling2d_tunable!(
    Tiling2DTileLoadVectorized,
    kernel::matmul::tile_vectorized::matmul_tiling_2d
);

#[derive(new)]
struct MemoryCoalescing<E: WgpuElement> {
    workgroup_size_x: usize,
    workgroup_size_y: usize,
    _elem: PhantomData<E>,
}

impl<E: WgpuElement> KernelFunction for MemoryCoalescing<E> {
    type Input = (WgpuTensorDyn<E>, WgpuTensorDyn<E>);
    type Output = WgpuTensorDyn<E>;

    fn call(&self, (lhs, rhs): Self::Input) -> Self::Output {
        fn call_dyn<E: WgpuElement, const D: usize>(
            lhs: WgpuTensorDyn<E>,
            rhs: WgpuTensorDyn<E>,
            workgroup_size_x: usize,
            workgroup_size_y: usize,
        ) -> WgpuTensor<E, D> {
            let lhs = WgpuTensor::from(lhs);
            let rhs = WgpuTensor::from(rhs);

            matmul_mem_coalescing::<E, D>(lhs, rhs, workgroup_size_x, workgroup_size_y)
        }

        call_dim!(
            call_dyn,
            lhs.shape.len(),
            lhs,
            rhs,
            self.workgroup_size_x,
            self.workgroup_size_y
        )
    }

    fn description(&self) -> String {
        format!(
            "Memory Coalescing matmul - W_X {}, W_Y {}",
            self.workgroup_size_x, self.workgroup_size_y
        )
    }
}

#[derive(new)]
struct MatmulBenchmark<F: WgpuElement, const D: usize> {
    shape_lhs: Shape<D>,
    shape_rhs: Shape<D>,
    num_repeats: usize,
    matmul: PhantomData<F>,
    func: AutoTuneFunction<(WgpuTensorDyn<F>, WgpuTensorDyn<F>), WgpuTensorDyn<F>>,
}

impl<E, const D: usize, G> Benchmark<G> for MatmulBenchmark<E, D>
where
    E: WgpuElement + FloatElement,
    G: GraphicsApi,
{
    type Args = (WgpuTensorDyn<E>, WgpuTensorDyn<E>);

    fn name(&self) -> String {
        format!("{:?} x {:?}", self.shape_lhs.dims, self.shape_rhs.dims)
    }

    fn num_samples(&self) -> usize {
        5
    }

    fn execute(&self, (lhs, rhs): Self::Args) {
        for _ in 0..self.num_repeats {
            self.func.call((lhs.clone(), rhs.clone()));
        }
    }

    fn prepare(&self, device: &WgpuDevice) -> Self::Args {
        let lhs = Tensor::<WgpuBackend<G, E, i32>, D>::random_device(
            self.shape_lhs.clone(),
            Distribution::Default,
            device,
        );
        let rhs = Tensor::<WgpuBackend<G, E, i32>, D>::random_device(
            self.shape_rhs.clone(),
            Distribution::Default,
            device,
        );

        (lhs.into_primitive().into(), rhs.into_primitive().into())
    }
}

/// Choose the best matmul kernel by using autotuning.
pub fn tune<G: GraphicsApi, E, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D>
where
    E: WgpuElement + FloatElement,
{
    if D > 6 {
        log::debug!("Can't autotune matmul for tensors of rank 7 or more.");
        return kernel::matmul::matmul_tiling_2d_default(lhs, rhs);
    }

    let (shape_lhs, shape_rhs) = calculate_benchmark_shapes(lhs.shape.clone(), rhs.shape.clone());
    let id = AutoTuneKey::new(
        vec![
            shape_lhs.dims[D - 2..].to_vec(),
            shape_rhs.dims[D - 2..].to_vec(),
        ],
        format!("matmul {}", E::type_name()),
        &lhs.context,
    );

    let context = lhs.context.clone();
    let input: (WgpuTensorDyn<E>, WgpuTensorDyn<E>) = (lhs.into(), rhs.into());
    let output: WgpuTensorDyn<E> = match context.tuner.execute(&id, input) {
        Execution::Executed(output) => output,
        Execution::NoCacheFound((lhs, rhs)) => {
            let tunables = matmul_candidates::<G, E, D>(shape_lhs, shape_rhs);

            context
                .tuner
                .tune(id, (lhs, rhs), tunables, &context.device)
        }
    };

    output.into()
}

/// Shape dims are anchored to the closest (on a log scale) power of 2
fn calculate_benchmark_shapes<const D: usize>(
    lhs: Shape<D>,
    rhs: Shape<D>,
) -> (Shape<D>, Shape<D>) {
    let anchor = |a| f32::powf(2., f32::min(f32::round(f32::log(a as f32, 2.)), 12.)) as usize;
    let m = anchor(lhs.dims[D - 2]);
    let k = anchor(lhs.dims[D - 1]);
    let n = anchor(rhs.dims[D - 1]);

    let mut lhs_shape = [1; D];
    lhs_shape[D - 2] = m;
    lhs_shape[D - 1] = k;
    let lhs_shape = Shape::new(lhs_shape);

    let mut rhs_shape = [1; D];
    rhs_shape[D - 2] = k;
    rhs_shape[D - 1] = n;
    let rhs_shape = Shape::new(rhs_shape);

    (lhs_shape, rhs_shape)
}

type MatmulTunable<G, E> = Tunable<G, (WgpuTensorDyn<E>, WgpuTensorDyn<E>), WgpuTensorDyn<E>>;

/// Enumerates all matmul versions that are candidates for autotuning
fn matmul_candidates<G: GraphicsApi, E, const D: usize>(
    shape_lhs: Shape<D>,
    shape_rhs: Shape<D>,
) -> Vec<MatmulTunable<G, E>>
where
    E: WgpuElement + FloatElement,
{
    let matmul_benchmark =
        |func: AutoTuneFunction<(WgpuTensorDyn<E>, WgpuTensorDyn<E>), WgpuTensorDyn<E>>| {
            Tunable::<G, _, _>::new(
                func.clone(),
                Arc::new(MatmulBenchmark::new(
                    shape_lhs.clone(),
                    shape_rhs.clone(),
                    5,
                    func.clone(),
                )),
            )
        };

    let mut candidates = Vec::new();

    // All combinations of tiling 2d parameters are pushed for a grid search
    for block_size in TILING_2D_BLOCK_SIZES {
        for block_size_k in TILING_2D_BLOCK_SIZES_K {
            for tile_size in TILING_2D_TILE_SIZES {
                candidates.push(matmul_benchmark(Arc::new(
                    Tiling2DContiguousLoad::<E>::new(
                        block_size,
                        block_size,
                        block_size_k,
                        tile_size,
                        tile_size,
                        block_size / tile_size,
                        block_size / tile_size,
                    ),
                )));
                candidates.push(matmul_benchmark(Arc::new(
                    Tiling2DContiguousLoadVectorized::<E>::new(
                        block_size,
                        block_size,
                        block_size_k,
                        tile_size,
                        tile_size,
                        block_size / tile_size,
                        block_size / tile_size,
                    ),
                )));
                candidates.push(matmul_benchmark(Arc::new(Tiling2DTileLoad::<E>::new(
                    block_size,
                    block_size,
                    block_size_k,
                    tile_size,
                    tile_size,
                    block_size / tile_size,
                    block_size / tile_size,
                ))));
                candidates.push(matmul_benchmark(Arc::new(
                    Tiling2DTileLoadVectorized::<E>::new(
                        block_size,
                        block_size,
                        block_size_k,
                        tile_size,
                        tile_size,
                        block_size / tile_size,
                        block_size / tile_size,
                    ),
                )));
            }
        }

        // All combinations of tiling 2d parameters are pushed for a grid search
        for workgroup_size in MEMORY_COALESCING_WORKGROUP_SIZES {
            candidates.push(matmul_benchmark(Arc::new(MemoryCoalescing::new(
                workgroup_size,
                workgroup_size,
            ))));
        }
    }
    candidates
}

#[cfg(test)]
mod tests {
    use super::calculate_benchmark_shapes;

    #[test]
    pub fn benchmark_shapes_are_anchored_correctly() {
        let m = f32::powf(2., 8.49) as usize;
        let k = f32::powf(2., 8.51) as usize;
        let n = f32::powf(2., 4.) as usize;
        let lhs_shape = [m, k].into();
        let rhs_shape = [k, n].into();
        let (lhs_shape, rhs_shape) = calculate_benchmark_shapes(lhs_shape, rhs_shape);
        assert_eq!(lhs_shape.dims, [256, 512]);
        assert_eq!(rhs_shape.dims, [512, 16]);
    }
}
