use burn_tensor::{Element, ElementConversion};
use cubecl::{
    linalg::matmul::{
        Strategy, SyncLoadingStrategy, kernels::tiling2d::Tiling2dConfig,
        tune_key::MatmulAutotuneKey,
    },
    tune::{LocalTuner, TunableSet, local_tuner},
};

use crate::{
    CubeRuntime, CubeTuneId,
    element::FloatElement,
    kernel::{matmul::utils::init_matmul_output, prng::random_like_uniform},
    ops::numeric::empty_device,
    tensor::CubeTensor,
};

fn matmul_input_gen<R: CubeRuntime, E: FloatElement>(
    _key: &MatmulAutotuneKey,
    lhs: &CubeTensor<R>,
    rhs: &CubeTensor<R>,
    out: &CubeTensor<R>,
) -> (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>) {
    let random_bounds: (E, E) = ((-10.0).elem::<E>(), (10.0).elem::<E>());
    let lhs = random_like_uniform(lhs, random_bounds.0, random_bounds.1);
    let rhs = random_like_uniform(rhs, random_bounds.0, random_bounds.1);

    let out = empty_device::<R, E>(out.client.clone(), out.device.clone(), out.shape.clone());

    (lhs, rhs, out)
}

/// Executes autotune on matmul operations
pub fn matmul_autotune<R: CubeRuntime, E: FloatElement + Element>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: Option<CubeTensor<R>>,
) -> CubeTensor<R> {
    let output = out.unwrap_or_else(|| init_matmul_output::<R, E>(&lhs, &rhs));

    let client = lhs.client.clone();

    static TUNER: LocalTuner<MatmulAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TunableSet::new(create_key::<R, E>, matmul_input_gen::<R, E>)
        .with_tunable(matmul_tiling2d::<R, E>)
        .with_tunable(matmul_accelerated::<R, E>)
        .with_tunable(matmul_naive::<R, E>);

    TUNER.execute(
        &CubeTuneId::new::<R>(&lhs.client, &lhs.device),
        &client,
        &tunables,
        (lhs, rhs, output.clone()),
    );

    output
}

fn create_key<R: CubeRuntime, E: FloatElement>(
    lhs: &CubeTensor<R>,
    rhs: &CubeTensor<R>,
    _out: &CubeTensor<R>,
) -> MatmulAutotuneKey {
    MatmulAutotuneKey::generate(
        &lhs.shape.dims,
        &rhs.shape.dims,
        &lhs.strides,
        &rhs.strides,
        E::dtype().into(),
        E::dtype().into(),
        E::dtype().into(),
    )
}

fn matmul_accelerated<R: CubeRuntime, E: FloatElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    cubecl::linalg::matmul::launch_ref::<R, E>(
        &Strategy::Simple(SyncLoadingStrategy::Cyclic),
        &lhs.client,
        &lhs.as_handle_ref(),
        &rhs.as_handle_ref(),
        &out.as_handle_ref(),
    )
    .map_err(|err| format!("{err:?}"))
}

fn matmul_tiling2d<R: CubeRuntime, E: FloatElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    cubecl::linalg::matmul::launch_ref::<R, E>(
        &Strategy::Tiling2D(Tiling2dConfig::default()),
        &lhs.client,
        &lhs.as_handle_ref(),
        &rhs.as_handle_ref(),
        &out.as_handle_ref(),
    )
    .map_err(|err| format!("{err:?}"))
}

fn matmul_naive<R: CubeRuntime, E: FloatElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    cubecl::linalg::matmul::launch_ref::<R, E>(
        &Strategy::Naive,
        &lhs.client,
        &lhs.as_handle_ref(),
        &rhs.as_handle_ref(),
        &out.as_handle_ref(),
    )
    .map_err(|err| format!("{err:?}"))
}
