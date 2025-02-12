use burn_tensor::{Element, ElementConversion};
use cubecl::{
    linalg::matmul::{kernels::tiling2d::Tiling2dConfig, Strategy},
    tune::{local_tuner, LocalTuner, TunableSet},
};

use crate::{
    element::FloatElement,
    kernel::{matmul::utils::init_matmul_output, prng::random_like_uniform},
    ops::numeric::empty_device,
    tensor::JitTensor,
    tune_key::JitAutotuneKey,
    JitRuntime, JitTuneId,
};

use super::key::create_key;

fn matmul_input_gen<R: JitRuntime, E: FloatElement>(
    _key: &JitAutotuneKey,
    lhs: &JitTensor<R>,
    rhs: &JitTensor<R>,
    out: &JitTensor<R>,
) -> (JitTensor<R>, JitTensor<R>, JitTensor<R>) {
    let random_bounds: (E, E) = ((-10.0).elem::<E>(), (10.0).elem::<E>());
    let lhs = random_like_uniform(lhs, random_bounds.0, random_bounds.1);
    let rhs = random_like_uniform(rhs, random_bounds.0, random_bounds.1);

    let out = empty_device::<R, E>(out.client.clone(), out.device.clone(), out.shape.clone());

    (lhs, rhs, out)
}

/// Executes autotune on matmul operations
pub fn matmul_autotune<R: JitRuntime, E: FloatElement + Element>(
    lhs: JitTensor<R>,
    rhs: JitTensor<R>,
    out: Option<JitTensor<R>>,
) -> JitTensor<R> {
    let output = out.unwrap_or_else(|| init_matmul_output::<R, E>(&lhs, &rhs));

    let client = lhs.client.clone();

    static TUNER: LocalTuner<JitAutotuneKey, JitTuneId> = local_tuner!();

    let tunables = TunableSet::new(create_key::<R, E>, matmul_input_gen::<R, E>)
        .with_tunable(matmul_tiling2d::<R, E>)
        .with_tunable(matmul_accelerated::<R, E>)
        .with_tunable(matmul_simple::<R, E>);

    TUNER.execute(
        &JitTuneId::new::<R>(&lhs.device),
        &client,
        &tunables,
        (lhs, rhs, output.clone()),
    );

    output
}

fn matmul_accelerated<R: JitRuntime, E: FloatElement>(
    lhs: JitTensor<R>,
    rhs: JitTensor<R>,
    out: JitTensor<R>,
) -> Result<(), String> {
    cubecl::linalg::matmul::launch_ref::<R, E>(
        &Strategy::Standard,
        &lhs.client,
        &lhs.as_handle_ref(),
        &rhs.as_handle_ref(),
        &out.as_handle_ref(),
    )
    .map_err(|err| format!("{err:?}"))
}

fn matmul_tiling2d<R: JitRuntime, E: FloatElement>(
    lhs: JitTensor<R>,
    rhs: JitTensor<R>,
    out: JitTensor<R>,
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

fn matmul_simple<R: JitRuntime, E: FloatElement>(
    lhs: JitTensor<R>,
    rhs: JitTensor<R>,
    out: JitTensor<R>,
) -> Result<(), String> {
    cubecl::linalg::matmul::launch_ref::<R, E>(
        &Strategy::Simple,
        &lhs.client,
        &lhs.as_handle_ref(),
        &rhs.as_handle_ref(),
        &out.as_handle_ref(),
    )
    .map_err(|err| format!("{err:?}"))
}
