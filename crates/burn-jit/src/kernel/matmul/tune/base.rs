use burn_tensor::{Element, ElementConversion};
use cubecl::{
    ir::{Elem, FloatKind},
    linalg::matmul::{kernels::tiling2d::Tiling2dConfig, Strategy},
    tune,
    tune::{local_tuner, tune_with, LocalTuner},
    Feature,
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

#[tune(
    operations(matmul_tiling2d, matmul_accelerated, matmul_simple),
    create_key = create_key::<R, E>,
    should_run = should_run
)]
fn matmul_ops<R: JitRuntime, E: FloatElement>(
    key: JitAutotuneKey,
    lhs: JitTensor<R>,
    rhs: JitTensor<R>,
    out: JitTensor<R>,
) {
    let random_bounds: (E, E) = ((-10.0).elem::<E>(), (10.0).elem::<E>());
    let lhs = random_like_uniform(lhs, random_bounds.0, random_bounds.1);
    let rhs = random_like_uniform(rhs, random_bounds.0, random_bounds.1);

    let out = empty_device::<R, E>(out.client.clone(), out.device.clone(), out.shape.clone());

    tune_with!(lhs, rhs, out)
}

fn should_run<R: JitRuntime, E: FloatElement>(
    op: &MatmulOps<R, E>,
    _key: &JitAutotuneKey,
    index: usize,
) -> bool {
    match index {
        // Accelerated
        // TODO: Add way to query actual requirements from cubecl
        1 => op.lhs.client.properties().feature_enabled(Feature::Cmma {
            a: Elem::Float(FloatKind::F16),
            b: Elem::Float(FloatKind::F16),
            c: Elem::Float(FloatKind::F32),
            m: 16,
            k: 16,
            n: 16,
        }),
        _ => true,
    }
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

    TUNER.execute(
        &JitTuneId::new::<R>(&lhs.device),
        &client,
        Box::new(MatmulOps::<R, E>::new(lhs, rhs, output.clone())),
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
