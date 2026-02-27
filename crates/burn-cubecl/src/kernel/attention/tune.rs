use crate::{
    CubeRuntime, CubeTuneId,
    kernel::attention::{AttentionStrategy, attention},
    tensor::CubeTensor,
};
use burn_backend::ops::AttentionModuleOptions;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, TuneGroup, local_tuner};
use cubek::attention::{definition::AttentionSetupError, launch::AttentionAutotuneKey};

/// Executes autotune on attention operations
pub fn attention_autotune<R: CubeRuntime>(
    query: CubeTensor<R>,
    key: CubeTensor<R>,
    value: CubeTensor<R>,
    mask: Option<CubeTensor<R>>,
    attn_bias: Option<CubeTensor<R>>,
    options: AttentionModuleOptions,
    out: CubeTensor<R>,
) -> Result<CubeTensor<R>, AttentionSetupError> {
    let client = query.client.clone();

    static TUNER: LocalTuner<AttentionAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TUNER.init(|| {
        const PRIORITY_MAX: i8 = 3;
        const PRIORITY_MIN: i8 = 0;

        let flash_attention =
            TuneGroup::<AttentionAutotuneKey>::new("flash_attention", |_key| PRIORITY_MAX);
        let fallback = TuneGroup::<AttentionAutotuneKey>::new("fallback", |_key| PRIORITY_MIN);

        let mut set = TunableSet::new(create_key::<R>, input_gen::<R>);

        // First entry should always work, since it is considered the fallback.
        set = set.with(
            Tunable::new(
                "fallback",
                |query, key, value, mask, attn_bias, out, options| {
                    attention::<R>(
                        query,
                        key,
                        value,
                        mask,
                        attn_bias,
                        options,
                        &AttentionStrategy::Fallback,
                        Some(out),
                    )
                    .map_err(|err| std::format!("{err:?}"))
                },
            )
            .group(&fallback, |_key| PRIORITY_MAX),
        );

        set = set.with(
            Tunable::new(
                "blackbox_accelerated",
                |query, key, value, mask, attn_bias, out, options| {
                    attention::<R>(
                        query,
                        key,
                        value,
                        mask,
                        attn_bias,
                        options,
                        &AttentionStrategy::FlashBlackboxAccelerated,
                        Some(out),
                    )
                    .map_err(|err| std::format!("{err:?}"))
                },
            )
            .group(&flash_attention, |_key| PRIORITY_MAX),
        );

        set = set.with(
            Tunable::new(
                "unit",
                |query, key, value, mask, attn_bias, out, options| {
                    attention::<R>(
                        query,
                        key,
                        value,
                        mask,
                        attn_bias,
                        options,
                        &AttentionStrategy::FlashUnit,
                        Some(out),
                    )
                    .map_err(|err| std::format!("{err:?}"))
                },
            )
            .group(&flash_attention, |_key| PRIORITY_MIN),
        );

        set
    });

    TUNER.execute(
        &CubeTuneId::new(&client, &query.device),
        &client,
        tunables,
        (query, key, value, mask, attn_bias, out.clone(), options),
    );

    Ok(out)
}

fn create_key<R: CubeRuntime>(
    query: &CubeTensor<R>,
    key: &CubeTensor<R>,
    value: &CubeTensor<R>,
    mask: &Option<CubeTensor<R>>,
    _attn_bias: &Option<CubeTensor<R>>,
    out: &CubeTensor<R>,
    _options: &AttentionModuleOptions,
) -> AttentionAutotuneKey {
    let total_batches = query.meta.shape[0] * query.meta.shape[1];
    let seq_q = query.meta.shape[2];
    let head_dim = query.meta.shape[3];
    let seq_kv = value.meta.shape[2];
    let val_dim = value.meta.shape[3];

    AttentionAutotuneKey::generate(
        query.dtype.into(),
        key.dtype.into(),
        value.dtype.into(),
        out.dtype.into(),
        total_batches,
        seq_q,
        head_dim,
        seq_kv,
        val_dim,
        mask.is_some(),
    )
}

#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
fn input_gen<R: CubeRuntime>(
    _key: &AttentionAutotuneKey,
    query: &CubeTensor<R>,
    key: &CubeTensor<R>,
    value: &CubeTensor<R>,
    mask: &Option<CubeTensor<R>>,
    attn_bias: &Option<CubeTensor<R>>,
    out: &CubeTensor<R>,
    options: &AttentionModuleOptions,
) -> (
    CubeTensor<R>,
    CubeTensor<R>,
    CubeTensor<R>,
    Option<CubeTensor<R>>,
    Option<CubeTensor<R>>,
    CubeTensor<R>,
    AttentionModuleOptions,
) {
    (
        query.clone(),
        key.clone(),
        value.clone(),
        mask.clone(),
        attn_bias.clone(),
        out.copy(),
        *options,
    )
}
