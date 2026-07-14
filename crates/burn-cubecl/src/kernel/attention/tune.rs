use crate::{
    CubeRuntime, CubeTuneId,
    kernel::attention::{AttentionStrategy, attention},
    tensor::CubeTensor,
};
use burn_backend::cubecl::dtype_to_elem_type;
use burn_backend::ops::AttentionModuleOptions;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, TuneGroup, local_tuner};
use cubek::attention::forward::{
    launch::AttentionAutotuneKey, routines::blackbox_accelerated::BlackboxAcceleratedStrategy,
};

/// Executes autotune on attention operations
pub fn attention_autotune<R: CubeRuntime>(
    query: CubeTensor<R>,
    key: CubeTensor<R>,
    value: CubeTensor<R>,
    mask: Option<CubeTensor<R>>,
    attn_bias: Option<CubeTensor<R>>,
    options: AttentionModuleOptions,
) -> CubeTensor<R> {
    let client = query.client.clone();
    let bounds_client = client.clone();

    static TUNER: LocalTuner<AttentionAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TUNER.init(move || {
        const PRIORITY_MAX: i8 = 3;
        const PRIORITY_MIN: i8 = 0;

        let flash_attention =
            TuneGroup::<AttentionAutotuneKey>::new("flash_attention", |_key| PRIORITY_MAX);

        let fallback = TuneGroup::<AttentionAutotuneKey>::new("fallback", |key| {
            // The fallback materializes the full (total_batches, seq_q, seq_kv)
            // score matrix, which the flash kernels never allocate — and even
            // *benchmarking* it pays that allocation. Let it compete only while
            // that matrix is no bigger than an activation the model already
            // produces — `[batch, seq_kv, d_model]`, a full-head K/V-sized
            // tensor — so it fits a memory budget sized for the model's own
            // activations. With `total_batches = batch · heads` and
            // `d_model = heads · head_dim`, the bound reduces to
            // `seq_q <= head_dim`: decode-like and short-chunk shapes qualify,
            // long-prefill shapes never do; for those it is strictly a last
            // resort for shapes no flash kernel can run, since an
            // O(seq_q · seq_kv) spike can exceed a flash-sized memory budget.
            if key.seq_q > key.head_dim {
                PRIORITY_MIN
            } else {
                PRIORITY_MAX
            }
        });

        let mut set = TunableSet::new(create_key::<R>, input_gen::<R>);
        set = set.with_bounds(super::bounds::create_attention_bounds(&bounds_client));

        // First entry should always work, since it is considered the fallback.
        set = set.with(
            Tunable::new(
                "fallback",
                |(query, key, value, mask, attn_bias, options)| {
                    attention::<R>(
                        query,
                        key,
                        value,
                        mask,
                        attn_bias,
                        options,
                        AttentionStrategy::Fallback,
                    )
                    .map_err(|err| std::format!("{err:?}"))
                },
            )
            .group(&fallback, |_key| PRIORITY_MAX),
        );

        let seq_q = 1;
        let seq_kv = 1;
        for num_planes in [2, 4, 8] {
            let name = format!("blackbox_accelerated_{num_planes}_planes_p_{seq_q}-{seq_kv}");
            set = set.with(
                Tunable::new(
                    &name,
                    move |(query, key, value, mask, attn_bias, options)| {
                        attention::<R>(
                            query,
                            key,
                            value,
                            mask,
                            attn_bias,
                            options,
                            AttentionStrategy::FlashBlackboxAccelerated(
                                BlackboxAcceleratedStrategy {
                                    num_planes,
                                    seq_q,
                                    seq_kv,
                                },
                            ),
                        )
                        .map_err(|err| std::format!("{err:?}"))
                    },
                )
                .group(&flash_attention, |_key| PRIORITY_MAX),
            );
        }

        set = set.with(
            Tunable::new("unit", |(query, key, value, mask, attn_bias, options)| {
                attention::<R>(
                    query,
                    key,
                    value,
                    mask,
                    attn_bias,
                    options,
                    AttentionStrategy::FlashUnit,
                )
                .map_err(|err| std::format!("{err:?}"))
            })
            .group(&flash_attention, |_key| PRIORITY_MIN),
        );

        set
    });

    TUNER.execute(
        &CubeTuneId::new(&client, &query.device),
        &client,
        tunables,
        (query, key, value, mask, attn_bias, options),
    )
}

#[allow(clippy::type_complexity)]
fn create_key<R: CubeRuntime>(
    (query, key, value, mask, _attn_bias, _options): &(
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        Option<CubeTensor<R>>,
        Option<CubeTensor<R>>,
        AttentionModuleOptions,
    ),
) -> AttentionAutotuneKey {
    let total_batches = query.meta.shape[0] * query.meta.shape[1];
    let seq_q = query.meta.shape[2];
    let head_dim = query.meta.shape[3];
    let seq_kv = value.meta.shape[2];
    let val_dim = value.meta.shape[3];

    AttentionAutotuneKey::generate(
        dtype_to_elem_type(query.dtype),
        dtype_to_elem_type(key.dtype),
        dtype_to_elem_type(value.dtype),
        dtype_to_elem_type(query.dtype),
        total_batches,
        seq_q,
        head_dim,
        seq_kv,
        val_dim,
        mask.is_some(),
    )
}

#[allow(clippy::type_complexity)]
fn input_gen<R: CubeRuntime>(
    _key: &AttentionAutotuneKey,
    (query, key, value, mask, attn_bias, options): &(
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        Option<CubeTensor<R>>,
        Option<CubeTensor<R>>,
        AttentionModuleOptions,
    ),
) -> (
    CubeTensor<R>,
    CubeTensor<R>,
    CubeTensor<R>,
    Option<CubeTensor<R>>,
    Option<CubeTensor<R>>,
    AttentionModuleOptions,
) {
    (
        query.clone(),
        key.clone(),
        value.clone(),
        mask.clone(),
        attn_bias.clone(),
        *options,
    )
}
