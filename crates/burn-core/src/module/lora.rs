use burn_tensor::{Distribution, FloatDType, Tensor};

use crate::module::{LoraAdapter, Param, ParamGroup, Quantizer, Reparameterizer};

/// A [`Reparameterizer`] that attaches LoRA adapters to 2-D weight parameters.
///
/// It is applied via [`Module::apply_lora`](crate::module::Module::apply_lora).
///
/// All existing floating-point parameters are frozen. Matching rank-2 parameters receive
/// trainable LoRA [adapter](LoraAdapter)s; other parameters remain frozen without adapters. No
/// model or layer code needs to change—the same `Linear` (and any other module) keeps working, now
/// producing `base + scale * (a @ b)` for adapted weights.
#[derive(Debug, Clone)]
pub struct Lora {
    /// Rank of the low-rank decomposition.
    pub rank: usize,
    /// Scaling numerator; the adapter contribution is scaled by `alpha / rank`.
    pub alpha: f64,
    /// Standard deviation used to initialize the `A` factor. Defaults to `1 / rank`.
    pub init_std: Option<f64>,
    /// Precision the factors are built at. Defaults to the base weight's own
    /// float dtype; a *quantized* base has none, so a model that computes in
    /// half precision over a packed base must say so here — the factors meet
    /// its activations and gradients in elementwise ops that do not promote.
    pub dtype: Option<FloatDType>,
    /// The parameter group on which to apply the LoRA.
    pub param_group: ParamGroup,
}

impl Lora {
    /// Create a new LoRA reparameterizer with the given rank and alpha.
    pub fn new(rank: usize, alpha: f64) -> Self {
        Self {
            rank,
            alpha,
            init_std: None,
            dtype: None,
            param_group: ParamGroup::all(),
        }
    }

    /// Set the parameter group on which to apply LoRA adapters.
    pub fn set_param_group(mut self, group: ParamGroup) -> Self {
        self.param_group = group;
        self
    }

    /// Set the precision the adapter factors are built at.
    pub fn set_dtype(mut self, dtype: FloatDType) -> Self {
        self.dtype = Some(dtype);
        self
    }
}

impl Reparameterizer for Lora {
    type Reparam = LoraAdapter;

    fn reparameterize<const D: usize>(
        &mut self,
        path: &str,
        param: Param<Tensor<D>>,
    ) -> (Param<Tensor<D>>, Option<Self::Reparam>) {
        // LoRA only adapts 2-D weight matrices. Every other base parameter is frozen too, so that
        // only the adapter factors are trained (the canonical LoRA fine-tuning contract).
        if D != 2 {
            let (id, tensor, mapper) = param.consume();
            return (
                Param::from_mapped_value(id, tensor.set_require_grad(false), mapper),
                None,
            );
        }

        let rank = self.rank;
        let (id, tensor, mapper) = param.consume();
        let device = tensor.device();
        let dims = tensor.dims();
        let (d_in, d_out) = (dims[0], dims[1]);
        // The factors compose with the base in `base + scale * (a @ b)`, so they
        // are built at the base's precision: a half-precision checkpoint would
        // otherwise get f32 adapters and fail that op on any backend that does
        // not promote silently. A *quantized* base is dequantized before it
        // composes, so its packed dtype says nothing about theirs — there the
        // configured `dtype` is the only source, and without one they stay at
        // the default.
        let base_dtype = tensor.dtype();
        let dtype = self
            .dtype
            .or_else(|| base_dtype.is_float().then(|| FloatDType::from(base_dtype)));

        // Freeze the base weight; only the adapter factors will be trained.
        let base = Param::from_mapped_value(id, tensor.set_require_grad(false), mapper);

        if self.param_group.matches(&id, Some(path)) {
            // Standard LoRA init: A ~ N(0, std) and B = 0, so the initial delta (and the model output)
            // is unchanged when the adapter is first attached.
            //
            // Built under a persistent-allocation window, as `Param::from_data` places its
            // value: the factors are parameters, alive as long as the module they attach to,
            // and this is the code that knows it. A caller sizing or capping the dynamic
            // pools around a prepared model must not find its trainable weights living there.
            let std = self.init_std.unwrap_or(1.0 / rank as f64);
            let (a, b) = device.memory_persistent_allocations((), |_| {
                let a = Tensor::<2>::random([d_in, rank], Distribution::Normal(0.0, std), &device);
                let b = Tensor::<2>::zeros([rank, d_out], &device);
                match dtype {
                    Some(dtype) => (a.cast(dtype), b.cast(dtype)),
                    None => (a, b),
                }
            });

            let adapter = LoraAdapter {
                a: Param::from_tensor(a),
                b: Param::from_tensor(b),
                scale: self.alpha / rank as f64,
            };

            return (base, Some(adapter));
        }

        (base, None)
    }
}

/// A [`Reparameterizer`] implementing QLoRA: it quantizes the (frozen) base weights and
/// attaches full-precision trainable LoRA adapters to 2-D weights.
///
/// It is applied via [`Module::apply_qlora`](crate::module::Module::apply_qlora).
///
/// The quantized base is kept at rest in its low-bit representation; the adapter contribution is
/// added on top during the forward pass (the base is dequantized on the fly when composed).
pub struct QLora {
    lora: Lora,
    quantizer: Quantizer,
}

impl QLora {
    /// Create a new QLoRA reparameterizer from LoRA settings and a quantizer.
    pub fn new(lora: Lora, quantizer: Quantizer) -> Self {
        Self { lora, quantizer }
    }
}

impl Reparameterizer for QLora {
    type Reparam = LoraAdapter;

    fn reparameterize<const D: usize>(
        &mut self,
        path: &str,
        param: Param<Tensor<D>>,
    ) -> (Param<Tensor<D>>, Option<Self::Reparam>) {
        // Quantize the frozen base weight first, then attach a trainable adapter to 2-D weights.
        let param = self.quantizer.map_float_at_path(param, path);
        self.lora.reparameterize(path, param)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "autodiff")]
    use crate::module::AutodiffModule;
    use crate::module::{Module, ParamId};
    use crate::test_device;
    use crate::test_utils::SimpleLinear;
    use burn_tensor::Tolerance;

    fn lora_model(in_features: usize, out_features: usize) -> (SimpleLinear, super::Lora) {
        let device = test_device();
        let lora = Lora::new(2, 4.0);
        let model = SimpleLinear::new(in_features, out_features, &device).apply_lora(lora.clone());
        (model, lora)
    }

    #[test]
    fn materialize_lora_matches_base_plus_delta() {
        let device = test_device();
        let (model, config) = lora_model(4, 6);

        let weight = &model.weight;
        let adapter = weight.adapter().expect("adapter should be attached");

        // The effective value must equal base + scale * (a @ b).
        let expected = weight.base() + adapter.delta();
        weight
            .val()
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());

        // Scale is alpha / rank.
        assert_eq!(adapter.scale, config.alpha / config.rank as f64);
        let _ = device;
    }

    #[test]
    fn lora_freezes_base_and_trains_adapter() {
        let (model, _) = lora_model(4, 6);
        let weight = &model.weight;
        let adapter = weight.adapter().expect("adapter should be attached");

        // Distinct parameter ids for base / a / b.
        let ids = [weight.id, adapter.a.id, adapter.b.id];
        assert_eq!(
            ids.iter()
                .collect::<alloc::collections::BTreeSet<&ParamId>>()
                .len(),
            3
        );

        // num_params includes the adapter factors:
        // weight [6,4]=24, bias [6]=6, a [6,2]=12, b [2,4]=8.
        assert_eq!(model.num_params(), 24 + 6 + 12 + 8);
    }

    #[test]
    fn lora_b_is_zero_initialized_so_initial_delta_is_zero() {
        let (model, _) = lora_model(4, 6);
        let weight = &model.weight;
        // B = 0 => delta = 0 => effective weight equals the (frozen) base at init.
        weight
            .val()
            .into_data()
            .assert_approx_eq::<f32>(&weight.base().into_data(), Tolerance::default());
    }

    #[test]
    fn lora_record_roundtrip_preserves_base_and_adapter() {
        let (model, config) = lora_model(4, 6);

        // A freshly-prepared model has different random base/A and zero B.
        let device = test_device();
        let target = SimpleLinear::new(4, 6, &device).apply_lora(config);

        let record = model.clone().into_record();
        let loaded = target.load_record(record);

        // Base, A and B must all be restored from the record (paths weight / weight.lora.a / .b).
        loaded
            .weight
            .base()
            .into_data()
            .assert_eq(&model.weight.base().into_data(), true);
        loaded
            .weight
            .adapter()
            .unwrap()
            .a
            .val()
            .into_data()
            .assert_eq(&model.weight.adapter().unwrap().a.val().into_data(), true);
        loaded
            .weight
            .val()
            .into_data()
            .assert_eq(&model.weight.val().into_data(), true);
    }

    #[cfg(feature = "autodiff")]
    #[test]
    fn lora_backward_grads_adapter_only() {
        let device = test_device().autodiff();
        let lora = Lora::new(2, 4.0);
        let model = SimpleLinear::new(4, 6, &device).apply_lora(lora);

        // Forward through the composed weight and backpropagate.
        let loss = model.weight.val().sum();
        let grads = loss.backward();

        let adapter = model.weight.adapter().unwrap();
        // Adapter factors receive gradients; the frozen base does not.
        assert!(adapter.a.val().grad(&grads).is_some());
        assert!(adapter.b.val().grad(&grads).is_some());
        assert!(model.weight.base().grad(&grads).is_none());
    }

    #[test]
    fn qlora_quantizes_base_and_attaches_adapter() {
        use crate::module::Quantizer;
        use burn_tensor::quantization::{Calibration, QuantValue};

        let device = test_device();
        let scheme = device
            .settings()
            .quantization
            .scheme
            .with_value(QuantValue::Q8S);
        let quantizer = Quantizer::new(Calibration::MinMax, scheme);

        let original = SimpleLinear::new(8, 8, &device).weight.val();

        let qlora = QLora::new(Lora::new(2, 4.0), quantizer);
        let model = SimpleLinear::new(8, 8, &device).apply_qlora(qlora);

        let weight = &model.weight;
        assert!(weight.adapter().is_some());

        // The composed value (dequant(base) + delta) has the right shape and is finite. With B = 0
        // the initial delta is zero, so it is just the dequantized base.
        let composed = weight.val();
        assert_eq!(composed.dims(), [8, 8]);
        assert_eq!(composed.into_data().shape, original.into_data().shape);
    }

    #[test]
    fn qlora_composes_packed_base_at_the_configured_factor_dtype() {
        use crate::module::Quantizer;
        use burn_tensor::DType;
        use burn_tensor::quantization::{Calibration, QuantValue};

        let device = test_device();
        let scheme = device
            .settings()
            .quantization
            .scheme
            .with_value(QuantValue::Q8S);
        let quantizer = Quantizer::new(Calibration::MinMax, scheme);

        // A packed base carries no float dtype, so the configured one is the
        // only source the factors have.
        let qlora = QLora::new(Lora::new(2, 4.0).set_dtype(FloatDType::F16), quantizer);
        let model = SimpleLinear::new(8, 8, &device).apply_qlora(qlora);

        let weight = &model.weight;
        let adapter = weight.adapter().expect("adapter should be attached");
        assert_eq!(adapter.a.val().dtype(), DType::F16);
        assert_eq!(adapter.b.val().dtype(), DType::F16);

        // Mixed QFloat/Float addition dequantizes the packed base directly to
        // the factors' dtype.
        assert_eq!(weight.val().dtype(), DType::F16);
    }

    #[test]
    fn param_group_restricts_adapter_to_matching_parameters() {
        use crate as burn;

        #[derive(Module, Debug)]
        struct TwoWeights {
            a: Param<Tensor<2>>,
            b: Param<Tensor<2>>,
        }

        let device = test_device();
        let model = TwoWeights {
            a: Param::from_tensor(Tensor::random(
                [4, 4],
                burn_tensor::Distribution::Default,
                &device,
            )),
            b: Param::from_tensor(Tensor::random(
                [4, 4],
                burn_tensor::Distribution::Default,
                &device,
            )),
        };

        let group = ParamGroup::from_predicate("a");
        let lora = Lora::new(2, 4.0).set_param_group(group);
        let model = model.apply_lora(lora);

        // Only the parameter whose path matches the group gets an adapter attached.
        assert!(
            model.a.adapter().is_some(),
            "parameter in the group should get a LoRA adapter"
        );
        assert!(
            model.b.adapter().is_none(),
            "parameter outside the group should not get a LoRA adapter"
        );

        // Every 2-D weight is frozen regardless of group membership.
        assert!(!model.a.base().is_require_grad());
        assert!(!model.b.val().is_require_grad());
    }

    #[cfg(feature = "autodiff")]
    #[test]
    fn lora_valid_folds_adapter_for_inference() {
        let device = test_device().autodiff();
        let lora = Lora::new(2, 4.0);
        let model = SimpleLinear::new(4, 6, &device).apply_lora(lora);

        let inference = model.valid();
        // The inference parameter has no adapter (folded) and equals the composed training weight.
        assert!(inference.weight.adapter().is_none());
        inference.weight.val().into_data().assert_approx_eq::<f32>(
            &model.weight.val().inner().into_data(),
            Tolerance::default(),
        );
    }
}
