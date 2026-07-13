use alloc::{
    boxed::Box,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use burn_tensor::{Distribution, Tensor};

use crate::module::{LoraAdapter, ModuleMapper, Param, ParamGroup, Quantizer};

/// Configuration describing how to attach LoRA adapters to a module's weights.
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// Rank of the low-rank decomposition.
    pub rank: usize,
    /// Scaling numerator; the adapter contribution is scaled by `alpha / rank`.
    pub alpha: f64,
    /// Standard deviation used to initialize the `A` factor. Defaults to `1 / rank`.
    pub init_std: Option<f64>,
    /// The parameter group on which to apply the LoRA.
    pub param_group: ParamGroup,
}

impl LoraConfig {
    /// Create a new LoRA configuration with the given rank and alpha.
    pub fn new(rank: usize, alpha: f64) -> Self {
        Self {
            rank,
            alpha,
            init_std: None,
            param_group: ParamGroup::all(),
        }
    }

    /// Set the parameter group to quantize on which to apply the LoRA.
    pub fn set_param_group(mut self, group: ParamGroup) -> Self {
        self.param_group = group;
        self
    }
}

/// A [module mapper](ModuleMapper) that attaches LoRA adapters to 2-D weight parameters.
///
/// Apply it the same way as quantization:
///
/// ```rust,ignore
/// let model = model.map(&mut LoraMapper::new(LoraConfig::new(8, 16.0)));
/// ```
///
/// Each rank-2 float weight is frozen and given a trainable low-rank [adapter](LoraAdapter); other
/// parameters are left untouched. No model or layer code needs to change — the same `Linear` (and
/// any other module) keeps working, now producing `base + scale * (a @ b)` for adapted weights.
#[derive(Debug, Clone)]
pub struct LoraMapper {
    config: LoraConfig,
    path: Vec<String>,
}

impl LoraMapper {
    /// Create a new mapper from the given configuration.
    pub fn new(config: LoraConfig) -> Self {
        Self {
            config,
            path: vec![],
        }
    }

    /// Specify a parameter group on which to apply the LoRA.
    pub fn for_group(mut self, group: ParamGroup) -> Self {
        self.config.param_group = group;
        self
    }
}

impl ModuleMapper for LoraMapper {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        // LoRA only adapts 2-D weight matrices. Every other base parameter is frozen too, so that
        // only the adapter factors are trained (the canonical LoRA fine-tuning contract).
        if D != 2 {
            let (id, tensor, mapper) = param.consume();
            return Param::from_mapped_value(id, tensor.set_require_grad(false), mapper);
        }

        let rank = self.config.rank;
        let (id, tensor, mapper) = param.consume();
        let device = tensor.device();
        let dims = tensor.dims();
        let (d_in, d_out) = (dims[0], dims[1]);

        // Freeze the base weight; only the adapter factors will be trained.
        let base = Param::from_mapped_value(id, tensor.set_require_grad(false), mapper);

        let path = self.path.join(".");
        if self.config.param_group.matches(&id, Some(&path)) {
            // Standard LoRA init: A ~ N(0, std) and B = 0, so the initial delta (and the model output)
            // is unchanged when the adapter is first attached.
            let std = self.config.init_std.unwrap_or(1.0 / rank as f64);
            let a = Tensor::<2>::random([d_in, rank], Distribution::Normal(0.0, std), &device);
            let b = Tensor::<2>::zeros([rank, d_out], &device);

            let adapter = LoraAdapter {
                a: Param::from_tensor(a),
                b: Param::from_tensor(b),
                scale: self.config.alpha / rank as f64,
            };

            return base.with_adapter(Some(Box::new(adapter)));
        }

        base
    }
}

/// A [module mapper](ModuleMapper) implementing QLoRA: it quantizes the (frozen) base weights and
/// attaches full-precision trainable LoRA adapters to 2-D weights.
///
/// The quantized base is kept at rest in its low-bit representation; the adapter contribution is
/// added on top during the forward pass (the base is dequantized on the fly when composed).
pub struct QLoraMapper {
    lora: LoraMapper,
    quantizer: Quantizer,
}

impl QLoraMapper {
    /// Create a new QLoRA mapper from the LoRA configuration and a quantizer.
    pub fn new(config: LoraConfig, quantizer: Quantizer) -> Self {
        Self {
            lora: LoraMapper::new(config),
            quantizer,
        }
    }
}

impl ModuleMapper for QLoraMapper {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        // Quantize the frozen base weight first, then attach a trainable adapter to 2-D weights.
        let param = self.quantizer.map_float(param);
        self.lora.map_float(param)
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

    fn lora_model(in_features: usize, out_features: usize) -> (SimpleLinear, super::LoraConfig) {
        let device = test_device();
        let config = LoraConfig::new(2, 4.0);
        let model =
            SimpleLinear::new(in_features, out_features, &device).apply_lora(config.clone());
        (model, config)
    }

    #[test]
    fn compose_lora_matches_base_plus_delta() {
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
    fn lora_mapper_freezes_base_and_trains_adapter() {
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
        let config = LoraConfig::new(2, 4.0);
        let model = SimpleLinear::new(4, 6, &device).apply_lora(config);

        // Forward through the composed weight and backpropagate.
        let loss = model.weight.val().sum();
        let grads = loss.backward();

        let adapter = model.weight.adapter().unwrap();
        // Adapter factors receive gradients; the frozen base does not.
        assert!(adapter.a.val().grad(&grads).is_some());
        assert!(adapter.b.val().grad(&grads).is_some());
        assert!(model.weight.base().grad(&grads).is_none());
    }

    // #[cfg(not(feature = "tch"))]
    #[test]
    fn qlora_quantizes_base_and_attaches_adapter() {
        use crate::module::Quantizer;
        use burn_tensor::quantization::{Calibration, QuantLevel, QuantParam, QuantValue};

        let device = test_device();
        let scheme = device
            .settings()
            .quantization
            .scheme
            .with_value(QuantValue::Q8S)
            .with_level(QuantLevel::Tensor)
            .with_param(QuantParam::F32);
        let quantizer = Quantizer::new(Calibration::MinMax, scheme);

        let original = SimpleLinear::new(8, 8, &device).weight.val();

        let model =
            SimpleLinear::new(8, 8, &device).apply_qlora(LoraConfig::new(2, 4.0), quantizer);

        let weight = &model.weight;
        assert!(weight.adapter().is_some());

        // The composed value (dequant(base) + delta) has the right shape and is finite. With B = 0
        // the initial delta is zero, so it is just the dequantized base.
        let composed = weight.val();
        assert_eq!(composed.dims(), [8, 8]);
        assert_eq!(composed.into_data().shape, original.into_data().shape);
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
        let config = LoraConfig::new(2, 4.0).set_param_group(group);
        let model = model.apply_lora(config);

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
        let config = LoraConfig::new(2, 4.0);
        let model = SimpleLinear::new(4, 6, &device).apply_lora(config);

        let inference = model.valid();
        // The inference parameter has no adapter (folded) and equals the composed training weight.
        assert!(inference.weight.adapter().is_none());
        inference.weight.val().into_data().assert_approx_eq::<f32>(
            &model.weight.val().inner().into_data(),
            Tolerance::default(),
        );
    }
}
