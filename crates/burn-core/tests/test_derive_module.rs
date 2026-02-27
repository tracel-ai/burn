use std::marker::PhantomData;

use burn::module::Initializer;
use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use burn_core as burn;

pub type TestBackend = burn_ndarray::NdArray<f32>;
#[cfg(feature = "std")]
pub type TestAutodiffBackend = burn_autodiff::Autodiff<TestBackend>;

#[derive(Module, Debug)]
pub struct ModuleBasic<B: Backend> {
    weight_basic: Param<Tensor<B, 2>>,
}

#[derive(Module, Debug)]
#[allow(unused)]
struct ModuleTensorConstInt<B: Backend> {
    weight_basic: Tensor<B, 2, Int>,
}

impl<B: Backend> ModuleBasic<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            weight_basic: Initializer::Normal {
                std: 1.0,
                mean: 0.0,
            }
            .init([20, 20], device),
        }
    }
}

#[derive(Module, Debug)]
struct ModuleWithConstGeneric<B: Backend, const N: usize> {
    modules: [ModuleBasic<B>; N],
}

#[derive(Module, Debug)]
struct ModuleWithGenericModule<B: Backend, M> {
    module: M,
    _backend: PhantomData<B>,
}

#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
enum ModuleEnum<B: Backend> {
    Basic(ModuleBasic<B>),
    Composed(ModuleComposed<B>),
}

#[derive(Module, Debug)]
#[allow(unused)]
enum ModuleEnumNested<B: Backend> {
    AnotherEnum(ModuleEnum<B>),
}

#[derive(Module, Debug)]
enum ModuleEnumWithGenericModule<B: Backend, M: Module<B>> {
    Basic(ModuleBasic<B>),
    Generic(ModuleWithGenericModule<B, M>),
}

#[derive(Module, Debug)]
pub struct ModuleComposed<B: Backend> {
    weight: Param<Tensor<B, 2>>,
    basic: ModuleBasic<B>,
    tuple: (ModuleBasic<B>, ModuleBasic<B>),
}

impl<B: Backend> ModuleComposed<B> {
    fn new(device: &B::Device) -> Self {
        let weight = Initializer::Normal {
            std: 1.0,
            mean: 0.0,
        }
        .init([20, 20], device);

        Self {
            weight,
            basic: ModuleBasic::new(device),
            tuple: (ModuleBasic::new(device), ModuleBasic::new(device)),
        }
    }
}

#[derive(Module, Debug)]
pub struct ModuleWithAttributes<B: Backend, M: Module<B>> {
    /// A normal parameter.
    weight: Param<Tensor<B, 2>>,
    /// A nested module.
    nested: ModuleEnumWithGenericModule<B, M>,
    /// By default, primitives were not persistent (same as `#[module(skip)]`).
    other_prob: f64,
    /// By default, tensors were not persistent (same as `#[module(skip)]`).
    tensor: Tensor<B, 1>,
    /// A persistent config value.
    #[module(constant)]
    dropout_prob: f64,
    /// A field that is recomputed at runtime.
    #[module(skip)]
    cached_mask: Option<Tensor<B, 2>>,
    /// A field that contains some debug state.
    #[module(skip)]
    debug_state: String,
}

impl<B: Backend> ModuleWithAttributes<B, ModuleBasic<B>> {
    fn new(device: &B::Device) -> Self {
        let basic = ModuleBasic::new(device);
        let weight = basic.weight_basic.clone();

        Self {
            weight: weight,
            nested: ModuleEnumWithGenericModule::Basic(basic),
            other_prob: 1.,
            tensor: Tensor::ones([2], device),
            dropout_prob: 0.,
            cached_mask: Some(Tensor::ones([2, 2], device)),
            debug_state: "Hello World".into(),
        }
    }
}

#[allow(dead_code)]
mod compiletime_clone_impl_check {
    use burn_core::{
        module::{Module, ModuleDisplay},
        prelude::Backend,
        record::{PrecisionSettings, Record},
    };

    use super::*;

    type RecordItem<M, B, S> = <<M as Module<B>>::Record as Record<B>>::Item<S>;

    fn implements_clone<T: Clone>() {}

    fn basic_implements_clone<B: Backend, S: PrecisionSettings>() {
        implements_clone::<RecordItem<ModuleBasic<B>, B, S>>();
        implements_clone::<RecordItem<ModuleComposed<B>, B, S>>();
    }

    fn generic_implements_clone<B, S, M>()
    where
        B: Backend,
        S: PrecisionSettings,
        M: Module<B> + ModuleDisplay,
        RecordItem<M, B, S>: Clone,
    {
        implements_clone::<RecordItem<ModuleWithGenericModule<B, M>, B, S>>();
        implements_clone::<RecordItem<ModuleEnumWithGenericModule<B, M>, B, S>>();
    }
}

mod state {
    use burn_core::module::{EmptyRecord, ValueRecord};

    use super::*;

    #[test]
    fn should_load_from_record_basic() {
        let device = <TestBackend as Backend>::Device::default();
        let module_1 = ModuleBasic::<TestBackend>::new(&device);
        let mut module_2 = ModuleBasic::<TestBackend>::new(&device);
        let state_1 = module_1.clone().into_record();

        assert_ne!(
            module_1.weight_basic.to_data(),
            module_2.weight_basic.to_data()
        );

        module_2 = module_2.load_record(state_1);

        assert_eq!(
            module_1.weight_basic.to_data(),
            module_2.weight_basic.to_data()
        );
    }

    #[test]
    fn should_load_from_record_compose() {
        let device = <TestBackend as Backend>::Device::default();
        let module_1 = ModuleComposed::<TestBackend>::new(&device);
        let mut module_2 = ModuleComposed::<TestBackend>::new(&device);
        assert_ne!(module_1.weight.to_data(), module_2.weight.to_data());
        assert_ne!(
            module_1.basic.weight_basic.to_data(),
            module_2.basic.weight_basic.to_data()
        );

        let state_1 = module_1.clone().into_record();
        module_2 = module_2.load_record(state_1);

        assert_eq!(module_1.weight.to_data(), module_2.weight.to_data());
        assert_eq!(
            module_1.basic.weight_basic.to_data(),
            module_2.basic.weight_basic.to_data()
        );
    }

    #[test]
    fn should_load_from_record_enum() {
        let device = <TestBackend as Backend>::Device::default();
        let module_1 = ModuleEnum::Basic(ModuleBasic::<TestBackend>::new(&device));
        let mut module_2 = ModuleEnum::Basic(ModuleBasic::<TestBackend>::new(&device));
        let state_1 = module_1.clone().into_record();

        let ModuleEnum::Basic(module_1_basic) = module_1 else {
            panic!("Invalid module type")
        };
        let ModuleEnum::Basic(module_2_basic) = module_2.clone() else {
            panic!("Invalid module type")
        };
        assert_ne!(
            module_1_basic.weight_basic.to_data(),
            module_2_basic.weight_basic.to_data()
        );

        module_2 = module_2.load_record(state_1);

        let ModuleEnum::Basic(module_2_basic) = module_2 else {
            panic!("Invalid module type")
        };
        assert_eq!(
            module_1_basic.weight_basic.to_data(),
            module_2_basic.weight_basic.to_data()
        );
    }

    #[test]
    fn should_load_from_record_based_on_attributes() {
        let device = <TestBackend as Backend>::Device::default();
        let mut module_1 = ModuleWithAttributes::<TestBackend, _>::new(&device);
        let mut module_2 = ModuleWithAttributes::new(&device);

        assert_ne!(module_1.weight.to_data(), module_2.weight.to_data(),);

        let ModuleEnumWithGenericModule::Basic(ref m1_basic) = module_1.nested else {
            panic!("Invalid module type")
        };
        let ModuleEnumWithGenericModule::Basic(ref m2_basic) = module_2.nested else {
            panic!("Invalid module type")
        };

        assert_ne!(
            m1_basic.weight_basic.to_data(),
            m2_basic.weight_basic.to_data(),
        );

        assert_eq!(module_1.tensor.to_data(), module_2.tensor.to_data());
        assert_eq!(
            module_1.cached_mask.as_ref().unwrap().to_data(),
            module_2.cached_mask.as_ref().unwrap().to_data()
        );

        assert_eq!(module_1.dropout_prob, module_2.dropout_prob);
        assert_eq!(module_1.other_prob, module_2.other_prob);
        assert_eq!(module_1.debug_state, module_2.debug_state);

        // Alter state of constant and skipped fields to validate persistence
        module_1.cached_mask = Some(module_1.cached_mask.unwrap() * 2);
        module_1.tensor = module_1.tensor * 2;
        module_1.dropout_prob = 1.0;
        module_1.other_prob = 0.;
        module_1.debug_state = "Hello World!".into();

        let state_1 = module_1.clone().into_record();

        assert_eq!(state_1.cached_mask, EmptyRecord);
        assert_eq!(state_1.dropout_prob, ValueRecord::new(1.0));
        assert_eq!(state_1.other_prob, EmptyRecord);
        assert_eq!(state_1.debug_state, EmptyRecord);

        module_2 = module_2.load_record(state_1);

        let ModuleEnumWithGenericModule::Basic(m2_basic) = module_2.nested else {
            panic!("Invalid module type")
        };

        // Modules & params
        assert_eq!(module_1.weight.to_data(), module_2.weight.to_data(),);
        assert_eq!(
            m1_basic.weight_basic.to_data(),
            m2_basic.weight_basic.to_data(),
        );

        // `#[module(constant)]`
        assert_eq!(module_1.dropout_prob, module_2.dropout_prob);

        // `#[module(skip)]` field and other skip-by-default
        assert_ne!(module_1.other_prob, module_2.other_prob);
        assert_ne!(module_1.debug_state, module_2.debug_state);
        assert_ne!(module_1.tensor.to_data(), module_2.tensor.to_data());
        assert_ne!(
            module_1.cached_mask.as_ref().unwrap().to_data(),
            module_2.cached_mask.as_ref().unwrap().to_data()
        );
    }

    #[test]
    fn should_load_from_record_const_generic() {
        let device = <TestBackend as Backend>::Device::default();
        let module_1 = ModuleWithConstGeneric {
            modules: [
                ModuleBasic::<TestBackend>::new(&device),
                ModuleBasic::<TestBackend>::new(&device),
            ],
        };
        let mut module_2 = ModuleWithConstGeneric {
            modules: [
                ModuleBasic::<TestBackend>::new(&device),
                ModuleBasic::<TestBackend>::new(&device),
            ],
        };
        let state_1 = module_1.clone().into_record();

        assert_ne!(
            module_1.modules[0].weight_basic.to_data(),
            module_2.modules[0].weight_basic.to_data(),
        );
        assert_ne!(
            module_1.modules[1].weight_basic.to_data(),
            module_2.modules[1].weight_basic.to_data(),
        );

        module_2 = module_2.load_record(state_1);

        assert_eq!(
            module_1.modules[0].weight_basic.to_data(),
            module_2.modules[0].weight_basic.to_data(),
        );
        assert_eq!(
            module_1.modules[1].weight_basic.to_data(),
            module_2.modules[1].weight_basic.to_data(),
        );
    }

    #[test]
    #[should_panic(expected = "Can't parse record from a different variant")]
    fn should_panic_load_from_incorrect_enum_variant() {
        let device = <TestBackend as Backend>::Device::default();
        let module_1 = ModuleEnum::Basic(ModuleBasic::<TestBackend>::new(&device));
        let module_2 = ModuleEnum::Composed(ModuleComposed::<TestBackend>::new(&device));
        let state_1 = module_1.clone().into_record();

        module_2.load_record(state_1);
    }
}

mod num_params {
    use super::*;

    #[test]
    fn should_calculate_num_params_basic() {
        let device = <TestBackend as Backend>::Device::default();
        let module = ModuleBasic::<TestBackend>::new(&device);
        assert_eq!(20 * 20, module.num_params());
    }

    #[test]
    fn should_output_state_composed() {
        let device = <TestBackend as Backend>::Device::default();
        let module = ModuleComposed::<TestBackend>::new(&device);
        assert_eq!(4 * 20 * 20, module.num_params());
    }

    #[test]
    fn should_calculate_num_params_enum() {
        let device = <TestBackend as Backend>::Device::default();
        let module = ModuleEnum::Basic(ModuleBasic::<TestBackend>::new(&device));
        assert_eq!(20 * 20, module.num_params());

        let module = ModuleEnum::Composed(ModuleComposed::<TestBackend>::new(&device));
        assert_eq!(4 * 20 * 20, module.num_params());
    }
}

#[cfg(feature = "std")]
mod require_grad {
    use burn_tensor::backend::AutodiffBackend;

    use super::*;

    #[test]
    fn should_have_grad_by_default() {
        let device = <TestBackend as Backend>::Device::default();
        let module = ModuleBasic::<TestAutodiffBackend>::new(&device);
        let mut grads = calculate_grads(&module);

        let grad_x = module.weight_basic.grad_remove(&mut grads);

        assert!(grad_x.is_some());
    }

    #[test]
    fn should_have_no_grad_after_no_grad() {
        let device = <TestAutodiffBackend as Backend>::Device::default();
        let module = ModuleBasic::<TestAutodiffBackend>::new(&device).no_grad();
        let mut grads = calculate_grads(&module);

        let grad_x = module.weight_basic.grad_remove(&mut grads);

        assert!(grad_x.is_none());
    }

    #[test]
    fn should_have_grad_when_from_record() {
        let device = <TestAutodiffBackend as Backend>::Device::default();
        let module = ModuleBasic::<TestAutodiffBackend>::new(&device);
        let record = ModuleBasicRecord {
            weight_basic: module.weight_basic.clone(), // Even when param is no_grad,
        };
        let module = module.load_record(record);
        let mut grads = calculate_grads(&module);

        let grad_x = module.weight_basic.grad_remove(&mut grads);

        assert!(grad_x.is_some());
    }

    fn calculate_grads(
        module: &ModuleBasic<TestAutodiffBackend>,
    ) -> <TestAutodiffBackend as AutodiffBackend>::Gradients {
        let device = module.weight_basic.device();
        let x = Tensor::ones([20, 20], &device).require_grad();
        let y = module.weight_basic.val().matmul(x);

        y.backward()
    }
}
