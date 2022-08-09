use burn::module::{Module, Param};
use burn::tensor::back::Backend;
use burn::tensor::{Distribution, Shape, Tensor};
use std::collections::HashSet;

type TestBackend = burn::tensor::back::NdArray<i16>;

#[derive(Module, Debug)]
struct ModuleBasic<B>
where
    B: Backend,
{
    weight_basic: Param<Tensor<B, 2>>,
}

impl<B: Backend> ModuleBasic<B> {
    fn new() -> Self {
        let weight_basic = Tensor::random(Shape::new([20, 20]), Distribution::Standard);
        Self {
            weight_basic: Param::new(weight_basic),
        }
    }
}

#[derive(Module, Debug)]
struct ModuleComposed<B>
where
    B: Backend,
{
    weight: Param<Tensor<B, 2>>,
    basic: Param<ModuleBasic<B>>,
}

impl<B: Backend> ModuleComposed<B> {
    fn new() -> Self {
        let weight = Tensor::random(Shape::new([20, 20]), Distribution::Standard);
        Self {
            weight: Param::new(weight),
            basic: Param::new(ModuleBasic::new()),
        }
    }
}

mod state {
    use super::*;

    #[test]
    fn should_output_state_basic() {
        let module = ModuleBasic::<TestBackend>::new();

        let state = module.state();

        let keys: Vec<String> = state.values.keys().map(|n| n.to_string()).collect();
        assert_eq!(keys, vec!["ModuleBasic.weight_basic".to_string()]);
    }

    #[test]
    fn should_output_state_composed() {
        let module = ModuleComposed::<TestBackend>::new();

        let state = module.state();

        let keys: HashSet<String> = state.values.keys().map(|n| n.to_string()).collect();
        assert!(keys.contains("ModuleComposed.basic.ModuleBasic.weight_basic"));
        assert!(keys.contains("ModuleComposed.weight"));
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn should_load_from_state_basic() {
        let module_1 = ModuleBasic::<TestBackend>::new();
        let mut module_2 = ModuleBasic::<TestBackend>::new();
        let state_1 = module_1.state();
        assert_ne!(
            module_1.weight_basic.to_data(),
            module_2.weight_basic.to_data()
        );

        module_2.load(&state_1);
        assert_eq!(
            module_1.weight_basic.to_data(),
            module_2.weight_basic.to_data()
        );
    }

    #[test]
    fn should_load_from_state_compose() {
        let module_1 = ModuleComposed::<TestBackend>::new();
        let mut module_2 = ModuleComposed::<TestBackend>::new();
        assert_ne!(module_1.weight.to_data(), module_2.weight.to_data());
        assert_ne!(
            module_1.basic.weight_basic.to_data(),
            module_2.basic.weight_basic.to_data()
        );

        let state_1 = module_1.state();
        module_2.load(&state_1);

        assert_eq!(module_1.weight.to_data(), module_2.weight.to_data());
        assert_eq!(
            module_1.basic.weight_basic.to_data(),
            module_2.basic.weight_basic.to_data()
        );
    }
}

mod num_params {
    use super::*;

    #[test]
    fn should_calculate_num_params_basic() {
        let module = ModuleBasic::<TestBackend>::new();
        assert_eq!(20 * 20, module.num_params());
    }

    #[test]
    fn should_output_state_composed() {
        let module = ModuleComposed::<TestBackend>::new();
        assert_eq!(2 * 20 * 20, module.num_params());
    }
}
