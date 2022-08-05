pub type TestBackend = burn::tensor::back::NdArray<f32>;

mod module_state {
    use super::*;

    use burn::module::{Module, Param};
    use burn::tensor::back::Backend;
    use burn::tensor::{Distribution, Shape, Tensor};
    use std::collections::HashSet;

    #[derive(Module, Debug)]
    struct ModuleBasic<B>
    where
        B: Backend,
    {
        weight: Param<Tensor<2, B>>,
    }

    impl<B: Backend> ModuleBasic<B> {
        fn new() -> Self {
            let weight = Tensor::random(Shape::new([20, 20]), Distribution::Standard);
            Self {
                weight: Param::new(weight),
            }
        }
    }

    #[derive(Module, Debug)]
    struct ModuleComposed<B>
    where
        B: Backend,
    {
        weight: Param<Tensor<2, B>>,
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

    #[test]
    fn should_output_state_depth_1() {
        let module = ModuleBasic::<TestBackend>::new();

        let state = module.state();

        let keys: Vec<String> = state.values.keys().map(|n| n.to_string()).collect();
        assert_eq!(keys, vec!["ModuleBasic.weight".to_string()]);
    }

    #[test]
    fn should_output_state_depth_2() {
        let module = ModuleComposed::<TestBackend>::new();

        let state = module.state();

        let keys: HashSet<String> = state.values.keys().map(|n| n.to_string()).collect();
        assert!(keys.contains("ModuleComposed.basic.ModuleBasic.weight"));
        assert!(keys.contains("ModuleComposed.weight"));
        assert_eq!(keys.len(), 2);
    }
}
