use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Shape, Tensor};
use burn_core as burn;

pub type TestBackend = burn_ndarray::NdArrayBackend<f32>;
#[cfg(feature = "std")]
pub type TestADBackend = burn_autodiff::ADBackendDecorator<TestBackend>;

#[derive(Module, Debug)]
pub struct ModuleBasic<B: Backend> {
    weight_basic: Param<Tensor<B, 2>>,
}

impl<B: Backend> ModuleBasic<B> {
    fn new() -> Self {
        let weight_basic = Tensor::random(Shape::new([20, 20]), Distribution::Default);
        Self {
            weight_basic: Param::from(weight_basic),
        }
    }
}

#[derive(Module, Debug)]
pub struct ModuleComposed<B: Backend> {
    weight: Param<Tensor<B, 2>>,
    basic: ModuleBasic<B>,
}

impl<B: Backend> ModuleComposed<B> {
    fn new() -> Self {
        let weight = Tensor::random(Shape::new([20, 20]), Distribution::Default);
        Self {
            weight: Param::from(weight),
            basic: ModuleBasic::new(),
        }
    }
}

mod state {
    use super::*;

    #[test]
    fn should_load_from_record_basic() {
        let module_1 = ModuleBasic::<TestBackend>::new();
        let mut module_2 = ModuleBasic::<TestBackend>::new();
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
        let module_1 = ModuleComposed::<TestBackend>::new();
        let mut module_2 = ModuleComposed::<TestBackend>::new();
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

#[cfg(feature = "std")]
mod require_grad {
    use burn_tensor::backend::ADBackend;

    use super::*;

    #[test]
    fn should_have_grad_by_default() {
        let module = ModuleBasic::<TestADBackend>::new();
        let mut grads = calculate_grads(&module);

        let grad_x = module.weight_basic.grad_remove(&mut grads);

        assert!(grad_x.is_some());
    }

    #[test]
    fn should_have_no_grad_after_no_grad() {
        let module = ModuleBasic::<TestADBackend>::new().no_grad();
        let mut grads = calculate_grads(&module);

        let grad_x = module.weight_basic.grad_remove(&mut grads);

        assert!(grad_x.is_none());
    }

    #[test]
    fn should_have_grad_when_from_record() {
        let module = ModuleBasic::<TestADBackend>::new();
        let record = ModuleBasicRecord {
            weight_basic: module.weight_basic.clone(), // Even when param is no_grad,
        };
        let module = module.load_record(record);
        let mut grads = calculate_grads(&module);

        let grad_x = module.weight_basic.grad_remove(&mut grads);

        assert!(grad_x.is_some());
    }

    fn calculate_grads(
        module: &ModuleBasic<TestADBackend>,
    ) -> <TestADBackend as ADBackend>::Gradients {
        let x = Tensor::ones([20, 20]).require_grad();
        let y = module.weight_basic.val().matmul(x);

        y.backward()
    }
}
