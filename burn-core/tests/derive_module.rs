use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Shape, Tensor};
use burn_core as burn;

pub type TestBackend = burn_ndarray::NdArray<f32>;
#[cfg(feature = "std")]
pub type TestAutodiffBackend = burn_autodiff::Autodiff<TestBackend>;

#[derive(Module, Debug)]
pub struct ModuleBasic<B: Backend> {
    weight_basic: Param<Tensor<B, 2>>,
}

#[derive(Module, Debug)]
struct ModuleTensorConstInt<B: Backend> {
    weight_basic: Tensor<B, 2, Int>,
}

impl<B: Backend> ModuleBasic<B> {
    fn new(device: &B::Device) -> Self {
        let weight_basic = Tensor::random(Shape::new([20, 20]), Distribution::Default, device);
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
    fn new(device: &B::Device) -> Self {
        let weight = Tensor::random(Shape::new([20, 20]), Distribution::Default, device);
        Self {
            weight: Param::from(weight),
            basic: ModuleBasic::new(device),
        }
    }
}

mod state {
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
        assert_eq!(2 * 20 * 20, module.num_params());
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
        let x = Tensor::ones_devauto([20, 20]).require_grad();
        let y = module.weight_basic.val().matmul(x);

        y.backward()
    }
}
