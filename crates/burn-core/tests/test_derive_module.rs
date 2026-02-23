use std::marker::PhantomData;

use burn::module::Initializer;
use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use burn_core as burn;
use burn_cubecl::CubeBackend;
use cubecl::cuda::CudaRuntime;

// pub type TestBackend = burn_ndarray::NdArray<f32>;
// #[cfg(feature = "std")]
// pub type TestAutodiffBackend = burn_autodiff::Autodiff<TestBackend>;

pub type TestBackend<F = f32, I = i32> = CubeBackend<CudaRuntime, F, I, u8>;
pub type TestAutodiffBackend<F = f32, I = i32> = burn_autodiff::Autodiff<TestBackend<F, I>>;

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
    use std::sync::mpsc::SyncSender;

    use burn_autodiff::start_gradient_sync_server;
    use burn_tensor::{
        TensorData,
        backend::{AutodiffBackend, PeerId, ReduceOperation},
    };
    use rand::{
        SeedableRng,
        rngs::{StdRng, SysRng},
    };
    use serial_test::{parallel, serial};

    use super::*;

    #[test]
    #[parallel]
    fn should_have_grad_by_default() {
        let device = <TestBackend as Backend>::Device::default();
        let module = ModuleBasic::<TestAutodiffBackend>::new(&device);
        let grad_x = calculate_grads(&module, |weights, x| weights.matmul(x));

        assert!(grad_x.is_some());
    }

    #[test]
    #[parallel]
    fn should_have_no_grad_after_no_grad() {
        let device = <TestAutodiffBackend as Backend>::Device::default();
        let module = ModuleBasic::<TestAutodiffBackend>::new(&device).no_grad();
        let grad_x = calculate_grads(&module, |weights, x| weights.matmul(x));

        assert!(grad_x.is_none());
    }

    #[test]
    #[parallel]
    fn should_have_grad_when_from_record() {
        let device = <TestAutodiffBackend as Backend>::Device::default();
        let module = ModuleBasic::<TestAutodiffBackend>::new(&device);
        let record = ModuleBasicRecord {
            weight_basic: module.weight_basic.clone(), // Even when param is no_grad,
        };
        let module = module.load_record(record);
        let grad_x = calculate_grads(&module, |weights, x| weights.matmul(x));

        assert!(grad_x.is_some());
    }

    #[test]
    #[serial]
    fn sharded_module_should_sync_gradients_sum() {
        compare_sync_gradients(ReduceOperation::Sum, |weights, x| weights.matmul(x));
    }

    #[test]
    #[serial]
    fn sharded_module_should_sync_gradients_mean() {
        compare_sync_gradients(ReduceOperation::Mean, |weights, x| weights.matmul(x));
    }

    #[test]
    #[serial]
    fn sharded_module_should_sync_gradients_sum_residual() {
        compare_sync_gradients(ReduceOperation::Sum, |weights, x| {
            let y = weights.clone().matmul(x);
            y.add(weights)
        });
    }

    #[test]
    #[serial]
    fn sharded_module_should_sync_gradients_mean_residual() {
        compare_sync_gradients(ReduceOperation::Mean, |weights, x| {
            let y = weights.clone().matmul(x);
            y.add(weights)
        });
    }

    #[test]
    #[serial]
    fn sharded_module_should_sync_gradients_sum_activation() {
        compare_sync_gradients(ReduceOperation::Sum, |weights, x| {
            let y = weights.clone().matmul(x);
            let y = y.add(weights);
            burn_tensor::activation::relu(y)
        });
    }

    #[test]
    #[serial]
    fn sharded_module_should_sync_gradients_mean_activation() {
        compare_sync_gradients(ReduceOperation::Mean, |weights, x| {
            let y = weights.clone().matmul(x);
            let y = y.add(weights);
            burn_tensor::activation::relu(y)
        });
    }

    #[test]
    #[serial]
    fn sharded_module_should_sync_gradients_sum_diamond_graph() {
        compare_sync_gradients(ReduceOperation::Sum, |weights, x| {
            let left = weights.clone().matmul(x.clone().mul_scalar(2));
            let right = weights.clone().matmul(x.clone().exp());
            Tensor::cat(vec![left, right], 0)
        });
    }

    #[test]
    #[serial]
    fn sharded_module_should_sync_gradients_mean_diamond_graph() {
        compare_sync_gradients(ReduceOperation::Mean, |weights, x| {
            let left = weights.clone().matmul(x.clone().mul_scalar(2));
            let right = weights.clone().matmul(x.clone().exp());
            Tensor::cat(vec![left, right], 0)
        });
    }

    fn compare_sync_gradients(
        op: ReduceOperation,
        transformation: fn(
            Tensor<TestAutodiffBackend, 2>,
            Tensor<TestAutodiffBackend, 2>,
        ) -> Tensor<TestAutodiffBackend, 2>,
    ) {
        let num_devices = 4;
        let device = <TestBackend as Backend>::Device::default();
        let module = ModuleBasic::<TestAutodiffBackend>::new(&device);

        let mut recvs = vec![];
        start_gradient_sync_server::<TestBackend>(num_devices);
        for i in 0..num_devices {
            let (send, recv) = std::sync::mpsc::sync_channel(32);
            recvs.push(recv);
            std::thread::spawn({
                let module_thread = module.clone();
                move || run_peer_sharded(&module_thread, PeerId::from(i), op, send, transformation)
            });
        }

        let grad_x1 = recvs[0].recv().unwrap();
        for i in 1..num_devices {
            grad_x1
                .clone()
                .unwrap()
                .to_data()
                .assert_eq(&recvs[i].recv().unwrap().unwrap().to_data(), true);
        }
    }

    pub fn run_peer_sharded(
        module: &ModuleBasic<TestAutodiffBackend>,
        id: PeerId,
        op: ReduceOperation,
        output: SyncSender<
            Option<Tensor<<TestAutodiffBackend as AutodiffBackend>::InnerBackend, 2>>,
        >,
        transformation: fn(
            Tensor<TestAutodiffBackend, 2>,
            Tensor<TestAutodiffBackend, 2>,
        ) -> Tensor<TestAutodiffBackend, 2>,
    ) {
        let device = <TestAutodiffBackend as Backend>::Device::default();
        let module = module.clone().fork(&device);

        let module = module.grad_sharded(id, op);
        let grads_x = calculate_grads(&module, transformation);

        output.send(grads_x).unwrap();
    }

    fn calculate_grads(
        module: &ModuleBasic<TestAutodiffBackend>,
        transformation: fn(
            Tensor<TestAutodiffBackend, 2>,
            Tensor<TestAutodiffBackend, 2>,
        ) -> Tensor<TestAutodiffBackend, 2>,
    ) -> Option<Tensor<<TestAutodiffBackend as AutodiffBackend>::InnerBackend, 2>> {
        let device = module.weight_basic.device();
        let data = TensorData::random::<f32, _, _>(
            module.weight_basic.shape(),
            burn_tensor::Distribution::Default,
            &mut StdRng::try_from_rng(&mut SysRng).unwrap(),
        );
        let x = Tensor::from_data(data, &device).require_grad();
        let y = transformation(module.weight_basic.val(), x);

        let mut grads = y.backward();
        module.weight_basic.grad_remove(&mut grads)
    }
}
