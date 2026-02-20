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

    use burn_collective::{AllReduceStrategy, CollectiveConfig, register};
    use burn_tensor::{
        TensorData,
        backend::{AutodiffBackend, PeerId, ReduceOperation},
    };
    use rand::{
        SeedableRng,
        rngs::{StdRng, SysRng},
    };

    use super::*;

    // TODO : grad_sharded() tests

    #[test]
    fn should_have_grad_by_default() {
        let device = <TestBackend as Backend>::Device::default();
        let module = ModuleBasic::<TestAutodiffBackend>::new(&device);
        let grad_x = calculate_grads(&module);

        assert!(grad_x.is_some());
    }

    #[test]
    fn should_have_no_grad_after_no_grad() {
        let device = <TestAutodiffBackend as Backend>::Device::default();
        let module = ModuleBasic::<TestAutodiffBackend>::new(&device).no_grad();
        let grad_x = calculate_grads(&module);

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
        let grad_x = calculate_grads(&module);

        assert!(grad_x.is_some());
    }

    #[test]
    fn grad_sharded_module_should_sync_gradients() {
        let device = <TestBackend as Backend>::Device::default();
        let module = ModuleBasic::<TestAutodiffBackend>::new(&device);
        let config = CollectiveConfig::default()
            .with_num_devices(2)
            .with_local_all_reduce_strategy(AllReduceStrategy::Centralized);

        let mut recvs = vec![];
        for i in 0..2 {
            let (send, recv) = std::sync::mpsc::sync_channel(32);
            recvs.push(recv);
            std::thread::spawn({
                let config_thread = config.clone();
                let module_thread = module.clone();
                move || {
                    run_peer_sharded(
                        &module_thread,
                        PeerId::from(i),
                        config_thread,
                        ReduceOperation::Sum,
                        send,
                    )
                }
            });
        }

        let grad_x1 = recvs[0].recv().unwrap();
        grad_x1
            .unwrap()
            .to_data()
            .assert_eq(&recvs[1].recv().unwrap().unwrap().to_data(), true);
    }

    pub fn run_peer_sharded(
        module: &ModuleBasic<TestAutodiffBackend>,
        id: PeerId,
        config: CollectiveConfig,
        op: ReduceOperation,
        output: SyncSender<
            Option<Tensor<<TestAutodiffBackend as AutodiffBackend>::InnerBackend, 2>>,
        >,
    ) {
        let device = <TestAutodiffBackend as Backend>::Device::default();
        let module = module.clone().fork(&device);

        register::<<TestAutodiffBackend as AutodiffBackend>::InnerBackend>(
            id,
            device.clone(),
            config,
        )
        .unwrap();

        let module = module.grad_sharded(id, op);
        let grads_x = calculate_grads(&module);

        output.send(grads_x).unwrap();
    }

    fn calculate_grads(
        module: &ModuleBasic<TestAutodiffBackend>,
    ) -> Option<Tensor<<TestAutodiffBackend as AutodiffBackend>::InnerBackend, 2>> {
        let device = module.weight_basic.device();
        let data = TensorData::random::<f32, _, _>(
            module.weight_basic.shape(),
            burn_tensor::Distribution::Default,
            &mut StdRng::try_from_rng(&mut SysRng).unwrap(),
        );
        let x = Tensor::from_data(data, &device).require_grad();
        let y = module.weight_basic.val().matmul(x);

        let mut grads = y.backward();
        module.weight_basic.grad_remove(&mut grads)
    }
}
