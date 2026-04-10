use std::marker::PhantomData;

use burn::module::Initializer;
use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use burn_core as burn;

// pub type TestBackend = burn_ndarray::NdArray<f32>;
pub type TestBackend = burn_cuda::Cuda;
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

#[derive(Debug, Clone)]
pub enum PaddingConfig {
    Default,
    Other,
}

#[derive(Module, Debug)]
pub struct ModuleWithAttributes<B: Backend, M: Module<B>, N> {
    /// A normal parameter.
    weight: Param<Tensor<B, 2>>,
    /// A nested module.
    nested: ModuleEnumWithGenericModule<B, M>,
    /// By default, primitives were not persistent (same as `#[module(skip)]`).
    other_prob: f64,
    /// By default, tensors were not persistent and not visited/mapped (same as `#[module(skip)]`).
    tensor: Tensor<B, 1>,
    /// A field that is recomputed at runtime.
    #[module(skip)]
    cached_mask: Option<Tensor<B, 2>>,
    /// A field that contains some debug state.
    debug_state: String,
    /// Hint required: this generic is NOT a module.
    #[module(skip)]
    config: N,
}

impl<B: Backend> ModuleWithAttributes<B, ModuleBasic<B>, PaddingConfig> {
    fn new(device: &B::Device) -> Self {
        let basic = ModuleBasic::new(device);
        let weight = basic.weight_basic.clone();

        Self {
            weight,
            nested: ModuleEnumWithGenericModule::Basic(basic),
            other_prob: 1.,
            tensor: Tensor::ones([2], device),
            cached_mask: Some(Tensor::ones([2, 2], device)),
            debug_state: "Hello World".into(),
            config: PaddingConfig::Default,
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
    use burn_core::module::EmptyRecord;

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
        let mut module_1 = ModuleWithAttributes::<TestBackend, _, _>::new(&device);
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

        assert_eq!(module_1.other_prob, module_2.other_prob);
        assert_eq!(module_1.debug_state, module_2.debug_state);

        // Alter state of skipped fields to validate persistence
        module_1.cached_mask = Some(module_1.cached_mask.unwrap() * 2);
        module_1.tensor = module_1.tensor * 2;
        module_1.other_prob = 0.;
        module_1.debug_state = "Hello World!".into();
        module_1.config = PaddingConfig::Other;

        let state_1 = module_1.clone().into_record();

        assert_eq!(state_1.cached_mask, EmptyRecord);
        assert_eq!(state_1.other_prob, EmptyRecord);
        assert_eq!(state_1.debug_state, EmptyRecord);
        assert_eq!(state_1.config, EmptyRecord);

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

        // `#[module(skip)]` field and other skip-by-default
        assert_ne!(module_1.other_prob, module_2.other_prob);
        assert_ne!(module_1.debug_state, module_2.debug_state);
        assert!(matches!(module_1.config, PaddingConfig::Other));
        assert!(matches!(module_2.config, PaddingConfig::Default));
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

    #[test]
    fn should_calculate_num_params_based_on_attributes() {
        let device = <TestBackend as Backend>::Device::default();
        let module = ModuleWithAttributes::<TestBackend, _, _>::new(&device);
        assert_eq!(20 * 20 * 2, module.num_params());
    }
}

#[cfg(feature = "std")]
mod require_grad {
    use burn_tensor::{TensorData, backend::AutodiffBackend};
    use rand::{
        SeedableRng,
        rngs::{StdRng, SysRng},
    };

    use super::*;

    #[test]
    fn should_have_grad_by_default() {
        let device = <TestBackend as Backend>::Device::default();
        let module = ModuleBasic::<TestAutodiffBackend>::new(&device);
        let grad_x = calculate_grads(&module, |weights, x| weights.matmul(x));

        assert!(grad_x.is_some());
    }

    #[test]
    fn should_have_no_grad_after_no_grad() {
        let device = <TestAutodiffBackend as Backend>::Device::default();
        let module = ModuleBasic::<TestAutodiffBackend>::new(&device).no_grad();
        let grad_x = calculate_grads(&module, |weights, x| weights.matmul(x));

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
        let grad_x = calculate_grads(&module, |weights, x| weights.matmul(x));

        assert!(grad_x.is_some());
    }

    fn calculate_grads<B: AutodiffBackend>(
        module: &ModuleBasic<B>,
        transformation: fn(Tensor<B, 2>, Tensor<B, 2>) -> Tensor<B, 2>,
    ) -> Option<Tensor<B::InnerBackend, 2>> {
        let device = module.weight_basic.device();
        let data = TensorData::random::<f32, _, _>(
            module.weight_basic.shape(),
            burn_tensor::Distribution::Default,
            &mut StdRng::try_from_rng(&mut SysRng).unwrap(),
        );
        let x = Tensor::from_data(data, &device).require_grad();
        let t = module.weight_basic.val();
        let y = transformation(t, x);

        let mut grads = y.backward();
        module.weight_basic.grad_remove(&mut grads)
    }
}

#[cfg(feature = "distributed")]
mod grad_distributed {
    use burn_std::device::{Device, DeviceId};
    use burn_tensor::Tolerance;
    use burn_tensor::backend::distributed::DistributedBackend;
    use burn_tensor::backend::distributed::{DistributedParamId, ReduceOperation};
    use burn_tensor::{TensorData, backend::AutodiffBackend};
    use rand::{
        SeedableRng,
        rngs::{StdRng, SysRng},
    };
    use serial_test::serial;
    use std::sync::mpsc::{Receiver, Sender};

    use super::*;

    #[test]
    #[serial]
    fn sharded_module_should_sync_gradients_sum() {
        compare_sync_gradients::<TestAutodiffBackend>(ReduceOperation::Sum, |weights, x| {
            weights.matmul(x)
        });
    }

    #[test]
    #[serial]
    fn sharded_module_should_sync_gradients_mean() {
        compare_sync_gradients::<TestAutodiffBackend>(ReduceOperation::Mean, |weights, x| {
            weights.matmul(x)
        });
    }

    #[test]
    #[serial]
    fn sharded_module_should_sync_gradients_sum_residual() {
        compare_sync_gradients::<TestAutodiffBackend>(ReduceOperation::Sum, |weights, x| {
            let y = weights.clone().matmul(x);
            y.add(weights)
        });
    }

    #[test]
    #[serial]
    fn sharded_module_should_sync_gradients_mean_residual() {
        compare_sync_gradients::<TestAutodiffBackend>(ReduceOperation::Mean, |weights, x| {
            let y = weights.clone().matmul(x);
            y.add(weights)
        });
    }

    #[test]
    #[serial]
    fn sharded_module_should_sync_gradients_sum_activation() {
        compare_sync_gradients::<TestAutodiffBackend>(ReduceOperation::Sum, |weights, x| {
            let y = weights.clone().matmul(x);
            let y = y.add(weights);
            burn_tensor::activation::relu(y)
        });
    }

    #[test]
    #[serial]
    fn sharded_module_should_sync_gradients_mean_activation() {
        compare_sync_gradients::<TestAutodiffBackend>(ReduceOperation::Mean, |weights, x| {
            let y = weights.clone().matmul(x);
            let y = y.add(weights);
            burn_tensor::activation::relu(y)
        });
    }

    #[test]
    #[serial]
    fn sharded_module_should_sync_gradients_sum_diamond_graph() {
        compare_sync_gradients::<TestAutodiffBackend>(ReduceOperation::Sum, |weights, x| {
            let left = weights.clone().matmul(x.clone().mul_scalar(2));
            let right = weights.clone().matmul(x.clone().exp());
            Tensor::cat(vec![left, right], 0)
        });
    }

    #[test]
    #[serial]
    fn sharded_module_should_sync_gradients_mean_diamond_graph() {
        compare_sync_gradients::<TestAutodiffBackend>(ReduceOperation::Mean, |weights, x| {
            let left = weights.clone().matmul(x.clone().mul_scalar(2));
            let right = weights.clone().matmul(x.clone().exp());
            Tensor::cat(vec![left, right], 0)
        });
    }

    fn compare_sync_gradients<B: AutodiffBackend + DistributedBackend>(
        op: ReduceOperation,
        transformation: fn(Tensor<B, 2>, Tensor<B, 2>) -> Tensor<B, 2>,
    ) {
        use burn_tensor::backend::distributed::DistributedConfig;

        const NUM_ITERATIONS: usize = 100;
        let type_id = 0u16;

        // let device_count = <B as Backend>::device_count(type_id);
        let device_count = 2;
        let devices = create_devices::<B::Device>(type_id, device_count);
        let module = ModuleBasic::<B>::new(&devices[0]);
        let (synced_senders, synced_receivers) = (1..device_count)
            .map(|_| std::sync::mpsc::channel())
            .unzip();
        let (original_senders, original_receivers) = (1..device_count)
            .map(|_| std::sync::mpsc::channel())
            .unzip();

        let config = DistributedConfig { all_reduce_op: op };
        B::start_communication_server(devices.as_slice(), config);

        let join_handles = spawn_peer_threads(
            &module,
            &devices,
            synced_senders,
            original_senders,
            synced_receivers,
            original_receivers,
            transformation,
            NUM_ITERATIONS,
        );

        for handle in join_handles {
            handle.join().unwrap();
        }

        B::close_communication_server(&devices[0]);
    }

    fn create_devices<D: Device>(type_id: u16, count: usize) -> Vec<D> {
        (0..count)
            .map(|i| D::from_id(DeviceId::new(type_id, i as u32)))
            .collect()
    }

    fn create_channels(
        device_count: usize,
    ) -> (Vec<Sender<TensorData>>, Vec<Receiver<TensorData>>) {
        (1..device_count)
            .map(|_| std::sync::mpsc::channel())
            .unzip()
    }

    fn spawn_peer_threads<B: AutodiffBackend>(
        module: &ModuleBasic<B>,
        devices: &[<B as Backend>::Device],
        synced_senders: Vec<Sender<TensorData>>,
        original_senders: Vec<Sender<Tensor<B::InnerBackend, 2>>>,
        synced_receivers: Vec<Receiver<TensorData>>,
        original_receivers: Vec<Receiver<Tensor<B::InnerBackend, 2>>>,
        transformation: fn(Tensor<B, 2>, Tensor<B, 2>) -> Tensor<B, 2>,
        num_iter: usize,
    ) -> Vec<std::thread::JoinHandle<()>> {
        let mut handles = vec![];

        // Spawn main peer thread (id=0)
        let module_clone = module.clone();
        let device = devices[0].clone();
        handles.push(std::thread::spawn(move || {
            run_peer_sharded(
                &module_clone,
                None,
                None,
                transformation,
                device,
                num_iter,
                true,
                synced_receivers,
                original_receivers,
            )
        }));

        // Spawn worker peer threads (id > 0)
        for i in 1..devices.len() {
            let module_clone = module.clone();
            let device = devices[i].clone();
            let synced_sender = Some(synced_senders[i - 1].clone());
            let original_sender = Some(original_senders[i - 1].clone());
            handles.push(std::thread::spawn(move || {
                run_peer_sharded(
                    &module_clone,
                    synced_sender,
                    original_sender,
                    transformation,
                    device,
                    num_iter,
                    false,
                    vec![],
                    vec![],
                )
            }));
        }

        handles
    }

    pub fn run_peer_sharded<B: AutodiffBackend>(
        module: &ModuleBasic<B>,
        synced_sender: Option<Sender<TensorData>>,
        original_sender: Option<Sender<Tensor<B::InnerBackend, 2>>>,
        transformation: fn(Tensor<B, 2>, Tensor<B, 2>) -> Tensor<B, 2>,
        device: B::Device,
        num_iter: usize,
        is_main: bool,
        synced_recvs: Vec<Receiver<TensorData>>,
        original_recvs: Vec<Receiver<Tensor<B::InnerBackend, 2>>>,
    ) {
        let mut module = module.clone().fork(&device);

        for _ in 0..2 {
            module = set_distributed(&module, &device);
            let (grads_synced, grads_original) = calculate_grads(&module, transformation);
            let data = grads_synced.unwrap().to_data();
            if !is_main {
                original_sender
                    .clone()
                    .unwrap()
                    .send(grads_original.unwrap())
                    .unwrap();
                synced_sender.clone().unwrap().send(data).unwrap();
            } else {
                let mut expected = grads_original.clone().unwrap();
                let device = expected.device();
                for r in original_recvs.iter().by_ref() {
                    expected = expected.add(r.recv().unwrap().to_device(&device));
                }
                for r in synced_recvs.iter().by_ref() {
                    let data_other = r.recv().unwrap();
                    println!(
                        "expected : {:?}\n",
                        expected.to_data().to_vec::<f32>().unwrap()
                    );
                    println!("data : {:?}\n", data.to_vec::<f32>().unwrap());
                    println!("data_other : {:?}\n", data_other.to_vec::<f32>().unwrap());
                    data.assert_approx_eq::<f32>(&expected.to_data(), Tolerance::default());
                }
            }
        }
    }

    fn set_distributed<B: AutodiffBackend>(
        module: &ModuleBasic<B>,
        device: &B::Device,
    ) -> ModuleBasic<B> {
        let mut module = module.clone().fork(&device);
        let (id, tensor, mapper) = module.weight_basic.consume();
        let tensor = tensor.set_distributed(DistributedParamId::from(id.val()));
        module.weight_basic = Param::from_mapped_value(id, tensor, mapper);
        module
    }

    fn calculate_grads<B: AutodiffBackend>(
        module: &ModuleBasic<B>,
        transformation: fn(Tensor<B, 2>, Tensor<B, 2>) -> Tensor<B, 2>,
    ) -> (
        Option<Tensor<B::InnerBackend, 2>>,
        Option<Tensor<B::InnerBackend, 2>>,
    ) {
        let device = module.weight_basic.device();
        let data = TensorData::random::<f32, _, _>(
            module.weight_basic.shape(),
            burn_tensor::Distribution::Default,
            &mut StdRng::try_from_rng(&mut SysRng).unwrap(),
        );
        let x = Tensor::from_data(data.clone(), &device).require_grad();
        let t = module.weight_basic.val();
        let y = transformation(t, x);

        let mut grads = y.backward();
        let grads_synced = module.weight_basic.grad_remove(&mut grads);

        // Kind of hacky, but running the backward pass again without marking the tensor as distributed will not sync gradients.
        // We can use this to compute the expected sum.
        let x = Tensor::from_data(data, &device).require_grad();
        let t = module.weight_basic.val();
        let y = transformation(t, x);
        let mut grads = y.backward();
        let grads_original = module.weight_basic.grad_remove(&mut grads);
        (grads_synced, grads_original)
    }
}
