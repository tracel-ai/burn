use burn::module::Initializer;
use burn::module::{Module, Param};
use burn::tensor::{Int, Tensor};
use burn_core as burn;
use burn_tensor::Device;

#[derive(Module, Debug)]
pub struct ModuleBasic {
    weight_basic: Param<Tensor<2>>,
}

#[derive(Module, Debug)]
#[allow(unused)]
struct ModuleTensorConstInt {
    weight_basic: Tensor<2, Int>,
}

impl ModuleBasic {
    fn new(device: &Device) -> Self {
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
struct ModuleWithConstGeneric<const N: usize> {
    modules: [ModuleBasic; N],
}

#[derive(Module, Debug)]
struct ModuleWithGenericModule<M> {
    module: M,
}

#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
enum ModuleEnum {
    Basic(ModuleBasic),
    Composed(ModuleComposed),
}

#[derive(Module, Debug)]
#[allow(unused)]
enum ModuleEnumNested {
    AnotherEnum(ModuleEnum),
}

#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
enum ModuleEnumWithGenericModule<M: Module> {
    Basic(ModuleBasic),
    Generic(ModuleWithGenericModule<M>),
}

#[derive(Module, Debug)]
pub struct ModuleComposed {
    weight: Param<Tensor<2>>,
    basic: ModuleBasic,
    tuple: (ModuleBasic, ModuleBasic),
}

impl ModuleComposed {
    fn new(device: &Device) -> Self {
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
pub struct ModuleWithAttributes<M: Module, N> {
    /// A normal parameter.
    weight: Param<Tensor<2>>,
    /// A nested module.
    nested: ModuleEnumWithGenericModule<M>,
    /// By default, primitives were not persistent (same as `#[module(skip)]`).
    other_prob: f64,
    /// By default, tensors were not persistent and not visited/mapped (same as `#[module(skip)]`).
    tensor: Tensor<1>,
    /// A field that is recomputed at runtime.
    #[module(skip)]
    cached_mask: Option<Tensor<2>>,
    /// A field that contains some debug state.
    debug_state: String,
    /// Hint required: this generic is NOT a module.
    #[module(skip)]
    config: N,
}

impl ModuleWithAttributes<ModuleBasic, PaddingConfig> {
    fn new(device: &Device) -> Self {
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
        record::{PrecisionSettings, Record},
    };

    use super::*;

    type RecordItem<M, S> = <<M as Module>::Record as Record>::Item<S>;

    fn implements_clone<T: Clone>() {}

    fn basic_implements_clone<S: PrecisionSettings>() {
        implements_clone::<RecordItem<ModuleBasic, S>>();
        implements_clone::<RecordItem<ModuleComposed, S>>();
    }

    fn generic_implements_clone<S, M>()
    where
        S: PrecisionSettings,
        M: Module + ModuleDisplay,
        RecordItem<M, S>: Clone,
    {
        implements_clone::<RecordItem<ModuleWithGenericModule<M>, S>>();
        implements_clone::<RecordItem<ModuleEnumWithGenericModule<M>, S>>();
    }
}

pub fn test_device() -> Device {
    burn_tensor::Device::flex()
}

mod state {
    use burn_core::module::EmptyRecord;

    use super::*;

    #[test]
    fn should_load_from_record_basic() {
        let device = test_device();
        let module_1 = ModuleBasic::new(&device);
        let mut module_2 = ModuleBasic::new(&device);

        // Access module_1 to trigger initialization before cloning.
        // Cloning an uninitialized module preserves lazy state (no memory allocation),
        // so we need to initialize first if we want the clone to have the same values.
        assert_ne!(
            module_1.weight_basic.to_data(),
            module_2.weight_basic.to_data()
        );

        let state_1 = module_1.clone().into_record();

        module_2 = module_2.load_record(state_1);

        assert_eq!(
            module_1.weight_basic.to_data(),
            module_2.weight_basic.to_data()
        );
    }

    #[test]
    fn should_load_from_record_compose() {
        let device = test_device();
        let module_1 = ModuleComposed::new(&device);
        let mut module_2 = ModuleComposed::new(&device);
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
        let device = test_device();
        let module_1 = ModuleEnum::Basic(ModuleBasic::new(&device));
        let mut module_2 = ModuleEnum::Basic(ModuleBasic::new(&device));

        // Trigger initialization before cloning so clone has the same values.
        let ModuleEnum::Basic(ref module_1_basic) = module_1 else {
            panic!("Invalid module type")
        };
        let ModuleEnum::Basic(module_2_basic) = module_2.clone() else {
            panic!("Invalid module type")
        };
        assert_ne!(
            module_1_basic.weight_basic.to_data(),
            module_2_basic.weight_basic.to_data()
        );

        let state_1 = module_1.clone().into_record();

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
        let device = test_device();
        let mut module_1 = ModuleWithAttributes::new(&device);
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
        let device = test_device();
        let module_1 = ModuleWithConstGeneric {
            modules: [ModuleBasic::new(&device), ModuleBasic::new(&device)],
        };
        let mut module_2 = ModuleWithConstGeneric {
            modules: [ModuleBasic::new(&device), ModuleBasic::new(&device)],
        };

        // Trigger initialization before cloning so clone has the same values.
        assert_ne!(
            module_1.modules[0].weight_basic.to_data(),
            module_2.modules[0].weight_basic.to_data(),
        );
        assert_ne!(
            module_1.modules[1].weight_basic.to_data(),
            module_2.modules[1].weight_basic.to_data(),
        );

        let state_1 = module_1.clone().into_record();

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
        let device = test_device();
        let module_1 = ModuleEnum::Basic(ModuleBasic::new(&device));
        let module_2 = ModuleEnum::Composed(ModuleComposed::new(&device));
        let state_1 = module_1.clone().into_record();

        module_2.load_record(state_1);
    }
}

mod lazy_clone {
    use burn_tensor::Device;

    use super::*;

    #[test]
    fn clone_uninitialized_param_should_not_trigger_init() {
        let device = test_device();
        let module = ModuleBasic::new(&device);

        // Module starts uninitialized (lazy).
        assert!(!module.weight_basic.is_initialized());

        // Cloning should preserve the lazy state, not trigger initialization.
        let cloned = module.clone();
        assert!(!module.weight_basic.is_initialized());
        assert!(!cloned.weight_basic.is_initialized());
    }

    #[test]
    fn clone_initialized_param_should_share_values() {
        let device = test_device();
        let module = ModuleBasic::new(&device);

        // Force initialization by accessing the tensor.
        let _ = module.weight_basic.to_data();
        assert!(module.weight_basic.is_initialized());

        // Clone of an initialized param should have the same values.
        let cloned = module.clone();
        assert_eq!(module.weight_basic.to_data(), cloned.weight_basic.to_data());
    }

    #[test]
    fn lazy_clone_should_produce_valid_tensor_on_access() {
        let device = test_device();
        let module = ModuleBasic::new(&device);
        let cloned = module.clone();

        // Both are uninitialized.
        assert!(!module.weight_basic.is_initialized());
        assert!(!cloned.weight_basic.is_initialized());

        // Accessing the clone should produce a valid tensor with the right shape.
        let data = cloned.weight_basic.to_data();
        assert_eq!(data.shape, [20, 20].into());

        data.assert_eq(&module.weight_basic.val().into_data(), true);
    }

    #[test]
    fn lazy_clone_and_original_have_same_init() {
        let device = test_device();
        let module = ModuleBasic::new(&device);
        let cloned = module.clone();

        let clone_data = cloned.weight_basic.to_data();
        let orig_data = module.weight_basic.to_data();

        assert_eq!(clone_data.shape, [20, 20].into());
        assert_eq!(orig_data.shape, [20, 20].into());
        assert_eq!(clone_data, orig_data);
    }

    #[test]
    fn lazy_clone_deref_should_trigger_init() {
        let device = test_device();
        let module = ModuleBasic::new(&device);
        let cloned = module.clone();

        // Access via Deref (shape() uses Deref, not val()) on the clone.
        let shape = cloned.weight_basic.shape();
        assert_eq!(shape, [20, 20].into());
        assert!(cloned.weight_basic.is_initialized());
        assert!(module.weight_basic.is_initialized());
    }

    #[test]
    fn init_mapper_on_lazy_clone_should_not_affect_original() {
        use burn::module::ParamId;
        use burn::tensor::Shape;

        let device: Device = test_device();

        // Create two uninitialized params from the same init function.
        let param: Param<Tensor<2>> = Param::uninitialized(
            ParamId::new(),
            move |d, _| Tensor::random([4, 4], Default::default(), d),
            device,
            false,
            Shape::from([4, 4]),
        );

        let mut cloned = param.clone();

        // Apply init_mapper on the clone to double all values.
        cloned = cloned.init_mapper(|t| t.mul_scalar(2.0));

        // Random tensor should still have the same initialization point, but * 2.0
        cloned
            .val()
            .div_scalar(2.0)
            .into_data()
            .assert_approx_eq::<f32>(&param.val().into_data(), Default::default());
    }

    #[test]
    fn load_record_into_uninitialized_module_should_work() {
        let device = test_device();
        let module_1 = ModuleBasic::new(&device);

        // Initialize module_1 so we have a record to load.
        let _ = module_1.weight_basic.to_data();
        let record = module_1.clone().into_record();

        // Create a fresh uninitialized module and load weights into it.
        let module_2 = ModuleBasic::new(&device);
        assert!(!module_2.weight_basic.is_initialized());

        let module_2 = module_2.load_record(record);

        // After loading, the param should be initialized with the loaded values.
        assert_eq!(
            module_1.weight_basic.to_data(),
            module_2.weight_basic.to_data()
        );
    }
}

mod num_params {
    use super::*;

    #[test]
    fn should_calculate_num_params_basic() {
        let device = test_device();
        let module = ModuleBasic::new(&device);
        assert_eq!(20 * 20, module.num_params());
    }

    #[test]
    fn should_output_state_composed() {
        let device = test_device();
        let module = ModuleComposed::new(&device);
        assert_eq!(4 * 20 * 20, module.num_params());
    }

    #[test]
    fn should_calculate_num_params_enum() {
        let device = test_device();
        let module = ModuleEnum::Basic(ModuleBasic::new(&device));
        assert_eq!(20 * 20, module.num_params());

        let module = ModuleEnum::Composed(ModuleComposed::new(&device));
        assert_eq!(4 * 20 * 20, module.num_params());
    }

    #[test]
    fn should_calculate_num_params_based_on_attributes() {
        let device = test_device();
        let module = ModuleWithAttributes::new(&device);
        assert_eq!(20 * 20 * 2, module.num_params());
    }
}

#[cfg(all(feature = "std", feature = "autodiff"))]
mod require_grad {
    use burn_tensor::TensorData;
    use rand::{
        SeedableRng,
        rngs::{StdRng, SysRng},
    };

    use super::*;

    #[test]
    fn should_have_grad_by_default() {
        let device = test_device().autodiff();
        let module = ModuleBasic::new(&device);
        let grad_x = calculate_grads(&module, |weights, x| weights.matmul(x));

        assert!(grad_x.is_some());
    }

    #[test]
    fn should_have_no_grad_after_no_grad() {
        let device = test_device().autodiff();
        let module = ModuleBasic::new(&device).no_grad();
        let grad_x = calculate_grads(&module, |weights, x| weights.matmul(x));

        assert!(grad_x.is_none());
    }

    #[test]
    fn should_have_grad_when_from_record() {
        let device = test_device().autodiff();
        let module = ModuleBasic::new(&device);
        let record = ModuleBasicRecord {
            weight_basic: module.weight_basic.clone(), // Even when param is no_grad,
        };
        let module = module.load_record(record);
        let grad_x = calculate_grads(&module, |weights, x| weights.matmul(x));

        assert!(grad_x.is_some());
    }

    fn calculate_grads(
        module: &ModuleBasic,
        transformation: fn(Tensor<2>, Tensor<2>) -> Tensor<2>,
    ) -> Option<Tensor<2>> {
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

#[cfg(feature = "cuda")]
mod grad_distributed {
    use burn_tensor::TensorData;
    use burn_tensor::distributed::{DistributedContext, ReduceOperation};
    use burn_tensor::{Device, DeviceType, Tolerance};
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
        compare_sync_gradient(ReduceOperation::Sum, |weights, x| weights.matmul(x));
    }

    #[test]
    #[serial]
    fn sharded_module_should_sync_gradients_mean() {
        compare_sync_gradient(ReduceOperation::Mean, |weights, x| weights.matmul(x));
    }

    #[test]
    #[serial]
    fn sharded_module_should_sync_gradients_sum_residual() {
        compare_sync_gradient(ReduceOperation::Sum, |weights, x| {
            let y = weights.clone().matmul(x);
            y.add(weights)
        });
    }

    #[test]
    #[serial]
    fn sharded_module_should_sync_gradients_mean_residual() {
        compare_sync_gradient(ReduceOperation::Mean, |weights, x| {
            let y = weights.clone().matmul(x);
            y.add(weights)
        });
    }

    #[test]
    #[serial]
    fn sharded_module_should_sync_gradients_sum_activation() {
        compare_sync_gradient(ReduceOperation::Sum, |weights, x| {
            let y = weights.clone().matmul(x);
            let y = y.add(weights);
            burn_tensor::activation::relu(y)
        });
    }

    #[test]
    #[serial]
    fn sharded_module_should_sync_gradients_mean_activation() {
        compare_sync_gradient(ReduceOperation::Mean, |weights, x| {
            let y = weights.clone().matmul(x);
            let y = y.add(weights);
            burn_tensor::activation::relu(y)
        });
    }

    #[test]
    #[serial]
    fn sharded_module_should_sync_gradients_sum_diamond_graph() {
        compare_sync_gradient(ReduceOperation::Sum, |weights, x| {
            let left = weights.clone().matmul(x.clone().mul_scalar(2));
            let right = weights.clone().matmul(x.clone().exp());
            Tensor::cat(vec![left, right], 0)
        });
    }

    #[test]
    #[serial]
    fn sharded_module_should_sync_gradients_mean_diamond_graph() {
        compare_sync_gradient(ReduceOperation::Mean, |weights, x| {
            let left = weights.clone().matmul(x.clone().mul_scalar(2));
            let right = weights.clone().matmul(x.clone().exp());
            Tensor::cat(vec![left, right], 0)
        });
    }

    fn compare_sync_gradient(
        op: ReduceOperation,
        transformation: fn(Tensor<2>, Tensor<2>) -> Tensor<2>,
    ) {
        use burn_tensor::distributed::DistributedConfig;

        const NUM_ITERATIONS: usize = 100;
        let devices = Device::enumerate(DeviceType::Cuda).autodiff().into_vec();

        let module = ModuleBasic::new(&devices[0]);
        let (synced_senders, synced_receivers): (
            Vec<Sender<TensorData>>,
            Vec<Receiver<TensorData>>,
        ) = (0..devices.len())
            .map(|_| std::sync::mpsc::channel())
            .unzip();
        let (original_senders, original_receivers): (
            Vec<Sender<TensorData>>,
            Vec<Receiver<TensorData>>,
        ) = (0..devices.len())
            .map(|_| std::sync::mpsc::channel())
            .unzip();

        let config = DistributedConfig { all_reduce_op: op };
        let _context = DistributedContext::init(devices.clone(), config);

        let join_handles = spawn_peer_threads(
            &module,
            &devices,
            synced_senders,
            original_senders,
            transformation,
            NUM_ITERATIONS,
        );

        for _ in 0..NUM_ITERATIONS {
            let device = devices.first().unwrap();
            let mut expected: Tensor<2> =
                Tensor::from_data(original_receivers.first().unwrap().recv().unwrap(), device);
            for r in original_receivers[1..].iter().by_ref() {
                let data = r.recv().unwrap();
                expected = expected.add(Tensor::from_data(data, device));
            }
            if op == ReduceOperation::Mean {
                expected = expected.div_scalar(original_receivers.len() as f32);
            }

            for r in synced_receivers.iter().by_ref() {
                let data = r.recv().unwrap();
                data.assert_approx_eq::<f32>(&expected.to_data(), Tolerance::default());
            }
        }

        for handle in join_handles {
            handle.join().unwrap();
        }

        // DistributedContext goes out of scope -> close_communication_server
    }

    fn spawn_peer_threads(
        module: &ModuleBasic,
        devices: &[Device],
        synced_senders: Vec<Sender<TensorData>>,
        original_senders: Vec<Sender<TensorData>>,
        transformation: fn(Tensor<2>, Tensor<2>) -> Tensor<2>,
        num_iter: usize,
    ) -> Vec<std::thread::JoinHandle<()>> {
        let mut handles = vec![];

        for i in 0..devices.len() {
            let module_clone = module.clone();
            let device = devices[i].clone();
            let synced_sender = synced_senders[i].clone();
            let original_sender = original_senders[i].clone();
            handles.push(std::thread::spawn(move || {
                run_peer_sharded(
                    &module_clone,
                    synced_sender,
                    original_sender,
                    transformation,
                    device,
                    num_iter,
                )
            }));
        }

        handles
    }

    pub fn run_peer_sharded(
        module: &ModuleBasic,
        synced_sender: Sender<TensorData>,
        original_sender: Sender<TensorData>,
        transformation: fn(Tensor<2>, Tensor<2>) -> Tensor<2>,
        device: Device,
        num_iter: usize,
    ) {
        let mut module = module.clone().fork(&device);

        for _ in 0..num_iter {
            module = set_distributed(&module, &device);
            let (grads_synced, grads_original) = calculate_grads(&module, transformation);

            let data = grads_original.unwrap().to_data();
            original_sender.clone().send(data).unwrap();

            let data = grads_synced.unwrap().to_data();
            synced_sender.clone().send(data).unwrap();
        }
    }

    fn set_distributed(module: &ModuleBasic, device: &Device) -> ModuleBasic {
        let mut module = module.clone().fork(&device);
        let (id, tensor, mapper) = module.weight_basic.consume();
        let tensor = tensor.set_distributed(id);
        module.weight_basic = Param::from_mapped_value(id, tensor, mapper);
        module
    }

    fn calculate_grads(
        module: &ModuleBasic,
        transformation: fn(Tensor<2>, Tensor<2>) -> Tensor<2>,
    ) -> (Option<Tensor<2>>, Option<Tensor<2>>) {
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
