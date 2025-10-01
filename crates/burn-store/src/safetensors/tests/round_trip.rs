use crate::{ModuleSnapshot, SafetensorsStore};
use burn::nn::{Linear, LinearConfig};
use burn_core::module::{Module, Param};
use burn_tensor::Tensor;
use burn_tensor::backend::Backend;

type TestBackend = burn_ndarray::NdArray;

#[derive(Module, Debug)]
pub(super) struct ComplexModule<B: Backend> {
    pub encoder: EncoderModule<B>,
    pub decoder: DecoderModule<B>,
    pub layers: Vec<Linear<B>>,
}

#[derive(Module, Debug)]
pub(super) struct EncoderModule<B: Backend> {
    pub weight: Param<Tensor<B, 3>>,
    pub bias: Param<Tensor<B, 1>>,
    pub norm: Param<Tensor<B, 1>>,
}

#[derive(Module, Debug)]
pub(super) struct DecoderModule<B: Backend> {
    pub projection: Linear<B>,
    pub scale: Param<Tensor<B, 2>>,
}

impl<B: Backend> ComplexModule<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            encoder: EncoderModule {
                weight: Param::from_data(
                    [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                    device,
                ),
                bias: Param::from_data([0.1, 0.2, 0.3], device),
                norm: Param::from_data([1.0, 1.0, 1.0], device),
            },
            decoder: DecoderModule {
                projection: LinearConfig::new(4, 2).with_bias(true).init(device),
                scale: Param::from_data([[0.5, 0.5], [0.5, 0.5]], device),
            },
            layers: vec![
                LinearConfig::new(3, 4).with_bias(false).init(device),
                LinearConfig::new(4, 3).with_bias(true).init(device),
            ],
        }
    }

    pub fn new_zeros(device: &B::Device) -> Self {
        Self {
            encoder: EncoderModule {
                weight: Param::from_tensor(Tensor::zeros([2, 2, 2], device)),
                bias: Param::from_tensor(Tensor::zeros([3], device)),
                norm: Param::from_tensor(Tensor::zeros([3], device)),
            },
            decoder: DecoderModule {
                projection: LinearConfig::new(4, 2).with_bias(true).init(device),
                scale: Param::from_tensor(Tensor::zeros([2, 2], device)),
            },
            layers: vec![
                LinearConfig::new(3, 4).with_bias(false).init(device),
                LinearConfig::new(4, 3).with_bias(true).init(device),
            ],
        }
    }
}

#[test]
fn complex_module_round_trip() {
    let device = Default::default();
    let module1 = ComplexModule::<TestBackend>::new(&device);
    let mut module2 = ComplexModule::<TestBackend>::new_zeros(&device);

    // Save module1 using new store API
    let mut save_store = SafetensorsStore::from_bytes(None);
    module1.collect_to(&mut save_store).unwrap();

    // Load into module2
    let mut load_store = SafetensorsStore::from_bytes(None);
    if let SafetensorsStore::Memory(ref mut p) = load_store
        && let SafetensorsStore::Memory(ref p_save) = save_store
    {
        // Get Arc and extract data
        let data_arc = p_save.data().unwrap();
        p.set_data(data_arc.as_ref().clone());
    }
    let result = module2.apply_from(&mut load_store).unwrap();

    assert!(result.is_success());
    assert!(result.applied.len() > 5);
    assert_eq!(result.errors.len(), 0);

    // Verify data was imported correctly
    let module2_views = module2.collect();
    let encoder_weight = module2_views
        .iter()
        .find(|v| v.full_path() == "encoder.weight")
        .unwrap()
        .to_data();
    assert_eq!(encoder_weight.shape, vec![2, 2, 2]);
}
