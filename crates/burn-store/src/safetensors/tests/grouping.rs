//! Entries a [`TensorGrouping`] pairs read as one quantized tensor.

use burn_core as burn;

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use burn_core::module::{Module, Param};
use burn_core::tensor::quantization::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
use burn_core::tensor::{DType, Device, Tensor, TensorData};
use safetensors::tensor::TensorView;

use crate::{
    ModuleSnapshot, QuantizedGroup, SafetensorsDtype, SafetensorsEntry, SafetensorsStore,
    SafetensorsStoreError, TensorGrouping,
};

#[derive(Module, Debug)]
struct Projection {
    weight: Param<Tensor<2>>,
}

#[derive(Module, Debug)]
struct Model {
    layer: Projection,
}

const ROWS: usize = 32;
const COLUMNS: usize = 64;

/// A grouping as a DeepSeek-style release needs one: each `F8_E4M3` `<p>.weight` read with its
/// `<p>.scale`, one `ue8m0` scale per 32×32 block.
struct WeightWithScale;

impl TensorGrouping for WeightWithScale {
    fn group(
        &self,
        entries: &[SafetensorsEntry],
    ) -> Result<Vec<QuantizedGroup>, SafetensorsStoreError> {
        let scheme = QuantScheme::default()
            .with_value(QuantValue::E4M3)
            .with_store(QuantStore::Native)
            .per_block([32, 32], ScaleDtype::UE8M0);
        Ok(entries
            .iter()
            .filter(|entry| entry.dtype == SafetensorsDtype::F8_E4M3)
            .filter_map(|entry| {
                let prefix = entry.name.strip_suffix(".weight")?;
                Some(QuantizedGroup {
                    name: entry.name.clone(),
                    codes: entry.name.clone(),
                    scales: alloc::format!("{prefix}.scale"),
                    tensor_scale: None,
                    scheme,
                    shape: entry.shape.clone(),
                })
            })
            .collect())
    }
}

/// The `e4m3` codes of 1, 2, -1 and 0.5, cycled over the weight.
const CODES: [(u8, f32); 4] = [(0x38, 1.0), (0x40, 2.0), (0xb8, -1.0), (0x30, 0.5)];

/// The `ue8m0` codes of the two blocks' scales, 1 and 4.
const SCALES: [(u8, f32); 2] = [(127, 1.0), (129, 4.0)];

/// A file holding `layer.weight` at `F8_E4M3`, `[ROWS, COLUMNS]`, and `layer.scale` at
/// `F8_E8M0`, one scale per 32×32 block, and the values the pair stands for.
fn fp8_file() -> (Vec<u8>, Vec<f32>) {
    let codes: Vec<u8> = (0..ROWS * COLUMNS)
        .map(|index| CODES[index % CODES.len()].0)
        .collect();
    let scales: Vec<u8> = SCALES.iter().map(|(code, _)| *code).collect();
    let values = (0..ROWS * COLUMNS)
        .map(|index| CODES[index % CODES.len()].1 * SCALES[index % COLUMNS / 32].1)
        .collect();
    let tensors = [
        (
            "layer.weight",
            TensorView::new(
                SafetensorsDtype::F8_E4M3,
                alloc::vec![ROWS, COLUMNS],
                &codes,
            )
            .unwrap(),
        ),
        (
            "layer.scale",
            TensorView::new(SafetensorsDtype::F8_E8M0, alloc::vec![1, 2], &scales).unwrap(),
        ),
    ];
    let bytes = safetensors::serialize(tensors, None).unwrap();
    (bytes, values)
}

fn model(device: &Device) -> Model {
    Model {
        layer: Projection {
            weight: Param::from_tensor(Tensor::zeros([ROWS, COLUMNS], device)),
        },
    }
}

/// An `F8_E4M3` weight and its `F8_E8M0` block scales, grouped, load into a float parameter as
/// one quantized tensor that reads back as the values the pair stands for.
#[test]
fn a_grouped_fp8_weight_loads_as_its_dequantized_values() {
    let device = Device::default();
    let (bytes, values) = fp8_file();
    let mut store = SafetensorsStore::from_bytes(Some(bytes)).with_grouping(WeightWithScale);
    let mut module = model(&device);

    module.load_from(&mut store).unwrap();

    let weight = module.layer.weight.val();
    assert!(matches!(weight.dtype(), DType::QFloat(_)));
    weight
        .dequantize()
        .into_data()
        .assert_eq(&TensorData::new(values, [ROWS, COLUMNS]), false);
}

/// The scale a group claims is no tensor of its own, and the group reads under its own name.
#[test]
fn a_grouped_file_lists_the_group_and_not_its_scale() {
    let (bytes, _) = fp8_file();
    let mut store = SafetensorsStore::from_bytes(Some(bytes)).with_grouping(WeightWithScale);

    let keys = crate::ModuleStore::keys(&mut store).unwrap();

    assert_eq!(keys, ["layer.weight".to_string()]);
}

/// Without a grouping, an FP8 entry fails the load, naming the tensor and the way out.
#[test]
fn an_ungrouped_fp8_file_fails_naming_the_tensor() {
    let device = Device::default();
    let (bytes, _) = fp8_file();
    let mut store = SafetensorsStore::from_bytes(Some(bytes));
    let mut module = model(&device);

    let error: String = module.load_from(&mut store).unwrap_err().to_string();

    assert!(error.contains("`layer"), "{error}");
    assert!(error.contains("with_grouping"), "{error}");
}

/// A group whose scales do not fit its shape and scheme fails when the file is opened, naming
/// the tensor, rather than when its data is drawn.
#[test]
fn a_group_with_the_wrong_scales_fails_naming_the_tensor() {
    struct WrongBlocks;

    impl TensorGrouping for WrongBlocks {
        fn group(
            &self,
            entries: &[SafetensorsEntry],
        ) -> Result<Vec<QuantizedGroup>, SafetensorsStoreError> {
            let mut groups = WeightWithScale.group(entries)?;
            for group in &mut groups {
                group.scheme = group.scheme.per_block([32, 16], ScaleDtype::UE8M0);
            }
            Ok(groups)
        }
    }
    let (bytes, _) = fp8_file();
    let mut store = SafetensorsStore::from_bytes(Some(bytes)).with_grouping(WrongBlocks);

    let error = crate::ModuleStore::keys(&mut store)
        .unwrap_err()
        .to_string();

    assert!(error.contains("`layer.weight`"), "{error}");
    assert!(error.contains("Scales"), "{error}");
}
