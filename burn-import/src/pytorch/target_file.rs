use std::{marker::PhantomData, path::PathBuf};

use crate::pytorch::RecordType;

use super::{converter::NestedValue, Converter};

use burn::{
    module::ParamId,
    record::{
        BurnRecord, FullPrecisionSettings, HalfPrecisionSettings, NamedMpkFileRecorder,
        NamedMpkGzFileRecorder, PrecisionSettings, PrettyJsonFileRecorder, Recorder,
    },
    tensor::{DataSerialize, Element, ElementConversion},
};

use candle_core::{Tensor as CandleTensor, WithDType};
use half::{bf16, f16};
use serde::{
    ser::{SerializeMap, SerializeSeq},
    Serialize,
};

impl Converter {
    pub(super) fn convert_to_record(&self, map: NestedValue, out_file: PathBuf) {
        match self.record_type {
            RecordType::PrettyJson => {
                self.convert_to_pretty_json(map, out_file);
            }

            RecordType::NamedMpkGz => {
                self.convert_to_named_mpk_gz(map, out_file);
            }

            RecordType::NamedMpk => {
                self.convert_to_named_mpk(map, out_file);
            }
        }
    }

    /// Convert model weights from the `input` file and save them to the bincode `out_file`.
    pub(super) fn convert_to_pretty_json(&self, map: NestedValue, out_file: PathBuf) {
        if self.half_precision {
            PrettyJsonFileRecorder::<HalfPrecisionSettings>::new()
                .save_item(
                    BurnRecord::new::<PrettyJsonFileRecorder<HalfPrecisionSettings>>(StructMap::<
                        HalfPrecisionSettings,
                    >::new(
                        map
                    )),
                    out_file,
                )
                .unwrap();
        } else {
            PrettyJsonFileRecorder::<FullPrecisionSettings>::new()
                .save_item(
                    BurnRecord::new::<PrettyJsonFileRecorder<FullPrecisionSettings>>(StructMap::<
                        FullPrecisionSettings,
                    >::new(
                        map
                    )),
                    out_file,
                )
                .unwrap();
        }
    }

    /// Convert model weights from the `input` file and save them to the bincode `out_file`.
    pub(super) fn convert_to_named_mpk_gz(&self, map: NestedValue, out_file: PathBuf) {
        if self.half_precision {
            NamedMpkGzFileRecorder::<HalfPrecisionSettings>::new()
                .save_item(
                    BurnRecord::new::<NamedMpkGzFileRecorder<HalfPrecisionSettings>>(StructMap::<
                        HalfPrecisionSettings,
                    >::new(
                        map
                    )),
                    out_file,
                )
                .unwrap();
        } else {
            NamedMpkGzFileRecorder::<FullPrecisionSettings>::new()
                .save_item(
                    BurnRecord::new::<NamedMpkGzFileRecorder<FullPrecisionSettings>>(StructMap::<
                        FullPrecisionSettings,
                    >::new(
                        map
                    )),
                    out_file,
                )
                .unwrap();
        }
    }

    /// Convert model weights from the `input` file and save them to the bincode `out_file`.
    pub(super) fn convert_to_named_mpk(&self, map: NestedValue, out_file: PathBuf) {
        if self.half_precision {
            NamedMpkFileRecorder::<HalfPrecisionSettings>::new()
                .save_item(
                    BurnRecord::new::<NamedMpkFileRecorder<HalfPrecisionSettings>>(StructMap::<
                        HalfPrecisionSettings,
                    >::new(
                        map
                    )),
                    out_file,
                )
                .unwrap();
        } else {
            NamedMpkFileRecorder::<FullPrecisionSettings>::new()
                .save_item(
                    BurnRecord::new::<NamedMpkFileRecorder<FullPrecisionSettings>>(StructMap::<
                        FullPrecisionSettings,
                    >::new(
                        map
                    )),
                    out_file,
                )
                .unwrap();
        }
    }
}

/// A struct to serialize a nested map/vector of tensors
///
/// StructMap is used to represent a nested map/vector of tensors and treat Struct as map
/// for de/serialization purposes.
///
/// Notably, this approach is utilized by serialization formats such as PrettyJson, NamedMpk,
/// and NamedMpkGz.
///
/// # Notes
///
/// Mpk and Bincode cannot use this method because they do not support serializing maps.
/// Instead, they use the `StructTuple` serialization strategy (to avoid memory overhead presumably).
struct StructMap<PS: PrecisionSettings>(NestedValue, PhantomData<PS>);

impl<PS: PrecisionSettings> StructMap<PS> {
    fn new(value: NestedValue) -> Self {
        Self(value, PhantomData::<PS>)
    }
}

impl<PS: PrecisionSettings> serde::Serialize for StructMap<PS> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self.0 {
            NestedValue::Tensor(ref tensor) => serialize_tensor::<S, PS>(tensor, serializer),

            NestedValue::Map(ref map) => {
                let mut map_serializer = serializer.serialize_map(Some(map.len()))?;
                for (name, value) in map.iter() {
                    map_serializer
                        .serialize_entry(&name.to_string(), &StructMap::<PS>::new(value.clone()))?;
                }
                map_serializer.end()
            }
            NestedValue::Vec(ref vec) => {
                let mut vec_serializer = serializer.serialize_seq(Some(vec.len()))?;
                for value in vec.iter() {
                    vec_serializer.serialize_element(&StructMap::<PS>::new(value.clone()))?;
                }
                vec_serializer.end()
            }
        }
    }
}

///  Define a parameter.
#[derive(new, Debug, Clone, Serialize)]
pub struct Param<T> {
    pub id: String,
    pub param: T,
}

/// Serializes a candle tensor.
///
/// Tensors are wrapped in a `Param` struct (learnable parameters) and serialized as a `DataSerialize` struct.
///
/// Values are serialized as `FloatElem` or `IntElem` depending on the precision settings.
fn serialize_tensor<S, PS>(tensor: &CandleTensor, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
    PS: PrecisionSettings,
{
    let shape = tensor.shape().clone().into_dims();
    let flatten = tensor.flatten_all().unwrap();
    let param_id = ParamId::new().into_string();

    match tensor.dtype() {
        candle_core::DType::U8 => {
            serialize_data::<u8, PS::IntElem, _>(flatten, shape, param_id, serializer)
        }
        candle_core::DType::U32 => {
            serialize_data::<u32, PS::IntElem, _>(flatten, shape, param_id, serializer)
        }
        candle_core::DType::I64 => {
            serialize_data::<i64, PS::IntElem, _>(flatten, shape, param_id, serializer)
        }
        candle_core::DType::BF16 => {
            serialize_data::<bf16, PS::FloatElem, _>(flatten, shape, param_id, serializer)
        }
        candle_core::DType::F16 => {
            serialize_data::<f16, PS::FloatElem, _>(flatten, shape, param_id, serializer)
        }
        candle_core::DType::F32 => {
            serialize_data::<f32, PS::FloatElem, _>(flatten, shape, param_id, serializer)
        }
        candle_core::DType::F64 => {
            serialize_data::<f64, PS::FloatElem, _>(flatten, shape, param_id, serializer)
        }
    }
}

/// Serializes a candle tensor data.
fn serialize_data<T, E, S>(
    flatten: CandleTensor,
    shape: Vec<usize>,
    param_id: String,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    E: Element + Serialize,
    S: serde::Serializer,
    T: WithDType + ElementConversion,
{
    let data: Vec<E> = flatten
        .to_vec1::<T>()
        .unwrap()
        .into_iter()
        .map(ElementConversion::elem)
        .collect();

    Param::new(param_id, DataSerialize::new(data, shape)).serialize(serializer)
}
