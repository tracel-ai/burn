use std::collections::HashMap;

use burn::tensor::DType;
use serde::{
    Serialize,
    ser::{SerializeSeq, SerializeStruct, Serializer},
};

#[derive(Debug)]
pub struct SafetensorsTensorData {
    bytes: Vec<u8>,
    shape: Vec<usize>,
    dtype: DType,
}

impl safetensors::View for SafetensorsTensorData {
    fn dtype(&self) -> safetensors::Dtype {
        match self.dtype {
            burn::tensor::DType::F64 => safetensors::Dtype::F64,
            burn::tensor::DType::F32 => safetensors::Dtype::F32,
            burn::tensor::DType::F16 => safetensors::Dtype::F16,
            burn::tensor::DType::BF16 => safetensors::Dtype::BF16,
            burn::tensor::DType::I64 => safetensors::Dtype::I64,
            burn::tensor::DType::I32 => safetensors::Dtype::I32,
            burn::tensor::DType::I16 => safetensors::Dtype::I16,
            burn::tensor::DType::I8 => safetensors::Dtype::I8,
            burn::tensor::DType::U64 => safetensors::Dtype::U64,
            burn::tensor::DType::U32 => safetensors::Dtype::U32,
            burn::tensor::DType::U16 => safetensors::Dtype::U16,
            burn::tensor::DType::U8 => safetensors::Dtype::U8,
            burn::tensor::DType::Bool => safetensors::Dtype::BOOL,
            // `DType::Flex32` and `DType::QFloat` will not be accepted by the TensorDataSerializer
            _ => unimplemented!(),
        }
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> std::borrow::Cow<[u8]> {
        std::borrow::Cow::Borrowed(&self.bytes)
    }

    fn data_len(&self) -> usize {
        self.bytes.len()
    }
}

/// FlattenSerializer walks Modules storing their path.
/// When it encounters a TensorData struct, it uses TensorDataSerializer to extract it.
#[derive(Debug, Default)]
pub struct FlattenSerializer {
    map: HashMap<String, SafetensorsTensorData>,
    path: Vec<String>,
}

impl FlattenSerializer {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            path: Vec::new(),
        }
    }

    fn current_path(&self) -> String {
        self.path.join(".")
    }

    pub fn into_map(self) -> HashMap<String, SafetensorsTensorData> {
        // remove the ".param" suffix from each tensor path
        self.map
            .into_iter()
            .map(|(path, tensor)| {
                (
                    path.strip_suffix(".param").unwrap_or(&path).to_string(),
                    tensor,
                )
            })
            .collect()
    }
}

#[derive(Debug)]
pub enum Error {
    UnsupportedType,
    /// Indicates that the field being processed by the Serializer cannot be from a TensorData
    /// struct and thus aborts the serialization through TensorDataSerializer
    NotFromTensorData,
    Custom(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::UnsupportedType => write!(f, "Unsupported type TensorData"),
            Error::NotFromTensorData => write!(f, "The field cannot be from a TensorData"),
            Error::Custom(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for Error {}

impl serde::ser::Error for Error {
    fn custom<T: std::fmt::Display>(msg: T) -> Self {
        Error::Custom(msg.to_string())
    }
}

/// The FlattenSerializer serializer implementation ignores most data. It is only interested in
/// storing the paths Params of the Module and in its TensorData, which will be serlialized with
/// TensorDataSerializer
impl Serializer for &mut FlattenSerializer {
    type Ok = ();
    type Error = Error;

    type SerializeSeq = Self;
    type SerializeTuple = Self;
    type SerializeTupleStruct = Self;
    type SerializeTupleVariant = Self;
    type SerializeMap = Self;
    type SerializeStruct = Self;
    type SerializeStructVariant = Self;

    fn serialize_bool(self, _v: bool) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_i8(self, _v: i8) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_i16(self, _v: i16) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_i32(self, _v: i32) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_i64(self, _v: i64) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_u8(self, _v: u8) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_u16(self, _v: u16) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_u32(self, _v: u32) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_u64(self, _v: u64) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_f32(self, _v: f32) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_f64(self, _v: f64) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_char(self, _v: char) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_str(self, _v: &str) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_bytes(self, _v: &[u8]) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_some<T>(self, value: &T) -> Result<Self::Ok, Self::Error>
    where
        T: Serialize,
        T: ?Sized,
    {
        value.serialize(self)?;
        Ok(())
    }

    fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
    ) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_newtype_struct<T>(
        self,
        _name: &'static str,
        _value: &T,
    ) -> Result<Self::Ok, Self::Error>
    where
        T: Serialize,
        T: ?Sized,
    {
        Ok(())
    }

    fn serialize_newtype_variant<T>(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _value: &T,
    ) -> Result<Self::Ok, Self::Error>
    where
        T: Serialize,
        T: ?Sized,
    {
        Ok(())
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        Ok(self)
    }

    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple, Self::Error> {
        Ok(self)
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        Ok(self)
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        Ok(self)
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        Ok(self)
    }

    fn serialize_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        Ok(self)
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        Ok(self)
    }
}

fn extract_tensordata<T: ?Sized + Serialize>(value: &T) -> Option<SafetensorsTensorData> {
    use serde::ser::Impossible;

    #[derive(Debug, Default)]
    struct TensorDataSerializer {
        bytes: Option<Vec<u8>>,
        shape: Option<Vec<u64>>,
        dtype: Option<DType>,
    }

    impl TensorDataSerializer {
        /// Generates a SafetensorsTensorData if all the fields are populated
        fn get_tensor_data(self) -> Option<SafetensorsTensorData> {
            match (self.bytes, self.shape, self.dtype) {
                (Some(bytes), Some(shape), Some(dtype)) => Some(SafetensorsTensorData {
                    bytes,
                    shape: shape
                        .into_iter()
                        .map(|v| v.try_into().expect("failed to cast u64 to usize"))
                        .collect(),
                    dtype,
                }),
                _ => None,
            }
        }
    }

    impl SerializeSeq for TensorDataSerializer {
        type Ok = TensorDataSerializer;
        type Error = Error;

        fn serialize_element<T>(&mut self, value: &T) -> Result<(), Self::Error>
        where
            T: ?Sized + Serialize,
        {
            let serializer = TensorDataSerializer::default();
            let out = value.serialize(serializer)?;
            if let Some(ref mut shape) = self.shape {
                if let Some(shape_elem) = out.shape {
                    shape.extend_from_slice(&shape_elem);
                }
            } else {
                self.shape = out.shape
            }
            Ok(())
        }

        fn end(self) -> Result<Self::Ok, Self::Error> {
            Ok(self)
        }
    }

    impl SerializeStruct for TensorDataSerializer {
        type Ok = TensorDataSerializer;
        type Error = Error;

        fn serialize_field<T>(&mut self, key: &'static str, value: &T) -> Result<(), Self::Error>
        where
            T: ?Sized + Serialize,
        {
            let serializer = TensorDataSerializer::default();
            let out = value.serialize(serializer)?;
            match key {
                "bytes" => self.bytes = out.bytes,
                "shape" => self.shape = out.shape,
                "dtype" => self.dtype = out.dtype,
                _ => (),
            }
            Ok(())
        }

        fn end(self) -> Result<Self::Ok, Self::Error> {
            Ok(self)
        }
    }

    impl Serializer for TensorDataSerializer {
        type Ok = TensorDataSerializer;
        type Error = Error;

        type SerializeSeq = Self;
        type SerializeTuple = Impossible<Self::Ok, Error>;
        type SerializeTupleStruct = Impossible<Self::Ok, Error>;
        type SerializeTupleVariant = Impossible<Self::Ok, Error>;
        type SerializeMap = Impossible<Self::Ok, Error>;
        type SerializeStruct = Self;
        type SerializeStructVariant = Impossible<Self::Ok, Error>;

        fn serialize_f64(self, _v: f64) -> Result<Self::Ok, Self::Error> {
            Err(Error::NotFromTensorData)
        }

        fn serialize_f32(self, _v: f32) -> Result<Self::Ok, Self::Error> {
            Err(Error::NotFromTensorData)
        }

        fn serialize_i64(self, _v: i64) -> Result<Self::Ok, Self::Error> {
            Err(Error::NotFromTensorData)
        }

        fn serialize_i32(self, _v: i32) -> Result<Self::Ok, Self::Error> {
            Err(Error::NotFromTensorData)
        }

        fn serialize_u64(mut self, v: u64) -> Result<Self::Ok, Self::Error> {
            // for shape elements
            if let Some(ref mut shape) = self.shape {
                shape.push(v)
            } else {
                self.shape = Some(vec![v])
            }
            Ok(self)
        }

        fn serialize_u32(self, v: u32) -> Result<Self::Ok, Self::Error> {
            // for shape elements
            self.serialize_u64(v as u64)
        }

        fn serialize_i8(self, _v: i8) -> Result<Self::Ok, Self::Error> {
            Err(Error::NotFromTensorData)
        }

        fn serialize_i16(self, _v: i16) -> Result<Self::Ok, Self::Error> {
            Err(Error::NotFromTensorData)
        }

        fn serialize_u8(self, _v: u8) -> Result<Self::Ok, Self::Error> {
            Err(Error::NotFromTensorData)
        }

        fn serialize_u16(self, _v: u16) -> Result<Self::Ok, Self::Error> {
            Err(Error::NotFromTensorData)
        }

        fn serialize_bool(self, _v: bool) -> Result<Self::Ok, Self::Error> {
            Err(Error::NotFromTensorData)
        }

        fn serialize_char(self, _v: char) -> Result<Self::Ok, Self::Error> {
            Err(Error::NotFromTensorData)
        }

        fn serialize_str(self, _v: &str) -> Result<Self::Ok, Self::Error> {
            Err(Error::NotFromTensorData)
        }

        fn serialize_bytes(mut self, v: &[u8]) -> Result<Self::Ok, Self::Error> {
            // Tensor bytes
            self.bytes = Some(v.to_vec());
            Ok(self)
        }

        fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
            Err(Error::NotFromTensorData)
        }

        fn serialize_some<V>(self, _value: &V) -> Result<Self::Ok, Self::Error>
        where
            V: Serialize,
            V: ?Sized,
        {
            Err(Error::NotFromTensorData)
        }

        fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
            Err(Error::NotFromTensorData)
        }

        fn serialize_unit_struct(self, _name: &'static str) -> Result<Self::Ok, Self::Error> {
            Err(Error::NotFromTensorData)
        }

        fn serialize_unit_variant(
            mut self,
            name: &'static str,
            _variant_index: u32,
            variant: &'static str,
        ) -> Result<Self::Ok, Self::Error> {
            if name == "DType" {
                self.dtype = Some(match variant {
                    "F64" => DType::F64,
                    "F32" => DType::F32,
                    "F16" => DType::F16,
                    "BF16" => DType::BF16,
                    "I64" => DType::I64,
                    "I32" => DType::I32,
                    "I16" => DType::I16,
                    "I8" => DType::I8,
                    "U64" => DType::U64,
                    "U32" => DType::U32,
                    "U16" => DType::U16,
                    "U8" => DType::U8,
                    "Bool" => DType::Bool,
                    _ => return Err(Error::UnsupportedType),
                });
            }
            Ok(self)
        }

        fn serialize_newtype_struct<V>(
            self,
            _name: &'static str,
            _value: &V,
        ) -> Result<Self::Ok, Self::Error>
        where
            V: Serialize,
            V: ?Sized,
        {
            Err(Error::NotFromTensorData)
        }

        fn serialize_newtype_variant<V>(
            self,
            _name: &'static str,
            _variant_index: u32,
            _variant: &'static str,
            _value: &V,
        ) -> Result<Self::Ok, Self::Error>
        where
            V: Serialize,
            V: ?Sized,
        {
            Err(Error::NotFromTensorData)
        }

        fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
            // for shapes
            Ok(self)
        }

        fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple, Self::Error> {
            Err(Error::NotFromTensorData)
        }

        fn serialize_tuple_struct(
            self,
            _name: &'static str,
            _len: usize,
        ) -> Result<Self::SerializeTupleStruct, Self::Error> {
            Err(Error::NotFromTensorData)
        }

        fn serialize_tuple_variant(
            self,
            _name: &'static str,
            _variant_index: u32,
            _variant: &'static str,
            _len: usize,
        ) -> Result<Self::SerializeTupleVariant, Self::Error> {
            Err(Error::NotFromTensorData)
        }

        fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
            Err(Error::NotFromTensorData)
        }

        fn serialize_struct(
            self,
            _name: &'static str,
            _len: usize,
        ) -> Result<Self::SerializeStruct, Self::Error> {
            Ok(self)
        }

        fn serialize_struct_variant(
            self,
            _name: &'static str,
            _variant_index: u32,
            _variant: &'static str,
            _len: usize,
        ) -> Result<Self::SerializeStructVariant, Self::Error> {
            Err(Error::NotFromTensorData)
        }
    }

    let extractor = TensorDataSerializer::default();
    let tensor_data_populated = value.serialize(extractor);
    match tensor_data_populated {
        Ok(tensor_data_populated) => tensor_data_populated.get_tensor_data(),
        Err(_) => None,
    }
}

// Implement SerializeStruct for field serialization
impl serde::ser::SerializeStruct for &mut FlattenSerializer {
    type Ok = ();
    type Error = Error;

    fn serialize_field<T>(&mut self, key: &'static str, value: &T) -> Result<(), Self::Error>
    where
        T: Serialize,
        T: ?Sized,
    {
        match extract_tensordata(value) {
            Some(tensor_data) => {
                // The field is a TensorData, store it in the current path.
                // TensorData are stored under param in `ParamSerde`, however we dont want this
                // last key to be stored, so we just store the TensorData under the current path
                self.map.insert(self.current_path(), tensor_data);
            }
            None => {
                // This is a nested struct - add the field name to path and serialize recursively
                self.path.push(key.to_string());
                value.serialize(&mut **self)?;
                self.path.pop();
            }
        }
        Ok(())
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        // Don't pop anything - we manage the path in serialize_field
        Ok(())
    }
}

// Implement remaining serializer traits (stubs)
impl serde::ser::SerializeSeq for &mut FlattenSerializer {
    type Ok = ();
    type Error = Error;
    fn serialize_element<T>(&mut self, _value: &T) -> Result<(), Self::Error>
    where
        T: Serialize,
        T: ?Sized,
    {
        Ok(())
    }
    fn end(self) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }
}

impl serde::ser::SerializeTuple for &mut FlattenSerializer {
    type Ok = ();
    type Error = Error;
    fn serialize_element<T>(&mut self, _value: &T) -> Result<(), Self::Error>
    where
        T: Serialize,
        T: ?Sized,
    {
        Ok(())
    }
    fn end(self) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }
}

impl serde::ser::SerializeTupleStruct for &mut FlattenSerializer {
    type Ok = ();
    type Error = Error;
    fn serialize_field<T>(&mut self, _value: &T) -> Result<(), Self::Error>
    where
        T: Serialize,
        T: ?Sized,
    {
        Ok(())
    }
    fn end(self) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }
}

impl serde::ser::SerializeTupleVariant for &mut FlattenSerializer {
    type Ok = ();
    type Error = Error;
    fn serialize_field<T>(&mut self, _value: &T) -> Result<(), Self::Error>
    where
        T: Serialize,
        T: ?Sized,
    {
        Ok(())
    }
    fn end(self) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }
}

impl serde::ser::SerializeMap for &mut FlattenSerializer {
    type Ok = ();
    type Error = Error;
    fn serialize_key<T>(&mut self, _key: &T) -> Result<(), Self::Error>
    where
        T: Serialize,
        T: ?Sized,
    {
        Ok(())
    }
    fn serialize_value<T>(&mut self, _value: &T) -> Result<(), Self::Error>
    where
        T: Serialize,
        T: ?Sized,
    {
        Ok(())
    }
    fn end(self) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }
}

impl serde::ser::SerializeStructVariant for &mut FlattenSerializer {
    type Ok = ();
    type Error = Error;
    fn serialize_field<T>(&mut self, _key: &'static str, _value: &T) -> Result<(), Self::Error>
    where
        T: Serialize,
        T: ?Sized,
    {
        Ok(())
    }
    fn end(self) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }
}
