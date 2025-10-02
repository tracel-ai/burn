//! Just enough pickle support to be able to read PyTorch checkpoints.
//!
//! This implementation is based on the candle project's pickle loader with significant
//! modifications for improved separation of concerns and extended PyTorch compatibility.
//!
//! Original source: https://github.com/huggingface/candle/blob/main/candle-core/src/pickle.rs
//!
//! Modifications include:
//! - Lazy tensor data loading for memory efficiency
//! - Extended PyTorch version compatibility (0.1.10 - 2.x)
//! - Better separation of pickle parsing and tensor extraction
//! - Support for both legacy and modern PyTorch formats
use crate::TensorSnapshot;
use crate::pytorch::lazy_data::LazyDataSource;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use burn_core::module::ParamId;
use burn_tensor::{DType, TensorData};
use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::io::{self, BufRead};
use std::sync::Arc;

/// Error type for pickle operations
#[derive(Debug)]
pub enum PickleError {
    Io(io::Error),
    InvalidOpCode(u8),
    InvalidProtocol(u8),
    UnexpectedOpCode(OpCode),
    UnsupportedType(String),
    InvalidData(String),
    StackUnderflow,
    MemoNotFound(u32),
    InvalidShapeOrType,
}

impl From<io::Error> for PickleError {
    fn from(e: io::Error) -> Self {
        PickleError::Io(e)
    }
}

impl std::fmt::Display for PickleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PickleError::Io(e) => write!(f, "IO error: {}", e),
            PickleError::InvalidOpCode(code) => write!(
                f,
                "Invalid pickle opcode: 0x{:02x}. The file may be corrupted or use an unsupported pickle protocol.",
                code
            ),
            PickleError::InvalidProtocol(proto) => write!(
                f,
                "Invalid or unsupported pickle protocol version: {}. Supported versions are 2-5.",
                proto
            ),
            PickleError::UnexpectedOpCode(op) => {
                write!(f, "Unexpected pickle opcode {:?} in current context", op)
            }
            PickleError::UnsupportedType(ty) => write!(
                f,
                "Unsupported Python type '{}'. This may indicate a full model save rather than a state_dict.",
                ty
            ),
            PickleError::InvalidData(msg) => write!(f, "Invalid data in pickle file: {}", msg),
            PickleError::StackUnderflow => {
                write!(f, "Pickle stack underflow - the file may be corrupted")
            }
            PickleError::MemoNotFound(idx) => write!(
                f,
                "Pickle memo reference {} not found - the file may be corrupted",
                idx
            ),
            PickleError::InvalidShapeOrType => {
                write!(f, "Invalid tensor shape or data type in PyTorch file")
            }
        }
    }
}

impl std::error::Error for PickleError {}

type Result<T> = std::result::Result<T, PickleError>;

// https://docs.juliahub.com/Pickle/LAUNc/0.1.0/opcode/
#[repr(u8)]
#[derive(Debug, Eq, PartialEq, Clone)]
pub enum OpCode {
    // https://github.com/python/cpython/blob/ed25f097160b5cbb0c9a1f9a746d2f1bbc96515a/Lib/pickletools.py#L2123
    Proto = 0x80,
    Global = b'c',
    BinPut = b'q',
    LongBinPut = b'r',
    EmptyTuple = b')',
    Reduce = b'R',
    Mark = b'(',
    BinUnicode = b'X',
    ShortBinString = b'U',
    BinInt = b'J',
    Int = b'I',
    Tuple = b't',
    BinPersId = b'Q',
    BinInt1 = b'K',
    BinInt2 = b'M',
    Tuple1 = 0x85,
    Tuple2 = 0x86,
    Tuple3 = 0x87,
    NewTrue = 0x88,
    NewFalse = 0x89,
    None = b'N',
    BinGet = b'h',
    LongBinGet = b'j',
    SetItem = b's',
    SetItems = b'u',
    EmptyDict = b'}',
    Dict = b'd',
    Build = b'b',
    Stop = b'.',
    NewObj = 0x81,
    EmptyList = b']',
    List = b'l',
    BinFloat = b'G',
    Append = b'a',
    Appends = b'e',
    Long1 = 0x8a,
    Memoize = 0x94,
}

// Avoid using FromPrimitive so as not to drag another dependency.
impl TryFrom<u8> for OpCode {
    type Error = u8;
    fn try_from(value: u8) -> std::result::Result<Self, Self::Error> {
        match value {
            0x80 => Ok(Self::Proto),
            b'c' => Ok(Self::Global),
            b'q' => Ok(Self::BinPut),
            b'r' => Ok(Self::LongBinPut),
            b')' => Ok(Self::EmptyTuple),
            b'R' => Ok(Self::Reduce),
            b'(' => Ok(Self::Mark),
            b'X' => Ok(Self::BinUnicode),
            b'U' => Ok(Self::ShortBinString),
            b'J' => Ok(Self::BinInt),
            b'I' => Ok(Self::Int),
            b't' => Ok(Self::Tuple),
            b'Q' => Ok(Self::BinPersId),
            b'K' => Ok(Self::BinInt1),
            b'M' => Ok(Self::BinInt2),
            b'N' => Ok(Self::None),
            0x85 => Ok(Self::Tuple1),
            0x86 => Ok(Self::Tuple2),
            0x87 => Ok(Self::Tuple3),
            0x88 => Ok(Self::NewTrue),
            0x89 => Ok(Self::NewFalse),
            b'h' => Ok(Self::BinGet),
            b'j' => Ok(Self::LongBinGet),
            b's' => Ok(Self::SetItem),
            b'u' => Ok(Self::SetItems),
            b'}' => Ok(Self::EmptyDict),
            b'd' => Ok(Self::Dict),
            b'b' => Ok(Self::Build),
            b'.' => Ok(Self::Stop),
            0x81 => Ok(Self::NewObj),
            b']' => Ok(Self::EmptyList),
            b'l' => Ok(Self::List),
            b'G' => Ok(Self::BinFloat),
            b'a' => Ok(Self::Append),
            b'e' => Ok(Self::Appends),
            0x8a => Ok(Self::Long1),
            0x94 => Ok(Self::Memoize),
            value => Err(value),
        }
    }
}

fn read_to_newline<R: BufRead>(r: &mut R) -> Result<Vec<u8>> {
    let mut data: Vec<u8> = Vec::with_capacity(32);
    r.read_until(b'\n', &mut data)?;
    data.pop();
    if data.last() == Some(&b'\r') {
        data.pop();
    }
    Ok(data)
}

fn buf_to_str(buf: &[u8]) -> Result<String> {
    String::from_utf8(buf.to_vec())
        .map_err(|e| PickleError::InvalidData(format!("Invalid UTF-8: {}", e)))
}

#[derive(Debug, Clone)]
pub enum Object {
    Class {
        module_name: String,
        name: String,
    },
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    None,
    Tuple(Vec<Object>),
    List(Vec<Object>),
    Dict(HashMap<String, Object>),
    Persistent(Vec<u8>),
    PersistentTuple(Vec<Object>),
    Reduce {
        callable: Box<Object>,
        args: Box<Object>,
    },
    Build {
        callable: Box<Object>,
        args: Box<Object>,
    },
    TorchParam(TensorSnapshot),
}

fn rebuild_from_type_v2(
    o: Object,
    memo: &mut HashMap<u32, Object>,
    data_source: &Option<Arc<LazyDataSource>>,
) -> Result<Object> {
    let args = if let Object::Tuple(args) = o {
        if args.is_empty() {
            return Err(PickleError::InvalidData(
                "rebuild_from_type_v2: empty args".to_string(),
            ));
        }
        args
    } else {
        return Err(PickleError::InvalidData(format!(
            "rebuild_from_type_v2: expected tuple got {:?}",
            o
        )));
    };
    let func = &args[0];
    match func {
        Object::Class { module_name, name } => {
            let module_name = module_name.as_str();
            let name = name.as_str();
            // For rebuild_tensor_v2, the args might already be in a tuple
            let actual_args = if args.len() == 2 && matches!(&args[1], Object::Tuple(_)) {
                // If there's only one arg and it's a tuple, use it directly
                args[1].clone()
            } else {
                // Otherwise, wrap the remaining args in a tuple
                Object::Tuple(args[1..].to_vec())
            };
            if module_name == "torch._utils" && name == "_rebuild_tensor_v2" {
                rebuild_tensor_v2(actual_args, memo, data_source)
            } else if module_name == "torch._tensor" && name == "_rebuild_from_type_v2" {
                rebuild_from_type_v2(actual_args, memo, data_source)
            } else if module_name == "torch._utils" && name == "_rebuild_parameter" {
                rebuild_parameter(actual_args, memo, data_source)
            } else if module_name == "collections" && name == "OrderedDict" {
                // OrderedDict is treated as a regular Dict in our implementation
                Ok(Object::Dict(HashMap::new()))
            } else {
                Err(PickleError::UnsupportedType(format!(
                    "{}.{}",
                    module_name, name
                )))
            }
        }
        _ => Err(PickleError::InvalidData(format!(
            "rebuild_from_type_v2: expected class got {:?}",
            func
        ))),
    }
}

fn rebuild_parameter(
    args: Object,
    memo: &mut HashMap<u32, Object>,
    data_source: &Option<Arc<LazyDataSource>>,
) -> Result<Object> {
    let args = if let Object::Tuple(args) = args {
        if args.is_empty() {
            return Err(PickleError::InvalidData(
                "rebuild_parameter: empty args".to_string(),
            ));
        }
        args
    } else {
        return Err(PickleError::InvalidData(format!(
            "rebuild_parameter: expected tuple got {:?}",
            args
        )));
    };
    let data = &args[0];
    let tensor = match data {
        Object::Reduce {
            callable: _,
            args: _,
        } => rebuild_from_type_v2(data.clone(), memo, data_source)?,
        _ => data.clone(),
    };
    Ok(tensor)
}

fn rebuild_tensor_v2(
    args: Object,
    _memo: &mut HashMap<u32, Object>,
    data_source: &Option<Arc<LazyDataSource>>,
) -> Result<Object> {
    // args is (storage, storage_offset, shape, stride, requires_grad, backward_hooks)
    let args = if let Object::Tuple(args) = args {
        args
    } else {
        return Err(PickleError::InvalidData(format!(
            "rebuild_tensor_v2: expected tuple got {:?}",
            args
        )));
    };

    if args.len() < 5 {
        return Err(PickleError::InvalidData(format!(
            "rebuild_tensor_v2: expected at least 5 args, got {}",
            args.len()
        )));
    }

    // First argument is the storage (persistent ID)
    let (storage_info, storage_tuple) = match &args[0] {
        Object::Persistent(data) => (data.clone(), None),
        Object::PersistentTuple(tuple) => (vec![], Some(tuple.clone())),
        _ => {
            return Err(PickleError::InvalidData(format!(
                "rebuild_tensor_v2: expected persistent id got {:?}",
                args[0]
            )));
        }
    };

    // Second argument is storage offset
    let storage_offset = match &args[1] {
        Object::Int(offset) => *offset as usize,
        _ => 0,
    };

    // Third argument is shape
    let shape = match &args[2] {
        Object::Tuple(shape) => shape
            .iter()
            .map(|x| match x {
                Object::Int(i) => Ok(*i as usize),
                _ => Err(PickleError::InvalidData(
                    "shape must contain ints".to_string(),
                )),
            })
            .collect::<Result<Vec<_>>>()?,
        _ => {
            return Err(PickleError::InvalidData(format!(
                "rebuild_tensor_v2: expected shape tuple got {:?}",
                args[2]
            )));
        }
    };

    // Fourth argument is stride (we don't use it but validate it exists)
    let _stride = matches!(&args[3], Object::Tuple(_));

    // Parse the storage info to extract dtype and storage key
    // The persistent ID is typically a tuple like: ('storage', 'FloatStorage', '0', 'cpu', 4)
    let (dtype, storage_key) = if let Some(tuple) = storage_tuple {
        // Direct tuple access
        if tuple.len() >= 3 {
            let storage_type = match &tuple[1] {
                Object::String(s) => s.as_str(),
                Object::Class {
                    module_name: _,
                    name,
                } => name.as_str(),
                _ => "FloatStorage",
            };
            let dtype = match storage_type {
                "FloatStorage" => DType::F32,
                "DoubleStorage" => DType::F64,
                "HalfStorage" => DType::F16,
                "BFloat16Storage" => DType::BF16,
                "LongStorage" => DType::I64,
                "IntStorage" => DType::I32,
                "ShortStorage" => DType::I16,
                "CharStorage" | "ByteStorage" => DType::I8,
                "BoolStorage" => DType::Bool,
                _ => DType::F32, // Default to F32
            };
            let key = match &tuple[2] {
                Object::String(s) => s.clone(),
                _ => "0".to_string(),
            };
            (dtype, key)
        } else {
            (DType::F32, "0".to_string())
        }
    } else if !storage_info.is_empty() {
        // Legacy string-based parsing
        let storage_str = String::from_utf8_lossy(&storage_info);
        if storage_str.starts_with("Tuple(") {
            // Parse from the debug representation we stored
            let parts: Vec<&str> = storage_str
                .trim_start_matches("Tuple(")
                .trim_end_matches(")")
                .split(", ")
                .map(|s| {
                    let trimmed = s.trim_matches('"');
                    if let Some(inner) = trimmed
                        .strip_prefix("Object::String(\"")
                        .and_then(|s| s.strip_suffix("\")"))
                    {
                        inner
                    } else {
                        trimmed
                    }
                })
                .collect();

            if parts.len() >= 3 {
                let dtype = match parts[1] {
                    "FloatStorage" => DType::F32,
                    "DoubleStorage" => DType::F64,
                    "HalfStorage" => DType::F16,
                    "BFloat16Storage" => DType::BF16,
                    "LongStorage" => DType::I64,
                    "IntStorage" => DType::I32,
                    "ShortStorage" => DType::I16,
                    "CharStorage" | "ByteStorage" => DType::I8,
                    _ => DType::F32, // Default to F32
                };
                (dtype, parts[2].to_string())
            } else {
                (DType::F32, "0".to_string())
            }
        } else {
            (DType::F32, "0".to_string())
        }
    } else {
        (DType::F32, "0".to_string())
    };

    // If no data source, we can't load tensor data
    let data_source = match data_source {
        Some(ds) => ds.clone(),
        None => {
            return Err(PickleError::InvalidData(
                "Cannot load tensor data without a data source".to_string(),
            ));
        }
    };

    // Create clones for the closure
    let data_source_clone = data_source.clone();
    let shape_clone = shape.clone();

    // Find the correct data file key
    let data_file_key = {
        let exact_key = format!("data/{}", storage_key);
        if data_source.contains(&exact_key) {
            exact_key
        } else {
            // Try other patterns
            data_source
                .keys()
                .into_iter()
                .find(|key| {
                    key.ends_with(&format!("/data/{}", storage_key))
                        || (key.contains("/data/") && key.rsplit('/').next() == Some(&storage_key))
                })
                .unwrap_or_else(|| format!("data/{}", storage_key))
        }
    };

    // Track storage usage IMMEDIATELY for lazy boundary detection
    // This must happen BEFORE creating the closure, not inside it!
    if let LazyDataSource::LegacyMultiStorage(ref source) = *data_source {
        let source = source
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let num_elements: usize = shape.iter().product();
        let bytes_needed = storage_offset * dtype.size() + num_elements * dtype.size();
        source.track_storage_usage(&storage_key, 0, bytes_needed);
    }

    // Create a TensorSnapshot with a closure that loads the actual data on-demand
    Ok(Object::TorchParam(TensorSnapshot::from_closure(
        Rc::new(move || {
            // Load data only when needed
            if let Ok(data) = data_source_clone.read(&data_file_key) {
                // Parse the binary data based on dtype
                let num_elements = shape_clone.iter().product::<usize>().max(1);

                // Calculate expected size and actual element count
                let element_size = match dtype {
                    DType::F64 => 8,
                    DType::F32 => 4,
                    DType::F16 | DType::BF16 => 2,
                    DType::I64 => 8,
                    DType::I32 => 4,
                    DType::I16 => 2,
                    DType::I8 | DType::U8 | DType::Bool => 1,
                    _ => 4,
                };

                // Apply storage offset
                let offset_bytes = storage_offset * element_size;
                if offset_bytes >= data.len() {
                    return Ok(TensorData::new(
                        vec![0.0f32; num_elements],
                        shape_clone.clone(),
                    ));
                }

                let data_slice = &data[offset_bytes..];
                let available_elements = data_slice.len() / element_size;
                let elements_to_read = num_elements.min(available_elements);

                // Convert bytes to the appropriate type
                match dtype {
                    DType::F32 => {
                        let mut values = Vec::with_capacity(num_elements);
                        for i in 0..elements_to_read {
                            let bytes = [
                                data_slice[i * 4],
                                data_slice[i * 4 + 1],
                                data_slice[i * 4 + 2],
                                data_slice[i * 4 + 3],
                            ];
                            values.push(f32::from_le_bytes(bytes));
                        }
                        // Pad with zeros if needed
                        values.resize(num_elements, 0.0);
                        Ok(TensorData::new(values, shape_clone.clone()))
                    }
                    DType::F64 => {
                        let mut values = Vec::with_capacity(num_elements);
                        for i in 0..elements_to_read {
                            let mut bytes = [0u8; 8];
                            bytes.copy_from_slice(&data_slice[i * 8..(i + 1) * 8]);
                            values.push(f64::from_le_bytes(bytes));
                        }
                        values.resize(num_elements, 0.0);
                        Ok(TensorData::new(values, shape_clone.clone()))
                    }
                    DType::I64 => {
                        let mut values = Vec::with_capacity(num_elements);
                        for i in 0..elements_to_read {
                            let mut bytes = [0u8; 8];
                            bytes.copy_from_slice(&data_slice[i * 8..(i + 1) * 8]);
                            values.push(i64::from_le_bytes(bytes));
                        }
                        values.resize(num_elements, 0);
                        Ok(TensorData::new(values, shape_clone.clone()))
                    }
                    DType::I32 => {
                        let mut values = Vec::with_capacity(num_elements);
                        for i in 0..elements_to_read {
                            let mut bytes = [0u8; 4];
                            bytes.copy_from_slice(&data_slice[i * 4..(i + 1) * 4]);
                            values.push(i32::from_le_bytes(bytes));
                        }
                        values.resize(num_elements, 0);
                        Ok(TensorData::new(values, shape_clone.clone()))
                    }
                    DType::I16 => {
                        let mut values = Vec::with_capacity(num_elements);
                        for i in 0..elements_to_read {
                            let mut bytes = [0u8; 2];
                            bytes.copy_from_slice(&data_slice[i * 2..(i + 1) * 2]);
                            values.push(i16::from_le_bytes(bytes));
                        }
                        values.resize(num_elements, 0);
                        Ok(TensorData::new(values, shape_clone.clone()))
                    }
                    DType::I8 => {
                        let mut values = Vec::with_capacity(num_elements);
                        for &byte in data_slice.iter().take(elements_to_read) {
                            values.push(byte as i8);
                        }
                        values.resize(num_elements, 0);
                        Ok(TensorData::new(values, shape_clone.clone()))
                    }
                    DType::Bool => {
                        let mut values = Vec::with_capacity(num_elements);
                        for &byte in data_slice.iter().take(elements_to_read) {
                            values.push(byte != 0);
                        }
                        values.resize(num_elements, false);
                        Ok(TensorData::new(values, shape_clone.clone()))
                    }
                    _ => {
                        // For other types, default to f32 zeros for now
                        Ok(TensorData::new(
                            vec![0.0f32; num_elements],
                            shape_clone.clone(),
                        ))
                    }
                }
            } else {
                // If no data file found, return zeros of the appropriate type
                let num_elements = shape_clone.iter().product::<usize>().max(1);
                match dtype {
                    DType::I64 => Ok(TensorData::new(
                        vec![0i64; num_elements],
                        shape_clone.clone(),
                    )),
                    DType::I32 => Ok(TensorData::new(
                        vec![0i32; num_elements],
                        shape_clone.clone(),
                    )),
                    DType::I16 => Ok(TensorData::new(
                        vec![0i16; num_elements],
                        shape_clone.clone(),
                    )),
                    DType::I8 => Ok(TensorData::new(
                        vec![0i8; num_elements],
                        shape_clone.clone(),
                    )),
                    DType::F64 => Ok(TensorData::new(
                        vec![0.0f64; num_elements],
                        shape_clone.clone(),
                    )),
                    DType::Bool => Ok(TensorData::new(
                        vec![false; num_elements],
                        shape_clone.clone(),
                    )),
                    _ => Ok(TensorData::new(
                        vec![0.0f32; num_elements],
                        shape_clone.clone(),
                    )),
                }
            }
        }),
        dtype,
        shape,
        vec![],         // path_stack
        vec![],         // container_stack
        ParamId::new(), // tensor_id
    )))
}

pub struct Stack {
    stack: Vec<Object>,
    memo: HashMap<u32, Object>,
    data_source: Option<Arc<LazyDataSource>>,
}

impl Default for Stack {
    fn default() -> Self {
        Self::new()
    }
}

impl Stack {
    pub fn new() -> Self {
        // For cases where no data source is needed (pure pickle without tensor data)
        Self {
            stack: Vec::new(),
            memo: HashMap::new(),
            data_source: None,
        }
    }

    pub fn with_data_source(data_source: Arc<LazyDataSource>) -> Self {
        Self {
            stack: Vec::new(),
            memo: HashMap::new(),
            data_source: Some(data_source),
        }
    }

    fn push(&mut self, o: Object) {
        self.stack.push(o)
    }

    fn pop(&mut self) -> Result<Object> {
        match self.stack.pop() {
            None => Err(PickleError::StackUnderflow),
            Some(o) => Ok(o),
        }
    }

    fn top(&self) -> Result<Object> {
        match self.stack.last() {
            None => Err(PickleError::StackUnderflow),
            Some(o) => Ok(o.clone()),
        }
    }

    fn pop_to_marker(&mut self) -> Result<Vec<Object>> {
        let marker_pos = self
            .stack
            .iter()
            .rposition(|o| {
                matches!(o, Object::Class { module_name, name }
                if module_name == "mark" && name == "mark")
            })
            .ok_or(PickleError::InvalidData("marker not found".to_string()))?;

        let result = self.stack.split_off(marker_pos + 1);
        self.stack.pop(); // Remove the marker
        Ok(result)
    }

    fn last_mut(&mut self) -> Result<&mut Object> {
        match self.stack.last_mut() {
            None => Err(PickleError::StackUnderflow),
            Some(o) => Ok(o),
        }
    }

    fn push_mark(&mut self) {
        self.stack.push(Object::Class {
            module_name: "mark".to_string(),
            name: "mark".to_string(),
        });
    }

    fn memo_get(&self, idx: u32) -> Result<Object> {
        self.memo
            .get(&idx)
            .cloned()
            .ok_or(PickleError::MemoNotFound(idx))
    }

    fn memo_put(&mut self, idx: u32, obj: Object) {
        self.memo.insert(idx, obj);
    }

    fn memo_len(&self) -> usize {
        self.memo.len()
    }
}

fn read_global<R: BufRead>(r: &mut R, stack: &mut Stack) -> Result<()> {
    let module_name = buf_to_str(&read_to_newline(r)?)?;
    let name = buf_to_str(&read_to_newline(r)?)?;
    stack.push(Object::Class { module_name, name });
    Ok(())
}

fn read_long1<R: BufRead>(r: &mut R, stack: &mut Stack) -> Result<()> {
    let len = r.read_u8()? as usize;
    let mut data = vec![0u8; len];
    r.read_exact(&mut data)?;
    // Handle little-endian signed integer
    let mut value = 0i64;
    for (i, &byte) in data.iter().enumerate().take(8) {
        // Only process up to 8 bytes for i64, and use wrapping to avoid overflow
        value |= (byte as i64).wrapping_shl((i as u32) * 8);
    }
    // Handle sign extension for negative numbers
    if len < 8 && data.last().is_some_and(|&b| b & 0x80 != 0) {
        // Sign extend
        for i in len..8 {
            value |= 0xffi64.wrapping_shl((i as u32) * 8);
        }
    }
    stack.push(Object::Int(value));
    Ok(())
}

fn read_string<R: BufRead>(r: &mut R, stack: &mut Stack, len: usize) -> Result<()> {
    let mut data = vec![0u8; len];
    r.read_exact(&mut data)?;
    let s = buf_to_str(&data)?;
    stack.push(Object::String(s));
    Ok(())
}

fn read_bin_int<R: BufRead>(r: &mut R, stack: &mut Stack) -> Result<()> {
    let v = r.read_i32::<LittleEndian>()?;
    stack.push(Object::Int(v as i64));
    Ok(())
}

fn read_int<R: BufRead>(r: &mut R, stack: &mut Stack) -> Result<()> {
    // INT opcode reads an integer as ASCII string followed by newline
    let line = read_to_newline(r)?;
    let s = buf_to_str(&line)?;
    let v = s
        .parse::<i64>()
        .map_err(|e| PickleError::InvalidData(format!("Invalid INT value '{}': {}", s, e)))?;
    stack.push(Object::Int(v));
    Ok(())
}

fn read_bin_int1<R: BufRead>(r: &mut R, stack: &mut Stack) -> Result<()> {
    let v = r.read_u8()?;
    stack.push(Object::Int(v as i64));
    Ok(())
}

fn read_bin_int2<R: BufRead>(r: &mut R, stack: &mut Stack) -> Result<()> {
    let v = r.read_u16::<LittleEndian>()?;
    stack.push(Object::Int(v as i64));
    Ok(())
}

fn read_bin_float<R: BufRead>(r: &mut R, stack: &mut Stack) -> Result<()> {
    // Python's BINFLOAT uses big-endian encoding
    let v = r.read_f64::<byteorder::BigEndian>()?;
    stack.push(Object::Float(v));
    Ok(())
}

pub fn read_pickle<R: BufRead>(r: &mut R) -> Result<Object> {
    // For pure pickle without tensor data, no data source is needed
    read_pickle_with_optional_data(r, None)
}

/// Skip over a pickle without parsing it fully
/// This is useful for legacy format where we need to skip the main object
/// that contains tensors but we don't have a data source yet
pub fn skip_pickle<R: BufRead>(r: &mut R) -> Result<()> {
    // Read the protocol marker if present
    let mut first_byte = [0u8; 1];
    r.read_exact(&mut first_byte)?;

    if first_byte[0] == 0x80 {
        // PROTO marker - read protocol version
        let mut proto_version = [0u8; 1];
        r.read_exact(&mut proto_version)?;
    } else {
        // Not a PROTO marker, we need to handle this byte
        // Put it back by using a small state machine
        // For now, we'll track that we've seen a non-proto byte
    }

    // Scan until we find STOP (0x2e) opcode
    loop {
        let mut byte = [0u8; 1];
        r.read_exact(&mut byte)?;

        match byte[0] {
            0x2e => {
                // STOP - end of pickle
                break;
            }
            0x58 | 0x42 | 0x43 | 0x54 | 0x55 | 0x56 | 0x8c | 0x8d | 0x8e => {
                // String/bytes opcodes with length prefixes
                let length = match byte[0] {
                    0x43 | 0x55 | 0x8c => {
                        // SHORT versions - 1 byte length
                        let mut len_byte = [0u8; 1];
                        r.read_exact(&mut len_byte)?;
                        len_byte[0] as usize
                    }
                    0x42 | 0x54 | 0x58 | 0x56 => {
                        // Regular versions - 4 byte length
                        let mut len_bytes = [0u8; 4];
                        r.read_exact(&mut len_bytes)?;
                        u32::from_le_bytes(len_bytes) as usize
                    }
                    0x8d | 0x8e => {
                        // 8-byte length versions
                        let mut len_bytes = [0u8; 8];
                        r.read_exact(&mut len_bytes)?;
                        u64::from_le_bytes(len_bytes) as usize
                    }
                    _ => 0,
                };

                // Skip the actual data
                let mut skip_buf = vec![0u8; length.min(8192)];
                let mut skipped = 0;
                while skipped < length {
                    let to_skip = (length - skipped).min(skip_buf.len());
                    r.read_exact(&mut skip_buf[..to_skip])?;
                    skipped += to_skip;
                }
            }
            0x4b | 0x4d | 0x4e => {
                // BININT1, BININT2, BININT4 - skip the integer bytes
                let skip_count = match byte[0] {
                    0x4b => 1,
                    0x4d => 2,
                    0x4e => 4,
                    _ => 0,
                };
                let mut skip_buf = vec![0u8; skip_count];
                r.read_exact(&mut skip_buf)?;
            }
            0x47 => {
                // BINFLOAT - skip 8 bytes
                let mut skip_buf = [0u8; 8];
                r.read_exact(&mut skip_buf)?;
            }
            0x4a => {
                // BININT - skip 4 bytes (signed)
                let mut skip_buf = [0u8; 4];
                r.read_exact(&mut skip_buf)?;
            }
            0x8a => {
                // LONG1 - 1 byte length, then that many bytes
                let mut len_byte = [0u8; 1];
                r.read_exact(&mut len_byte)?;
                let length = len_byte[0] as usize;
                let mut skip_buf = vec![0u8; length];
                r.read_exact(&mut skip_buf)?;
            }
            0x8b => {
                // LONG4 - 4 byte length, then that many bytes
                let mut len_bytes = [0u8; 4];
                r.read_exact(&mut len_bytes)?;
                let length = u32::from_le_bytes(len_bytes) as usize;
                let mut skip_buf = vec![0u8; length.min(8192)];
                let mut skipped = 0;
                while skipped < length {
                    let to_skip = (length - skipped).min(skip_buf.len());
                    r.read_exact(&mut skip_buf[..to_skip])?;
                    skipped += to_skip;
                }
            }
            _ => {
                // Other opcodes - most don't have additional data
                // or are stack manipulation opcodes we can ignore
            }
        }
    }

    Ok(())
}

pub fn read_pickle_with_data<R: BufRead>(
    r: &mut R,
    data_source: Arc<LazyDataSource>,
) -> Result<Object> {
    read_pickle_with_optional_data(r, Some(data_source))
}

pub fn read_pickle_with_optional_data<R: BufRead>(
    r: &mut R,
    data_source: Option<Arc<LazyDataSource>>,
) -> Result<Object> {
    let mut stack = match data_source {
        Some(ds) => Stack::with_data_source(ds),
        None => Stack::new(),
    };
    loop {
        let op_code = r.read_u8()?;
        let op_code = OpCode::try_from(op_code).map_err(PickleError::InvalidOpCode)?;
        match op_code {
            OpCode::Proto => {
                let version = r.read_u8()?;
                if version > 5 {
                    return Err(PickleError::InvalidProtocol(version));
                }
            }
            OpCode::Global => read_global(r, &mut stack)?,
            OpCode::BinInt => read_bin_int(r, &mut stack)?,
            OpCode::Int => read_int(r, &mut stack)?,
            OpCode::BinInt1 => read_bin_int1(r, &mut stack)?,
            OpCode::BinInt2 => read_bin_int2(r, &mut stack)?,
            OpCode::BinFloat => read_bin_float(r, &mut stack)?,
            OpCode::BinUnicode => {
                let len = r.read_u32::<LittleEndian>()? as usize;
                read_string(r, &mut stack, len)?
            }
            OpCode::ShortBinString => {
                let len = r.read_u8()? as usize;
                read_string(r, &mut stack, len)?
            }
            OpCode::Long1 => read_long1(r, &mut stack)?,
            OpCode::None => stack.push(Object::None),
            OpCode::NewTrue => stack.push(Object::Bool(true)),
            OpCode::NewFalse => stack.push(Object::Bool(false)),
            OpCode::EmptyTuple => stack.push(Object::Tuple(Vec::new())),
            OpCode::EmptyList => stack.push(Object::List(Vec::new())),
            OpCode::EmptyDict => stack.push(Object::Dict(HashMap::new())),
            OpCode::Tuple => {
                let objs = stack.pop_to_marker()?;
                stack.push(Object::Tuple(objs))
            }
            OpCode::Tuple1 => {
                let obj = stack.pop()?;
                stack.push(Object::Tuple(vec![obj]))
            }
            OpCode::Tuple2 => {
                let obj2 = stack.pop()?;
                let obj1 = stack.pop()?;
                stack.push(Object::Tuple(vec![obj1, obj2]))
            }
            OpCode::Tuple3 => {
                let obj3 = stack.pop()?;
                let obj2 = stack.pop()?;
                let obj1 = stack.pop()?;
                stack.push(Object::Tuple(vec![obj1, obj2, obj3]))
            }
            OpCode::Append => {
                let value = stack.pop()?;
                match stack.last_mut()? {
                    Object::List(list) => list.push(value),
                    _ => return Err(PickleError::UnexpectedOpCode(op_code)),
                }
            }
            OpCode::Appends => {
                let objs = stack.pop_to_marker()?;
                match stack.last_mut()? {
                    Object::List(list) => list.extend(objs),
                    _ => return Err(PickleError::UnexpectedOpCode(op_code)),
                }
            }
            OpCode::SetItem => {
                let value = stack.pop()?;
                let key = stack.pop()?;
                match stack.last_mut()? {
                    Object::Dict(dict) => {
                        if let Object::String(key) = key {
                            dict.insert(key, value);
                        } else {
                            return Err(PickleError::InvalidData(
                                "dict key must be a string".to_string(),
                            ));
                        }
                    }
                    _ => return Err(PickleError::UnexpectedOpCode(op_code)),
                }
            }
            OpCode::SetItems => {
                let mut objs = stack.pop_to_marker()?;
                if objs.len() % 2 != 0 {
                    return Err(PickleError::InvalidData(
                        "setitems requires even number of objects".to_string(),
                    ));
                }
                match stack.last_mut()? {
                    Object::Dict(dict) => {
                        while !objs.is_empty() {
                            let key = objs.remove(0);
                            let value = objs.remove(0);
                            if let Object::String(key) = key {
                                dict.insert(key, value);
                            } else {
                                return Err(PickleError::InvalidData(
                                    "dict key must be a string".to_string(),
                                ));
                            }
                        }
                    }
                    _ => return Err(PickleError::UnexpectedOpCode(op_code)),
                }
            }
            OpCode::BinPut => {
                let idx = r.read_u8()? as u32;
                let obj = stack.top()?;
                stack.memo_put(idx, obj);
            }
            OpCode::LongBinPut => {
                let idx = r.read_u32::<LittleEndian>()?;
                let obj = stack.top()?;
                stack.memo_put(idx, obj);
            }
            OpCode::BinGet => {
                let idx = r.read_u8()? as u32;
                let obj = stack.memo_get(idx)?;
                stack.push(obj);
            }
            OpCode::LongBinGet => {
                let idx = r.read_u32::<LittleEndian>()?;
                let obj = stack.memo_get(idx)?;
                stack.push(obj);
            }
            OpCode::Mark => stack.push_mark(),
            OpCode::BinPersId => {
                let pid = stack.pop()?;
                match pid {
                    Object::String(s) => {
                        stack.push(Object::Persistent(s.into_bytes()));
                    }
                    Object::Tuple(tuple) => {
                        // The persistent ID is a tuple (e.g., ('storage', 'FloatStorage', '0', 'cpu', 4))
                        // Store it as a PersistentTuple for proper handling
                        stack.push(Object::PersistentTuple(tuple));
                    }
                    _ => {
                        return Err(PickleError::InvalidData(format!(
                            "persistent id must be a string or tuple, got {:?}",
                            pid
                        )));
                    }
                }
            }
            OpCode::Reduce => {
                let args = stack.pop()?;
                let callable = stack.pop()?;

                // Check if this is an OrderedDict
                if let Object::Class { module_name, name } = &callable {
                    if module_name == "collections" && name == "OrderedDict" {
                        // OrderedDict is created with empty args, just push an empty dict
                        stack.push(Object::Dict(HashMap::new()));
                    } else {
                        let _obj = Object::Reduce {
                            callable: Box::new(callable.clone()),
                            args: Box::new(args.clone()),
                        };
                        let obj = rebuild_from_type_v2(
                            Object::Tuple(vec![callable, args]),
                            &mut stack.memo,
                            &stack.data_source,
                        )?;
                        stack.push(obj);
                    }
                } else {
                    let _obj = Object::Reduce {
                        callable: Box::new(callable.clone()),
                        args: Box::new(args.clone()),
                    };
                    let obj = rebuild_from_type_v2(
                        Object::Tuple(vec![callable, args]),
                        &mut stack.memo,
                        &stack.data_source,
                    )?;
                    stack.push(obj);
                }
            }
            OpCode::Build => {
                let args = stack.pop()?;
                let obj = stack.pop()?;
                match obj {
                    Object::Dict(mut dict) => {
                        // For dicts, BUILD updates with the args
                        if let Object::Dict(update) = args {
                            dict.extend(update);
                        }
                        stack.push(Object::Dict(dict));
                    }
                    _ => {
                        stack.push(Object::Build {
                            callable: Box::new(obj),
                            args: Box::new(args),
                        });
                    }
                }
            }
            OpCode::NewObj => {
                let args = stack.pop()?;
                let cls = stack.pop()?;
                stack.push(Object::Reduce {
                    callable: Box::new(cls),
                    args: Box::new(args),
                });
            }
            OpCode::Dict => {
                let objs = stack.pop_to_marker()?;
                let mut dict = HashMap::new();
                if objs.len() % 2 != 0 {
                    return Err(PickleError::InvalidData(
                        "dict requires even number of objects".to_string(),
                    ));
                }
                for chunk in objs.chunks(2) {
                    if let Object::String(key) = &chunk[0] {
                        dict.insert(key.clone(), chunk[1].clone());
                    } else {
                        return Err(PickleError::InvalidData(
                            "dict key must be a string".to_string(),
                        ));
                    }
                }
                stack.push(Object::Dict(dict));
            }
            OpCode::List => {
                let objs = stack.pop_to_marker()?;
                stack.push(Object::List(objs));
            }
            OpCode::Memoize => {
                // Store top of stack in memo without popping
                // The memo index is the current number of items in the memo
                let obj = stack.top()?;
                let idx = stack.memo_len() as u32;
                stack.memo_put(idx, obj);
            }
            OpCode::Stop => break,
        }
    }
    stack.pop()
}

/// Load tensors from a pickle file (PyTorch checkpoint format)
pub fn read_pickle_tensors<R: BufRead>(reader: &mut R) -> Result<HashMap<String, TensorSnapshot>> {
    let obj = read_pickle(reader)?;

    // Extract tensors from the loaded object
    let mut tensors = HashMap::new();
    let mut path = Vec::new();
    extract_tensors(&obj, &mut path, &mut tensors);

    Ok(tensors)
}

fn extract_tensors<'a>(
    obj: &'a Object,
    path: &mut Vec<&'a str>,
    tensors: &mut HashMap<String, TensorSnapshot>,
) {
    match obj {
        Object::Dict(dict) => {
            for (key, value) in dict {
                path.push(key);
                extract_tensors(value, path, tensors);
                path.pop();
            }
        }
        Object::TorchParam(snapshot) => {
            // Only allocate the string here when we actually insert
            tensors.insert(path.join("."), snapshot.clone());
        }
        _ => {}
    }
}
