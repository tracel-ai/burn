//! Just enough pickle support to read PyTorch checkpoints.
//!
//! This implementation started from the candle project's pickle loader and has since been
//! reworked around lazy tensor data, the full range of pickle protocols PyTorch emits, and a
//! single tensor-building path shared by every container format.
//!
//! Original source: <https://github.com/huggingface/candle/blob/main/candle-core/src/pickle.rs>
//!
//! The parser is a stack machine like CPython's unpickler. Two hooks make it PyTorch aware:
//! persistent ids resolve to storage references (or, for the old TAR container, to tensors
//! built ahead of time), and `REDUCE` calls of `torch._utils._rebuild_tensor*` turn a storage
//! reference into a tensor whose bytes are read only when asked for.

use super::storage::{StorageSource, read_exact_len};
use crate::bridge;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use burn_core::tensor::{BoolStore, DType, TensorData};
use burn_pack::{Error as PackError, Tensor as PackTensor};
use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::io::{self, BufRead};
use std::sync::Arc;

/// Materialized size above which a broadcast view is refused.
///
/// A view whose logical size exceeds its storage (an `expand`, stride 0) has no upper bound
/// derivable from the file: a few bytes of storage can declare a shape of any size. Ordinary
/// expanded tensors in checkpoints are tiny, so a fixed ceiling refuses the pathological
/// case without touching real files.
pub(crate) const MAX_BROADCAST_BYTES: usize = 1 << 30;

/// Error type for pickle operations.
#[derive(Debug)]
pub enum PickleError {
    Io(io::Error),
    /// A byte that is not a pickle opcode, or an opcode this reader does not implement.
    InvalidOpCode(u8),
    /// An opcode that is understood but deliberately unsupported (extension registry,
    /// out-of-band buffers).
    UnsupportedOpCode(OpCode),
    InvalidProtocol(u8),
    UnexpectedOpCode(OpCode),
    UnsupportedType(String),
    InvalidData(String),
    StackUnderflow,
    MemoNotFound(u32),
    /// The pickle references tensor storages but nothing can supply their bytes.
    NoDataSource,
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
                "Invalid pickle opcode: 0x{:02x}. The file may be corrupted or use an unsupported pickle feature.",
                code
            ),
            PickleError::UnsupportedOpCode(op) => {
                write!(f, "Unsupported pickle opcode {:?}", op)
            }
            PickleError::InvalidProtocol(proto) => write!(
                f,
                "Invalid or unsupported pickle protocol version: {}. Supported versions are 0-5.",
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
            PickleError::NoDataSource => write!(
                f,
                "Pickle references tensor storages but no data source is available"
            ),
        }
    }
}

impl std::error::Error for PickleError {}

type Result<T> = std::result::Result<T, PickleError>;

// https://github.com/python/cpython/blob/main/Lib/pickletools.py
#[repr(u8)]
#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum OpCode {
    // Protocol 0
    Int = b'I',
    Long = b'L',
    Float = b'F',
    Unicode = b'V',
    Get = b'g',
    Put = b'p',
    PersId = b'P',
    Global = b'c',
    Mark = b'(',
    Stop = b'.',
    Pop = b'0',
    PopMark = b'1',
    Dup = b'2',
    None = b'N',
    Reduce = b'R',
    Build = b'b',
    Dict = b'd',
    List = b'l',
    Tuple = b't',
    SetItem = b's',
    Append = b'a',
    // Protocol 1
    BinInt = b'J',
    BinInt1 = b'K',
    BinInt2 = b'M',
    BinFloat = b'G',
    BinString = b'T',
    ShortBinString = b'U',
    BinUnicode = b'X',
    EmptyTuple = b')',
    EmptyList = b']',
    EmptyDict = b'}',
    Appends = b'e',
    SetItems = b'u',
    BinGet = b'h',
    LongBinGet = b'j',
    BinPut = b'q',
    LongBinPut = b'r',
    BinPersId = b'Q',
    // Protocol 2
    Proto = 0x80,
    NewObj = 0x81,
    Ext1 = 0x82,
    Ext2 = 0x83,
    Ext4 = 0x84,
    Tuple1 = 0x85,
    Tuple2 = 0x86,
    Tuple3 = 0x87,
    NewTrue = 0x88,
    NewFalse = 0x89,
    Long1 = 0x8a,
    Long4 = 0x8b,
    // Protocol 3
    BinBytes = b'B',
    ShortBinBytes = b'C',
    // Protocol 4
    ShortBinUnicode = 0x8c,
    BinUnicode8 = 0x8d,
    BinBytes8 = 0x8e,
    EmptySet = 0x8f,
    AddItems = 0x90,
    FrozenSet = 0x91,
    NewObjEx = 0x92,
    StackGlobal = 0x93,
    Memoize = 0x94,
    Frame = 0x95,
    // Protocol 5
    ByteArray8 = 0x96,
    NextBuffer = 0x97,
    ReadonlyBuffer = 0x98,
}

impl TryFrom<u8> for OpCode {
    type Error = u8;
    fn try_from(value: u8) -> std::result::Result<Self, Self::Error> {
        use OpCode::*;
        const ALL: [OpCode; 65] = [
            Int,
            Long,
            Float,
            Unicode,
            Get,
            Put,
            PersId,
            Global,
            Mark,
            Stop,
            Pop,
            PopMark,
            Dup,
            None,
            Reduce,
            Build,
            Dict,
            List,
            Tuple,
            SetItem,
            Append,
            BinInt,
            BinInt1,
            BinInt2,
            BinFloat,
            BinString,
            ShortBinString,
            BinUnicode,
            EmptyTuple,
            EmptyList,
            EmptyDict,
            Appends,
            SetItems,
            BinGet,
            LongBinGet,
            BinPut,
            LongBinPut,
            BinPersId,
            Proto,
            NewObj,
            Ext1,
            Ext2,
            Ext4,
            Tuple1,
            Tuple2,
            Tuple3,
            NewTrue,
            NewFalse,
            Long1,
            Long4,
            BinBytes,
            ShortBinBytes,
            ShortBinUnicode,
            BinUnicode8,
            BinBytes8,
            EmptySet,
            AddItems,
            FrozenSet,
            NewObjEx,
            StackGlobal,
            Memoize,
            Frame,
            ByteArray8,
            NextBuffer,
            ReadonlyBuffer,
        ];
        ALL.into_iter().find(|op| *op as u8 == value).ok_or(value)
    }
}

/// A storage referenced by a persistent id, before any tensor is built on it.
#[derive(Debug, Clone)]
pub struct StorageRef {
    /// Key of the storage within its container (`"0"`, `"1"`, ...).
    pub key: String,
    /// Element type of a typed storage; `None` for `torch.UntypedStorage`.
    pub dtype: Option<DType>,
    /// Size of the storage in bytes as declared by the pickle.
    pub byte_len: usize,
    /// Element offset into the root storage when the persistent id describes a view
    /// (written by very old PyTorch versions).
    pub view_offset: usize,
}

#[derive(Debug, Clone)]
pub enum Object {
    /// The `MARK` sentinel; never part of a finished object.
    Mark,
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Bytes(Vec<u8>),
    Tuple(Vec<Object>),
    List(Vec<Object>),
    Dict(HashMap<String, Object>),
    Class {
        module_name: String,
        name: String,
    },
    Storage(StorageRef),
    /// A `REDUCE` or `NEWOBJ` this reader does not interpret, kept opaque.
    Reduce {
        callable: Box<Object>,
        args: Box<Object>,
    },
    /// A `BUILD` applied to something other than a dict, kept opaque.
    Build {
        object: Box<Object>,
        state: Box<Object>,
    },
    Tensor(PackTensor),
}

impl Object {
    /// Recursively counts the total number of nodes in the object tree.
    /// Used to prevent CPU exhaustion from pickle memo bomb attacks.
    fn node_count(&self) -> usize {
        match self {
            Object::Tuple(v) | Object::List(v) => {
                1 + v.iter().map(|o| o.node_count()).sum::<usize>()
            }
            Object::Dict(m) => 1 + m.values().map(|o| o.node_count()).sum::<usize>(),
            Object::Reduce { callable, args } => 1 + callable.node_count() + args.node_count(),
            Object::Build { object, state } => 1 + object.node_count() + state.node_count(),
            _ => 1,
        }
    }

    fn type_name(&self) -> &'static str {
        match self {
            Object::Mark => "mark",
            Object::None => "None",
            Object::Bool(_) => "bool",
            Object::Int(_) => "int",
            Object::Float(_) => "float",
            Object::String(_) => "str",
            Object::Bytes(_) => "bytes",
            Object::Tuple(_) => "tuple",
            Object::List(_) => "list",
            Object::Dict(_) => "dict",
            Object::Class { .. } => "class",
            Object::Storage(_) => "storage",
            Object::Reduce { .. } => "object",
            Object::Build { .. } => "object",
            Object::Tensor(_) => "tensor",
        }
    }
}

/// How persistent ids in a pickle are resolved.
pub(crate) enum PersistentIds {
    /// Nothing can supply storage bytes; any persistent id is an error.
    Unavailable,
    /// `('storage', type, key, location, numel[, view])` tuples, backed by a source.
    Storages(Arc<StorageSource>),
    /// Ids naming tensors built ahead of time (the TAR container).
    Tensors(HashMap<String, PackTensor>),
}

/// Parse one pickle from `r`, leaving it positioned just after the `STOP` opcode.
pub(crate) fn read_pickle<R: BufRead>(r: &mut R, ids: &PersistentIds) -> Result<Object> {
    Unpickler::new(ids).run(r)
}

// ---------------------------------------------------------------------------------------------
// Persistent ids and REDUCE dispatch
// ---------------------------------------------------------------------------------------------

/// Convert a PyTorch storage class name to its element type.
pub(crate) fn storage_type_to_dtype(storage_type: &str) -> Result<DType> {
    match storage_type {
        "FloatStorage" => Ok(DType::F32),
        "DoubleStorage" => Ok(DType::F64),
        "HalfStorage" => Ok(DType::F16),
        "BFloat16Storage" => Ok(DType::BF16),
        "LongStorage" => Ok(DType::I64),
        "IntStorage" => Ok(DType::I32),
        "ShortStorage" => Ok(DType::I16),
        "CharStorage" => Ok(DType::I8),
        "ByteStorage" => Ok(DType::U8),
        "BoolStorage" => Ok(DType::Bool(BoolStore::Native)),
        _ => Err(PickleError::UnsupportedType(format!(
            "torch.{storage_type}"
        ))),
    }
}

/// Convert a `torch.<dtype>` attribute name to its element type.
fn torch_dtype_to_dtype(name: &str) -> Result<DType> {
    match name {
        "float32" | "float" => Ok(DType::F32),
        "float64" | "double" => Ok(DType::F64),
        "float16" | "half" => Ok(DType::F16),
        "bfloat16" => Ok(DType::BF16),
        "int64" | "long" => Ok(DType::I64),
        "int32" | "int" => Ok(DType::I32),
        "int16" | "short" => Ok(DType::I16),
        "int8" => Ok(DType::I8),
        "uint8" => Ok(DType::U8),
        "uint16" => Ok(DType::U16),
        "uint32" => Ok(DType::U32),
        "uint64" => Ok(DType::U64),
        "bool" => Ok(DType::Bool(BoolStore::Native)),
        _ => Err(PickleError::UnsupportedType(format!("torch.{name}"))),
    }
}

fn non_negative(value: &Object, what: &str) -> Result<usize> {
    match value {
        Object::Int(i) => usize::try_from(*i)
            .map_err(|_| PickleError::InvalidData(format!("{what} must be non-negative, got {i}"))),
        other => Err(PickleError::InvalidData(format!(
            "{what} must be an int, got {}",
            other.type_name()
        ))),
    }
}

fn key_string(value: &Object, what: &str) -> Result<String> {
    match value {
        Object::String(s) => Ok(s.clone()),
        Object::Int(i) => Ok(i.to_string()),
        other => Err(PickleError::InvalidData(format!(
            "{what} must be a str or int, got {}",
            other.type_name()
        ))),
    }
}

/// Resolve a `('storage', storage_type, key, location, numel[, view_metadata])` tuple.
fn resolve_storage_id(pid: &[Object], source: &StorageSource) -> Result<StorageRef> {
    if pid.len() < 5 {
        return Err(PickleError::InvalidData(format!(
            "storage persistent id has {} fields, expected at least 5",
            pid.len()
        )));
    }
    match &pid[0] {
        Object::String(tag) if tag == "storage" => {}
        other => {
            return Err(PickleError::InvalidData(format!(
                "persistent id tag must be 'storage', got {}",
                other.type_name()
            )));
        }
    }

    let type_name = match &pid[1] {
        Object::Class { name, .. } | Object::String(name) => name.as_str(),
        other => {
            return Err(PickleError::InvalidData(format!(
                "storage type must be a class, got {}",
                other.type_name()
            )));
        }
    };
    let dtype = if type_name == "UntypedStorage" {
        None
    } else {
        Some(storage_type_to_dtype(type_name)?)
    };

    let key = key_string(&pid[2], "storage key")?;
    // pid[3] is the device location, irrelevant for loading.
    let numel = non_negative(&pid[4], "storage element count")?;
    let element_size = dtype.map_or(1, |dtype| dtype.size());
    let byte_len = numel.checked_mul(element_size).ok_or_else(|| {
        PickleError::InvalidData(format!("storage '{key}' byte length overflows usize"))
    })?;

    // Very old files may describe a view of a root storage: (view_key, offset, view_size).
    let view_offset = match pid.get(5) {
        None | Some(Object::None) => 0,
        Some(Object::Tuple(view)) if view.len() == 3 => {
            non_negative(&view[1], "storage view offset")?
        }
        Some(other) => {
            return Err(PickleError::InvalidData(format!(
                "storage view metadata must be None or a 3-tuple, got {}",
                other.type_name()
            )));
        }
    };

    source
        .declare(&key, byte_len, element_size)
        .map_err(|err| PickleError::InvalidData(err.to_string()))?;

    Ok(StorageRef {
        key,
        dtype,
        byte_len,
        view_offset,
    })
}

fn resolve_persistent_id(pid: Object, ids: &PersistentIds) -> Result<Object> {
    match ids {
        PersistentIds::Unavailable => Err(PickleError::NoDataSource),
        PersistentIds::Storages(source) => match pid {
            Object::Tuple(fields) => Ok(Object::Storage(resolve_storage_id(&fields, source)?)),
            other => Err(PickleError::InvalidData(format!(
                "persistent id must be a tuple, got {}",
                other.type_name()
            ))),
        },
        PersistentIds::Tensors(tensors) => {
            let key = key_string(&pid, "persistent id")?;
            tensors
                .get(&key)
                .cloned()
                .map(Object::Tensor)
                .ok_or_else(|| {
                    PickleError::InvalidData(format!(
                        "persistent id '{key}' does not name a tensor in the archive"
                    ))
                })
        }
    }
}

/// Apply a `REDUCE`: call `callable` with `args`.
///
/// Only the calls PyTorch uses to rebuild tensors and dicts are interpreted. Any other call
/// is kept as an opaque object so a checkpoint carrying, say, numpy scalars or a device in
/// its metadata still loads its tensors.
fn reduce(callable: Object, args: Object, ids: &PersistentIds) -> Result<Object> {
    let (module_name, name) = match &callable {
        Object::Class { module_name, name } => (module_name.as_str(), name.as_str()),
        _ => return Ok(opaque_reduce(callable, args)),
    };

    match (module_name, name) {
        ("collections", "OrderedDict") => ordered_dict(args),
        ("torch._utils", "_rebuild_tensor") => rebuild_tensor(args, ids, TensorRebuild::Legacy),
        ("torch._utils", "_rebuild_tensor_v2") => rebuild_tensor(args, ids, TensorRebuild::V2),
        ("torch._utils", "_rebuild_tensor_v3") => rebuild_tensor(args, ids, TensorRebuild::V3),
        // _rebuild_parameter(data, requires_grad, backward_hooks[, state])
        ("torch._utils", "_rebuild_parameter" | "_rebuild_parameter_with_state") => match args {
            Object::Tuple(mut fields) if !fields.is_empty() => Ok(fields.swap_remove(0)),
            other => Err(PickleError::InvalidData(format!(
                "{name}: expected a non-empty argument tuple, got {}",
                other.type_name()
            ))),
        },
        // _rebuild_from_type_v2(func, new_type, args, state): the tensor is func(*args).
        ("torch._tensor", "_rebuild_from_type_v2") => match args {
            Object::Tuple(mut fields) if fields.len() >= 3 => {
                let inner_args = fields.swap_remove(2);
                let func = fields.swap_remove(0);
                reduce(func, inner_args, ids)
            }
            other => Err(PickleError::InvalidData(format!(
                "_rebuild_from_type_v2: expected at least 3 arguments, got {}",
                other.type_name()
            ))),
        },
        // Sparse, quantized, nested and meta tensors are rebuilt through other helpers in
        // this module. Those are tensors this reader cannot represent, which is worth an
        // error rather than a silently missing entry.
        ("torch._utils", name) if name.starts_with("_rebuild") => Err(
            PickleError::UnsupportedType(format!("{module_name}.{name}")),
        ),
        _ => Ok(opaque_reduce(callable, args)),
    }
}

fn opaque_reduce(callable: Object, args: Object) -> Object {
    Object::Reduce {
        callable: Box::new(callable),
        args: Box::new(args),
    }
}

/// `OrderedDict()` or `OrderedDict([(key, value), ...])`.
fn ordered_dict(args: Object) -> Result<Object> {
    let items = match args {
        Object::Tuple(mut fields) if !fields.is_empty() => match fields.swap_remove(0) {
            Object::List(items) | Object::Tuple(items) => items,
            _ => Vec::new(),
        },
        _ => Vec::new(),
    };

    let mut dict = HashMap::with_capacity(items.len());
    for item in items {
        match item {
            Object::List(mut pair) | Object::Tuple(mut pair) if pair.len() == 2 => {
                let value = pair.pop().expect("pair has two items");
                let key = pair.pop().expect("pair has two items");
                dict.insert(dict_key(key)?, value);
            }
            other => {
                return Err(PickleError::InvalidData(format!(
                    "OrderedDict items must be (key, value) pairs, got {}",
                    other.type_name()
                )));
            }
        }
    }
    Ok(Object::Dict(dict))
}

#[derive(Clone, Copy)]
enum TensorRebuild {
    /// `_rebuild_tensor(storage, storage_offset, size, stride)`, PyTorch < 1.6.
    Legacy,
    /// `_rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, hooks[, metadata])`.
    V2,
    /// `_rebuild_tensor_v3(storage, storage_offset, size, stride, requires_grad, hooks, dtype[, metadata])`,
    /// used for element types without a typed storage class (unsigned ints, float8).
    V3,
}

impl TensorRebuild {
    fn name(self) -> &'static str {
        match self {
            Self::Legacy => "_rebuild_tensor",
            Self::V2 => "_rebuild_tensor_v2",
            Self::V3 => "_rebuild_tensor_v3",
        }
    }

    fn arity(self) -> usize {
        match self {
            Self::Legacy => 4,
            Self::V2 => 6,
            Self::V3 => 7,
        }
    }
}

fn rebuild_tensor(args: Object, ids: &PersistentIds, kind: TensorRebuild) -> Result<Object> {
    let fn_name = kind.name();
    let fields = match args {
        Object::Tuple(fields) => fields,
        other => {
            return Err(PickleError::InvalidData(format!(
                "{fn_name}: expected an argument tuple, got {}",
                other.type_name()
            )));
        }
    };
    if fields.len() < kind.arity() {
        return Err(PickleError::InvalidData(format!(
            "{fn_name}: expected at least {} arguments, got {}",
            kind.arity(),
            fields.len()
        )));
    }

    let storage = match &fields[0] {
        Object::Storage(storage) => storage.clone(),
        other => {
            return Err(PickleError::InvalidData(format!(
                "{fn_name}: expected a storage, got {}",
                other.type_name()
            )));
        }
    };
    let storage_offset = non_negative(&fields[1], "storage offset")?;
    let shape = parse_dims(&fields[2], "shape")?;
    let stride = parse_dims(&fields[3], "stride")?;

    let dtype = match kind {
        TensorRebuild::V3 => match &fields[6] {
            Object::Class { name, .. } => torch_dtype_to_dtype(name)?,
            other => {
                return Err(PickleError::InvalidData(format!(
                    "{fn_name}: expected a torch dtype, got {}",
                    other.type_name()
                )));
            }
        },
        _ => storage.dtype.ok_or_else(|| {
            PickleError::InvalidData(format!(
                "{fn_name}: untyped storage '{}' needs an explicit dtype",
                storage.key
            ))
        })?,
    };
    if let Some(storage_dtype) = storage.dtype
        && storage_dtype != dtype
    {
        return Err(PickleError::InvalidData(format!(
            "{fn_name}: storage '{}' holds {storage_dtype:?} but the tensor is {dtype:?}",
            storage.key
        )));
    }

    let source = match ids {
        PersistentIds::Storages(source) => source.clone(),
        _ => return Err(PickleError::NoDataSource),
    };

    build_tensor(storage, dtype, storage_offset, shape, stride, source).map(Object::Tensor)
}

fn parse_dims(value: &Object, what: &str) -> Result<Vec<usize>> {
    match value {
        Object::Tuple(items) | Object::List(items) => {
            items.iter().map(|item| non_negative(item, what)).collect()
        }
        other => Err(PickleError::InvalidData(format!(
            "{what} must be a tuple, got {}",
            other.type_name()
        ))),
    }
}

// ---------------------------------------------------------------------------------------------
// Tensor construction
// ---------------------------------------------------------------------------------------------

/// Return the number of logical elements, rejecting shape arithmetic overflow.
fn tensor_num_elements(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |total, dim| {
        total.checked_mul(*dim).ok_or_else(|| {
            PickleError::InvalidData("Tensor shape element count overflows usize".to_string())
        })
    })
}

/// Return the minimum number of storage elements required by a strided tensor view.
///
/// For a non-empty view, this is one past the largest storage index it reads. An empty view
/// requires only its storage offset. Size-one dimensions still contribute no storage movement.
fn tensor_storage_extent(
    shape: &[usize],
    stride: &[usize],
    storage_offset: usize,
) -> Result<usize> {
    if shape.len() != stride.len() {
        return Err(PickleError::InvalidData(format!(
            "Tensor stride rank {} does not match shape rank {}",
            stride.len(),
            shape.len()
        )));
    }

    if tensor_num_elements(shape)? == 0 {
        return Ok(storage_offset);
    }

    let max_index = shape
        .iter()
        .zip(stride)
        .try_fold(storage_offset, |index, (&dim, &step)| {
            let offset = (dim - 1).checked_mul(step).ok_or_else(|| {
                PickleError::InvalidData("Tensor stride calculation overflows usize".to_string())
            })?;
            index.checked_add(offset).ok_or_else(|| {
                PickleError::InvalidData("Tensor storage extent overflows usize".to_string())
            })
        })?;

    max_index.checked_add(1).ok_or_else(|| {
        PickleError::InvalidData("Tensor storage extent overflows usize".to_string())
    })
}

/// Check whether a stride describes row-major contiguous storage.
///
/// PyTorch ignores strides for size-one dimensions because those dimensions cannot be stepped.
fn is_contiguous_stride(shape: &[usize], stride: &[usize]) -> bool {
    let mut expected = 1usize;
    for (&dim, &step) in shape.iter().zip(stride).rev() {
        if dim > 1 && step != expected {
            return false;
        }
        let Some(next) = expected.checked_mul(dim) else {
            return false;
        };
        expected = next;
    }
    true
}

/// Gather a strided view into logical row-major byte order.
///
/// The caller has validated that every storage index the view touches lies within `data`.
fn gather_strided(
    data: &[u8],
    shape: &[usize],
    stride: &[usize],
    storage_offset: usize,
    element_size: usize,
    byte_len: usize,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(byte_len);
    let num_elements = byte_len / element_size;
    for linear_index in 0..num_elements {
        let mut remaining = linear_index;
        let mut storage_index = storage_offset;
        for (&dim, &step) in shape.iter().zip(stride).rev() {
            storage_index += (remaining % dim) * step;
            remaining /= dim;
        }
        let start = storage_index * element_size;
        out.extend_from_slice(&data[start..start + element_size]);
    }
    out
}

/// Convert little-endian file bytes to the target's native byte order.
fn to_native_endian(bytes: &mut [u8], element_size: usize) {
    if cfg!(target_endian = "big") && element_size > 1 {
        for element in bytes.chunks_exact_mut(element_size) {
            element.reverse();
        }
    }
}

/// Build a tensor that reads its bytes from `source` on demand.
///
/// Everything derivable from the metadata is validated here, so a file whose declarations are
/// inconsistent fails at parse time; the data itself is validated when it is read.
pub(crate) fn build_tensor(
    storage: StorageRef,
    dtype: DType,
    storage_offset: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,
    source: Arc<StorageSource>,
) -> Result<PackTensor> {
    let element_size = dtype.size();
    let storage_offset = storage_offset
        .checked_add(storage.view_offset)
        .ok_or_else(|| PickleError::InvalidData("Storage offset overflows usize".to_string()))?;
    let num_elements = tensor_num_elements(&shape)?;
    let extent = tensor_storage_extent(&shape, &stride, storage_offset)?;
    let byte_len = num_elements.checked_mul(element_size).ok_or_else(|| {
        PickleError::InvalidData("Tensor byte length overflows usize".to_string())
    })?;

    let declared_elements = storage.byte_len / element_size;
    if extent > declared_elements {
        return Err(PickleError::InvalidData(format!(
            "Tensor with shape {shape:?} and stride {stride:?} at offset {storage_offset} needs {extent} elements, but storage '{}' declares {declared_elements}",
            storage.key
        )));
    }
    if byte_len > storage.byte_len && byte_len > MAX_BROADCAST_BYTES {
        return Err(PickleError::InvalidData(format!(
            "Tensor with shape {shape:?} would materialize {byte_len} bytes from a {} byte storage, above the {MAX_BROADCAST_BYTES} byte limit for broadcast views",
            storage.byte_len
        )));
    }

    let key = storage.key;
    let provider_shape = shape.clone();
    let provider = move || -> std::result::Result<TensorData, PackError> {
        let shape = &provider_shape;
        let mut data = source.read(&key).map_err(|err| {
            PackError::ValidationError(format!(
                "Failed to read storage '{key}' for tensor with shape {shape:?}: {err}"
            ))
        })?;

        let available = data.len() / element_size;
        if extent > available {
            return Err(PackError::ValidationError(format!(
                "Tensor with shape {shape:?} requires {extent} elements from storage '{key}', but only {available} are available"
            )));
        }

        let mut bytes = if is_contiguous_stride(shape, &stride) {
            let start = storage_offset * element_size;
            data.truncate(start + byte_len);
            data.drain(..start);
            data
        } else {
            gather_strided(
                &data,
                shape,
                &stride,
                storage_offset,
                element_size,
                byte_len,
            )
        };

        to_native_endian(&mut bytes, element_size);
        if matches!(dtype, DType::Bool(_)) {
            // PyTorch writes 0 or 1, but only those two bit patterns are a valid `bool`.
            for byte in &mut bytes {
                *byte = u8::from(*byte != 0);
            }
        }

        Ok(TensorData::from_bytes_vec(bytes, shape.clone(), dtype))
    };

    // The tensor's name is a path through the pickle's dicts, assembled by the caller that
    // walks them. PyTorch carries no parameter identity, so `param_id` stays empty.
    Ok(bridge::deferred(
        String::new(),
        dtype,
        shape.into(),
        None,
        provider,
    ))
}

// ---------------------------------------------------------------------------------------------
// The stack machine
// ---------------------------------------------------------------------------------------------

/// Total nodes that may be re-materialized through memo lookups before the pickle is
/// rejected as a memo bomb (a `BINGET` of a large object copies it).
const MAX_MEMO_NODES: usize = 1_000_000;

struct Unpickler<'a> {
    stack: Vec<Object>,
    memo: HashMap<u32, Object>,
    memo_nodes: usize,
    ids: &'a PersistentIds,
}

impl<'a> Unpickler<'a> {
    fn new(ids: &'a PersistentIds) -> Self {
        Self {
            stack: Vec::new(),
            memo: HashMap::new(),
            memo_nodes: 0,
            ids,
        }
    }

    fn push(&mut self, o: Object) {
        self.stack.push(o)
    }

    fn pop(&mut self) -> Result<Object> {
        self.stack.pop().ok_or(PickleError::StackUnderflow)
    }

    fn top(&self) -> Result<&Object> {
        self.stack.last().ok_or(PickleError::StackUnderflow)
    }

    fn top_mut(&mut self) -> Result<&mut Object> {
        self.stack.last_mut().ok_or(PickleError::StackUnderflow)
    }

    fn pop_to_mark(&mut self) -> Result<Vec<Object>> {
        let mark = self
            .stack
            .iter()
            .rposition(|o| matches!(o, Object::Mark))
            .ok_or_else(|| PickleError::InvalidData("MARK not found on stack".to_string()))?;
        let items = self.stack.split_off(mark + 1);
        self.stack.pop();
        Ok(items)
    }

    fn memo_get(&mut self, idx: u32) -> Result<Object> {
        let obj = self.memo.get(&idx).ok_or(PickleError::MemoNotFound(idx))?;
        self.memo_nodes += obj.node_count();
        if self.memo_nodes > MAX_MEMO_NODES {
            return Err(PickleError::InvalidData(format!(
                "Pickle memo bomb detected: exceeded {MAX_MEMO_NODES} nodes"
            )));
        }
        Ok(obj.clone())
    }

    fn memo_put(&mut self, idx: u32) -> Result<()> {
        let obj = self.top()?.clone();
        self.memo.insert(idx, obj);
        Ok(())
    }

    fn run<R: BufRead>(mut self, r: &mut R) -> Result<Object> {
        loop {
            let byte = r.read_u8()?;
            let op = OpCode::try_from(byte).map_err(PickleError::InvalidOpCode)?;
            match op {
                OpCode::Proto => {
                    let version = r.read_u8()?;
                    if version > 5 {
                        return Err(PickleError::InvalidProtocol(version));
                    }
                }
                OpCode::Frame => {
                    // Framing only matters for streaming readers; the frame size is not needed.
                    r.read_u64::<LittleEndian>()?;
                }
                OpCode::Stop => break,

                // Scalars
                OpCode::None => self.push(Object::None),
                OpCode::NewTrue => self.push(Object::Bool(true)),
                OpCode::NewFalse => self.push(Object::Bool(false)),
                OpCode::Int => {
                    let text = read_line(r)?;
                    // Protocol 0 spells booleans as INT 00 / INT 01.
                    let value = match text.as_str() {
                        "00" => Object::Bool(false),
                        "01" => Object::Bool(true),
                        _ => Object::Int(parse_int(&text)?),
                    };
                    self.push(value);
                }
                OpCode::Long => {
                    let text = read_line(r)?;
                    let text = text.strip_suffix('L').unwrap_or(&text);
                    self.push(Object::Int(parse_int(text)?));
                }
                OpCode::Float => {
                    let text = read_line(r)?;
                    let value = text.parse::<f64>().map_err(|e| {
                        PickleError::InvalidData(format!("Invalid FLOAT value '{text}': {e}"))
                    })?;
                    self.push(Object::Float(value));
                }
                OpCode::Unicode => {
                    // Protocol 0 raw-unicode-escape text; escapes are rare and left as is.
                    let text = read_line(r)?;
                    self.push(Object::String(text));
                }
                OpCode::BinInt => {
                    let v = r.read_i32::<LittleEndian>()?;
                    self.push(Object::Int(v as i64));
                }
                OpCode::BinInt1 => {
                    let v = r.read_u8()?;
                    self.push(Object::Int(v as i64));
                }
                OpCode::BinInt2 => {
                    let v = r.read_u16::<LittleEndian>()?;
                    self.push(Object::Int(v as i64));
                }
                OpCode::BinFloat => {
                    let v = r.read_f64::<BigEndian>()?;
                    self.push(Object::Float(v));
                }
                OpCode::Long1 => {
                    let len = r.read_u8()? as u64;
                    let bytes = read_exact_len(r, len)?;
                    self.push(Object::Int(int_from_le_bytes(&bytes)?));
                }
                OpCode::Long4 => {
                    let len = r.read_u32::<LittleEndian>()? as u64;
                    let bytes = read_exact_len(r, len)?;
                    self.push(Object::Int(int_from_le_bytes(&bytes)?));
                }

                // Strings and bytes
                OpCode::ShortBinUnicode => {
                    let len = r.read_u8()? as u64;
                    self.push(read_string(r, len)?);
                }
                OpCode::BinUnicode => {
                    let len = r.read_u32::<LittleEndian>()? as u64;
                    self.push(read_string(r, len)?);
                }
                OpCode::ShortBinString => {
                    let len = r.read_u8()? as u64;
                    self.push(read_py2_string(r, len)?);
                }
                OpCode::BinString => {
                    let len = r.read_u32::<LittleEndian>()? as u64;
                    self.push(read_py2_string(r, len)?);
                }
                OpCode::BinUnicode8 => {
                    let len = r.read_u64::<LittleEndian>()?;
                    self.push(read_string(r, len)?);
                }
                OpCode::ShortBinBytes => {
                    let len = r.read_u8()? as u64;
                    self.push(Object::Bytes(read_exact_len(r, len)?));
                }
                OpCode::BinBytes => {
                    let len = r.read_u32::<LittleEndian>()? as u64;
                    self.push(Object::Bytes(read_exact_len(r, len)?));
                }
                OpCode::BinBytes8 | OpCode::ByteArray8 => {
                    let len = r.read_u64::<LittleEndian>()?;
                    self.push(Object::Bytes(read_exact_len(r, len)?));
                }

                // Containers
                OpCode::Mark => self.push(Object::Mark),
                OpCode::EmptyTuple => self.push(Object::Tuple(Vec::new())),
                OpCode::EmptyList | OpCode::EmptySet => self.push(Object::List(Vec::new())),
                OpCode::EmptyDict => self.push(Object::Dict(HashMap::new())),
                OpCode::Tuple => {
                    let items = self.pop_to_mark()?;
                    self.push(Object::Tuple(items));
                }
                OpCode::Tuple1 => {
                    let a = self.pop()?;
                    self.push(Object::Tuple(vec![a]));
                }
                OpCode::Tuple2 => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(Object::Tuple(vec![a, b]));
                }
                OpCode::Tuple3 => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(Object::Tuple(vec![a, b, c]));
                }
                OpCode::List | OpCode::FrozenSet => {
                    let items = self.pop_to_mark()?;
                    self.push(Object::List(items));
                }
                OpCode::Append => {
                    let value = self.pop()?;
                    match self.top_mut()? {
                        Object::List(list) => list.push(value),
                        _ => return Err(PickleError::UnexpectedOpCode(op)),
                    }
                }
                OpCode::Appends | OpCode::AddItems => {
                    let items = self.pop_to_mark()?;
                    match self.top_mut()? {
                        Object::List(list) => list.extend(items),
                        _ => return Err(PickleError::UnexpectedOpCode(op)),
                    }
                }
                OpCode::Dict => {
                    let items = self.pop_to_mark()?;
                    let mut dict = HashMap::with_capacity(items.len() / 2);
                    insert_pairs(&mut dict, items)?;
                    self.push(Object::Dict(dict));
                }
                OpCode::SetItem => {
                    let value = self.pop()?;
                    let key = self.pop()?;
                    match self.top_mut()? {
                        Object::Dict(dict) => {
                            dict.insert(dict_key(key)?, value);
                        }
                        _ => return Err(PickleError::UnexpectedOpCode(op)),
                    }
                }
                OpCode::SetItems => {
                    let items = self.pop_to_mark()?;
                    match self.top_mut()? {
                        Object::Dict(dict) => insert_pairs(dict, items)?,
                        _ => return Err(PickleError::UnexpectedOpCode(op)),
                    }
                }

                // Stack manipulation
                OpCode::Pop => {
                    self.pop()?;
                }
                OpCode::PopMark => {
                    self.pop_to_mark()?;
                }
                OpCode::Dup => {
                    let top = self.top()?.clone();
                    self.push(top);
                }

                // Memo
                OpCode::Get => {
                    let idx = parse_index(&read_line(r)?)?;
                    let obj = self.memo_get(idx)?;
                    self.push(obj);
                }
                OpCode::BinGet => {
                    let idx = r.read_u8()? as u32;
                    let obj = self.memo_get(idx)?;
                    self.push(obj);
                }
                OpCode::LongBinGet => {
                    let idx = r.read_u32::<LittleEndian>()?;
                    let obj = self.memo_get(idx)?;
                    self.push(obj);
                }
                OpCode::Put => {
                    let idx = parse_index(&read_line(r)?)?;
                    self.memo_put(idx)?;
                }
                OpCode::BinPut => {
                    let idx = r.read_u8()? as u32;
                    self.memo_put(idx)?;
                }
                OpCode::LongBinPut => {
                    let idx = r.read_u32::<LittleEndian>()?;
                    self.memo_put(idx)?;
                }
                OpCode::Memoize => {
                    let idx = self.memo.len() as u32;
                    self.memo_put(idx)?;
                }

                // Objects
                OpCode::Global => {
                    let module_name = read_line(r)?;
                    let name = read_line(r)?;
                    self.push(Object::Class { module_name, name });
                }
                OpCode::StackGlobal => {
                    let name = self.pop()?;
                    let module_name = self.pop()?;
                    match (module_name, name) {
                        (Object::String(module_name), Object::String(name)) => {
                            self.push(Object::Class { module_name, name })
                        }
                        _ => {
                            return Err(PickleError::InvalidData(
                                "STACK_GLOBAL expects two strings".to_string(),
                            ));
                        }
                    }
                }
                OpCode::PersId => {
                    let pid = Object::String(read_line(r)?);
                    let obj = resolve_persistent_id(pid, self.ids)?;
                    self.push(obj);
                }
                OpCode::BinPersId => {
                    let pid = self.pop()?;
                    let obj = resolve_persistent_id(pid, self.ids)?;
                    self.push(obj);
                }
                OpCode::Reduce => {
                    let args = self.pop()?;
                    let callable = self.pop()?;
                    let obj = reduce(callable, args, self.ids)?;
                    self.push(obj);
                }
                OpCode::NewObj => {
                    let args = self.pop()?;
                    let cls = self.pop()?;
                    self.push(opaque_reduce(cls, args));
                }
                OpCode::NewObjEx => {
                    let _kwargs = self.pop()?;
                    let args = self.pop()?;
                    let cls = self.pop()?;
                    self.push(opaque_reduce(cls, args));
                }
                OpCode::Build => {
                    let state = self.pop()?;
                    let object = self.pop()?;
                    match (object, state) {
                        (Object::Dict(mut dict), Object::Dict(update)) => {
                            dict.extend(update);
                            self.push(Object::Dict(dict));
                        }
                        // A dict subclass with non-dict state: the items are what matter.
                        (dict @ Object::Dict(_), _) => self.push(dict),
                        (object, state) => self.push(Object::Build {
                            object: Box::new(object),
                            state: Box::new(state),
                        }),
                    }
                }

                OpCode::Ext1
                | OpCode::Ext2
                | OpCode::Ext4
                | OpCode::NextBuffer
                | OpCode::ReadonlyBuffer => return Err(PickleError::UnsupportedOpCode(op)),
            }
        }

        let result = self.pop()?;
        if matches!(result, Object::Mark) {
            return Err(PickleError::InvalidData(
                "pickle ended with an unmatched MARK".to_string(),
            ));
        }
        Ok(result)
    }
}

fn read_line<R: BufRead>(r: &mut R) -> Result<String> {
    let mut data = Vec::with_capacity(32);
    r.read_until(b'\n', &mut data)?;
    if data.pop() != Some(b'\n') {
        return Err(PickleError::Io(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "unterminated text field",
        )));
    }
    if data.last() == Some(&b'\r') {
        data.pop();
    }
    String::from_utf8(data).map_err(|e| PickleError::InvalidData(format!("Invalid UTF-8: {e}")))
}

fn read_string<R: BufRead>(r: &mut R, len: u64) -> Result<Object> {
    let data = read_exact_len(r, len)?;
    let s = String::from_utf8(data)
        .map_err(|e| PickleError::InvalidData(format!("Invalid UTF-8: {e}")))?;
    Ok(Object::String(s))
}

/// A Python 2 `str` is bytes that usually hold text, but numpy pickles binary data in them
/// under protocol 2. Text is kept as a string and anything else as bytes.
fn read_py2_string<R: BufRead>(r: &mut R, len: u64) -> Result<Object> {
    let data = read_exact_len(r, len)?;
    Ok(match String::from_utf8(data) {
        Ok(text) => Object::String(text),
        Err(err) => Object::Bytes(err.into_bytes()),
    })
}

fn parse_int(text: &str) -> Result<i64> {
    text.parse::<i64>()
        .map_err(|e| PickleError::InvalidData(format!("Invalid integer '{text}': {e}")))
}

fn parse_index(text: &str) -> Result<u32> {
    text.parse::<u32>()
        .map_err(|e| PickleError::InvalidData(format!("Invalid memo index '{text}': {e}")))
}

/// Decode a two's-complement little-endian integer of any length that fits in an `i64`.
fn int_from_le_bytes(bytes: &[u8]) -> Result<i64> {
    let negative = bytes.last().is_some_and(|b| b & 0x80 != 0);
    let fill = if negative { 0xff } else { 0x00 };
    if bytes.len() > 8 && bytes[8..].iter().any(|&b| b != fill) {
        return Err(PickleError::InvalidData(format!(
            "integer of {} bytes does not fit in 64 bits",
            bytes.len()
        )));
    }
    let mut buf = [fill; 8];
    let len = bytes.len().min(8);
    buf[..len].copy_from_slice(&bytes[..len]);
    Ok(i64::from_le_bytes(buf))
}

fn dict_key(key: Object) -> Result<String> {
    match key {
        Object::String(s) => Ok(s),
        Object::Int(i) => Ok(i.to_string()),
        other => Err(PickleError::InvalidData(format!(
            "dict key must be a str or int, got {}",
            other.type_name()
        ))),
    }
}

fn insert_pairs(dict: &mut HashMap<String, Object>, items: Vec<Object>) -> Result<()> {
    if !items.len().is_multiple_of(2) {
        return Err(PickleError::InvalidData(
            "dict items must come in key/value pairs".to_string(),
        ));
    }
    let mut items = items.into_iter();
    while let (Some(key), Some(value)) = (items.next(), items.next()) {
        dict.insert(dict_key(key)?, value);
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Tensor extraction
// ---------------------------------------------------------------------------------------------

/// Walk a parsed pickle object tree, collecting each tensor found under a dict path.
///
/// Only dicts are descended into: a state_dict is one, and a tensor reached any other way is
/// ignored. A tensor is built without a name, its path being assembled only here, so this is
/// also where each one gets its final identity.
pub(crate) fn extract_tensors(obj: Object, tensors: &mut HashMap<String, PackTensor>) {
    fn walk(obj: Object, path: &mut Vec<String>, tensors: &mut HashMap<String, PackTensor>) {
        match obj {
            Object::Dict(dict) => {
                for (key, value) in dict {
                    path.push(key);
                    walk(value, path, tensors);
                    path.pop();
                }
            }
            Object::Tensor(mut tensor) => {
                tensor.name = path.join(".");
                tensors.insert(tensor.name.clone(), tensor);
            }
            _ => {}
        }
    }
    walk(obj, &mut Vec::new(), tensors);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn fixture_source() -> Arc<StorageSource> {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src/pytorch/tests/reader/test_data/non_contiguous.pt");
        Arc::new(StorageSource::Zip(
            super::super::storage::ZipSource::open(&path).unwrap(),
        ))
    }

    fn storage_pid(storage_type: &str, key: &str, numel: i64) -> Object {
        Object::Tuple(vec![
            Object::String("storage".to_string()),
            Object::Class {
                module_name: "torch".to_string(),
                name: storage_type.to_string(),
            },
            Object::String(key.to_string()),
            Object::String("cpu".to_string()),
            Object::Int(numel),
        ])
    }

    fn rebuild_args(
        storage_type: &str,
        key: &str,
        numel: i64,
        offset: i64,
        shape: &[i64],
        stride: &[i64],
        source: &Arc<StorageSource>,
    ) -> Object {
        let storage = match &storage_pid(storage_type, key, numel) {
            Object::Tuple(fields) => resolve_storage_id(fields, source).unwrap(),
            _ => unreachable!(),
        };
        Object::Tuple(vec![
            Object::Storage(storage),
            Object::Int(offset),
            Object::Tuple(shape.iter().copied().map(Object::Int).collect()),
            Object::Tuple(stride.iter().copied().map(Object::Int).collect()),
        ])
    }

    fn rebuild(args: Object, source: Arc<StorageSource>) -> Result<PackTensor> {
        let ids = PersistentIds::Storages(source);
        match rebuild_tensor(args, &ids, TensorRebuild::Legacy)? {
            Object::Tensor(tensor) => Ok(tensor),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    fn plain(bytes: &[u8]) -> Result<Object> {
        read_pickle(&mut Cursor::new(bytes), &PersistentIds::Unavailable)
    }

    #[test]
    fn test_memo_bomb_mitigation() {
        // Generate the Billion Laughs equivalent for pickles
        let mut poc = vec![0x89, b'q', 0x00];
        let n = 20; // 144 bytes, originally took 2.6s and 10+ million nodes
        for _ in 0..n {
            poc.extend_from_slice(&[b'h', 0x00, b'h', 0x00, 0x86, b'q', 0x00]);
        }
        poc.push(b'.');

        let err = plain(&poc).unwrap_err();
        assert!(
            matches!(err, PickleError::InvalidData(msg) if msg.contains("exceeded 1000000 nodes"))
        );
    }

    #[test]
    fn protocol_4_pickle_parses() {
        // pickle.dumps({"a": 1, "b": [1.5, True, None], "c": b"xy", "d": {"e": "f"}}, protocol=4)
        let bytes: &[u8] = &[
            0x80, 0x04, 0x95, 0x36, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7d, 0x94, 0x28,
            0x8c, 0x01, 0x61, 0x94, 0x4b, 0x01, 0x8c, 0x01, 0x62, 0x94, 0x5d, 0x94, 0x28, 0x47,
            0x3f, 0xf8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x88, 0x4e, 0x65, 0x8c, 0x01, 0x63,
            0x94, 0x43, 0x02, 0x78, 0x79, 0x94, 0x8c, 0x01, 0x64, 0x94, 0x7d, 0x94, 0x8c, 0x01,
            0x65, 0x94, 0x8c, 0x01, 0x66, 0x94, 0x73, 0x75, 0x2e,
        ];
        let Object::Dict(dict) = plain(bytes).unwrap() else {
            panic!("expected dict");
        };
        assert!(matches!(dict["a"], Object::Int(1)));
        let Object::List(items) = &dict["b"] else {
            panic!("expected list");
        };
        assert!(
            matches!(items[..], [Object::Float(f), Object::Bool(true), Object::None] if f == 1.5)
        );
        assert!(matches!(&dict["c"], Object::Bytes(b) if b == b"xy"));
        let Object::Dict(inner) = &dict["d"] else {
            panic!("expected dict");
        };
        assert!(matches!(&inner["e"], Object::String(s) if s == "f"));
    }

    #[test]
    fn stack_global_builds_class() {
        // pickle.dumps(collections.OrderedDict, protocol=4)
        let bytes: &[u8] = &[
            0x80, 0x04, 0x95, 0x1f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x8c, 0x0b, b'c',
            b'o', b'l', b'l', b'e', b'c', b't', b'i', b'o', b'n', b's', 0x94, 0x8c, 0x0b, b'O',
            b'r', b'd', b'e', b'r', b'e', b'd', b'D', b'i', b'c', b't', 0x94, 0x93, 0x94, 0x2e,
        ];
        assert!(matches!(
            plain(bytes).unwrap(),
            Object::Class { module_name, name } if module_name == "collections" && name == "OrderedDict"
        ));
    }

    #[test]
    fn protocol_0_text_opcodes_parse() {
        // pickle.dumps({"n": -3, "f": 2.5, "t": True, "big": 2**40}, protocol=0)
        let bytes =
            b"(dp0\nVn\np1\nI-3\nsVf\np2\nF2.5\nsVt\np3\nI01\nsVbig\np4\nL1099511627776L\ns.";
        let Object::Dict(dict) = plain(bytes).unwrap_or_else(|e| panic!("{e}")) else {
            panic!("expected dict");
        };
        assert!(matches!(dict["n"], Object::Int(-3)));
        assert!(matches!(dict["f"], Object::Float(f) if f == 2.5));
        assert!(matches!(dict["t"], Object::Bool(true)));
        assert!(matches!(dict["big"], Object::Int(1099511627776)));
    }

    #[test]
    fn long1_handles_sign_and_width() {
        assert_eq!(int_from_le_bytes(&[]).unwrap(), 0);
        assert_eq!(int_from_le_bytes(&[0xff]).unwrap(), -1);
        assert_eq!(int_from_le_bytes(&[0x00, 0x01]).unwrap(), 256);
        assert_eq!(
            int_from_le_bytes(&[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f, 0x00]).unwrap(),
            i64::MAX
        );
        assert!(
            int_from_le_bytes(&[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01]).is_err()
        );
    }

    #[test]
    fn truncated_string_length_does_not_allocate() {
        // BINUNICODE claiming 4 GiB followed by two bytes.
        let bytes: &[u8] = &[0x80, 0x02, b'X', 0xff, 0xff, 0xff, 0xff, b'h', b'i', b'.'];
        assert!(matches!(plain(bytes).unwrap_err(), PickleError::Io(_)));
    }

    #[test]
    fn unknown_reduce_is_kept_opaque() {
        // The shape of a pickled numpy scalar: REDUCE of numpy.core.multiarray.scalar.
        let bytes = b"\x80\x02cnumpy.core.multiarray\nscalar\nU\x02f8U\x08\x00\x00\x00\x00\x00\x00\xe0?\x86R.";
        assert!(matches!(plain(bytes).unwrap(), Object::Reduce { .. }));
    }

    #[test]
    fn unknown_torch_rebuild_is_an_error() {
        let bytes = b"\x80\x02ctorch._utils\n_rebuild_qtensor\n)R.";
        assert!(matches!(
            plain(bytes).unwrap_err(),
            PickleError::UnsupportedType(name) if name == "torch._utils._rebuild_qtensor"
        ));
    }

    #[test]
    fn persistent_id_without_source_is_an_error() {
        let bytes = b"\x80\x02U\x07storageQ.";
        assert!(matches!(
            plain(bytes).unwrap_err(),
            PickleError::NoDataSource
        ));
    }

    #[test]
    fn int_dict_keys_are_accepted_everywhere() {
        // SETITEM with an int key, then DICT with an int key.
        assert!(
            matches!(plain(b"\x80\x02}K\x01K\x02s.").unwrap(), Object::Dict(d) if d.contains_key("1"))
        );
        assert!(
            matches!(plain(b"\x80\x02(K\x03K\x04d.").unwrap(), Object::Dict(d) if d.contains_key("3"))
        );
    }

    #[test]
    fn stride_rejects_negative_values() {
        let negative_stride = Object::Tuple(vec![Object::Int(-1)]);
        assert!(matches!(
            parse_dims(&negative_stride, "stride"),
            Err(PickleError::InvalidData(msg)) if msg == "stride must be non-negative, got -1"
        ));
    }

    #[test]
    fn storage_extent_rejects_rank_mismatch() {
        assert!(matches!(
            tensor_storage_extent(&[2, 3], &[3], 0),
            Err(PickleError::InvalidData(msg))
                if msg == "Tensor stride rank 1 does not match shape rank 2"
        ));
    }

    #[test]
    fn storage_extent_rejects_overflow() {
        assert!(matches!(
            tensor_storage_extent(&[usize::MAX], &[2], 0),
            Err(PickleError::InvalidData(msg))
                if msg == "Tensor stride calculation overflows usize"
        ));
    }

    #[test]
    fn view_beyond_declared_storage_fails_at_parse_time() {
        let source = fixture_source();
        let err = rebuild(
            rebuild_args("FloatStorage", "0", 32, 30, &[2, 3], &[3, 1], &source),
            source,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            PickleError::InvalidData(msg) if msg.contains("needs 36 elements, but storage '0' declares 32")
        ));
    }

    #[test]
    fn declared_storage_larger_than_file_fails_at_read_time() {
        let source = fixture_source();
        let tensor = rebuild(
            rebuild_args("FloatStorage", "0", 64, 40, &[2, 3], &[3, 1], &source),
            source,
        )
        .unwrap();
        let err = bridge::into_data(tensor).unwrap_err();
        assert!(matches!(
            err,
            PackError::ValidationError(msg) if msg.contains("requires 46 elements from storage '0', but only 32 are available")
        ));
    }

    #[test]
    fn broadcast_view_above_limit_is_refused() {
        let source = fixture_source();
        let err = rebuild(
            rebuild_args(
                "FloatStorage",
                "0",
                32,
                0,
                &[1 << 21, 1 << 21],
                &[0, 0],
                &source,
            ),
            source,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            PickleError::InvalidData(msg) if msg.contains("limit for broadcast views")
        ));
    }

    #[test]
    fn small_broadcast_view_loads() {
        let source = fixture_source();
        let tensor = rebuild(
            rebuild_args("FloatStorage", "0", 32, 1, &[2, 3], &[0, 1], &source),
            source,
        )
        .unwrap();
        let data = bridge::into_data(tensor).unwrap();
        assert_eq!(
            data.as_slice::<f32>().unwrap(),
            &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn legacy_rebuild_tensor_loads_contiguous_stride() {
        let source = fixture_source();
        let tensor = rebuild(
            rebuild_args("FloatStorage", "0", 32, 5, &[2, 3], &[3, 1], &source),
            source,
        )
        .unwrap();
        let data = bridge::into_data(tensor).unwrap();
        assert_eq!(
            data.as_slice::<f32>().unwrap(),
            &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        );
    }

    #[test]
    fn legacy_rebuild_tensor_loads_permuted_stride() {
        let source = fixture_source();
        let tensor = rebuild(
            rebuild_args("FloatStorage", "0", 32, 5, &[2, 4, 3], &[12, 1, 4], &source),
            source,
        )
        .unwrap();
        let data = bridge::into_data(tensor).unwrap();
        assert_eq!(
            data.as_slice::<f32>().unwrap(),
            &[
                5.0, 9.0, 13.0, 6.0, 10.0, 14.0, 7.0, 11.0, 15.0, 8.0, 12.0, 16.0, 17.0, 21.0,
                25.0, 18.0, 22.0, 26.0, 19.0, 23.0, 27.0, 20.0, 24.0, 28.0,
            ]
        );
    }

    #[test]
    fn legacy_rebuild_tensor_loads_scalar_at_offset() {
        let source = fixture_source();
        let tensor = rebuild(
            rebuild_args("FloatStorage", "0", 32, 5, &[], &[], &source),
            source,
        )
        .unwrap();
        let data = bridge::into_data(tensor).unwrap();
        assert_eq!(data.as_slice::<f32>().unwrap(), &[5.0]);
    }

    #[test]
    fn storage_view_offset_is_applied() {
        let source = fixture_source();
        let mut pid = match storage_pid("FloatStorage", "0", 32) {
            Object::Tuple(fields) => fields,
            _ => unreachable!(),
        };
        pid.push(Object::Tuple(vec![
            Object::String("7".to_string()),
            Object::Int(4),
            Object::Int(10),
        ]));
        let storage = resolve_storage_id(&pid, &source).unwrap();
        assert_eq!(storage.view_offset, 4);
        let tensor = build_tensor(storage, DType::F32, 1, vec![3], vec![1], source).unwrap();
        let data = bridge::into_data(tensor).unwrap();
        assert_eq!(data.as_slice::<f32>().unwrap(), &[5.0, 6.0, 7.0]);
    }

    #[test]
    fn rebuild_v3_reads_untyped_storage_with_explicit_dtype() {
        let source = fixture_source();
        let pid = [
            Object::String("storage".to_string()),
            Object::Class {
                module_name: "torch".to_string(),
                name: "UntypedStorage".to_string(),
            },
            Object::String("0".to_string()),
            Object::String("cpu".to_string()),
            Object::Int(128),
        ];
        let storage = resolve_storage_id(&pid, &source).unwrap();
        assert_eq!(storage.dtype, None);
        let args = Object::Tuple(vec![
            Object::Storage(storage),
            Object::Int(0),
            Object::Tuple(vec![Object::Int(2)]),
            Object::Tuple(vec![Object::Int(1)]),
            Object::Bool(false),
            Object::None,
            Object::Class {
                module_name: "torch".to_string(),
                name: "uint32".to_string(),
            },
        ]);
        let ids = PersistentIds::Storages(source);
        let Object::Tensor(tensor) = rebuild_tensor(args, &ids, TensorRebuild::V3).unwrap() else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.dtype, DType::U32);
        let data = bridge::into_data(tensor).unwrap();
        // The storage holds f32 0.0 and 1.0; read as u32 bit patterns.
        assert_eq!(data.as_slice::<u32>().unwrap(), &[0, 1.0f32.to_bits()]);
    }

    #[test]
    fn missing_storage_returns_contextual_error() {
        let source = fixture_source();
        let tensor = rebuild(
            rebuild_args("FloatStorage", "missing", 6, 0, &[2, 3], &[3, 1], &source),
            source,
        )
        .unwrap();
        let err = bridge::into_data(tensor).unwrap_err();
        assert!(matches!(
            err,
            PackError::ValidationError(msg)
                if msg.contains("Failed to read storage 'missing' for tensor with shape [2, 3]")
        ));
    }
}
