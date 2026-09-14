//! Just enough pickle support to read PyTorch checkpoints.
//!
//! This implementation started from the candle project's pickle loader and has since been
//! reworked around lazy tensor data, every pickle protocol a Python 3 writer emits, and a
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
use burn_pack::{Error as PackError, MAX_TENSOR_SIZE, Tensor as PackTensor};
use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::io::{self, BufRead};
use std::sync::Arc;

/// Cap on the up-front allocation for a length-prefixed string, bytes or long value;
/// longer values grow as their bytes arrive.
const STRING_PREALLOC_BOUND: usize = 1 << 16;

/// Error type for pickle operations.
#[derive(Debug)]
#[non_exhaustive]
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
                "Pickle references tensor storages but no tensor data is available. Tensors can only be loaded from a PyTorch checkpoint file, not a plain pickle."
            ),
        }
    }
}

impl std::error::Error for PickleError {}

type Result<T> = std::result::Result<T, PickleError>;

// https://github.com/python/cpython/blob/main/Lib/pickletools.py
#[repr(u8)]
#[derive(Debug, Eq, PartialEq, Clone, Copy)]
#[non_exhaustive]
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
        Ok(match value {
            b'I' => Int,
            b'L' => Long,
            b'F' => Float,
            b'V' => Unicode,
            b'g' => Get,
            b'p' => Put,
            b'P' => PersId,
            b'c' => Global,
            b'(' => Mark,
            b'.' => Stop,
            b'0' => Pop,
            b'1' => PopMark,
            b'2' => Dup,
            b'N' => None,
            b'R' => Reduce,
            b'b' => Build,
            b'd' => Dict,
            b'l' => List,
            b't' => Tuple,
            b's' => SetItem,
            b'a' => Append,
            b'J' => BinInt,
            b'K' => BinInt1,
            b'M' => BinInt2,
            b'G' => BinFloat,
            b'T' => BinString,
            b'U' => ShortBinString,
            b'X' => BinUnicode,
            b')' => EmptyTuple,
            b']' => EmptyList,
            b'}' => EmptyDict,
            b'e' => Appends,
            b'u' => SetItems,
            b'h' => BinGet,
            b'j' => LongBinGet,
            b'q' => BinPut,
            b'r' => LongBinPut,
            b'Q' => BinPersId,
            0x80 => Proto,
            0x81 => NewObj,
            0x82 => Ext1,
            0x83 => Ext2,
            0x84 => Ext4,
            0x85 => Tuple1,
            0x86 => Tuple2,
            0x87 => Tuple3,
            0x88 => NewTrue,
            0x89 => NewFalse,
            0x8a => Long1,
            0x8b => Long4,
            b'B' => BinBytes,
            b'C' => ShortBinBytes,
            0x8c => ShortBinUnicode,
            0x8d => BinUnicode8,
            0x8e => BinBytes8,
            0x8f => EmptySet,
            0x90 => AddItems,
            0x91 => FrozenSet,
            0x92 => NewObjEx,
            0x93 => StackGlobal,
            0x94 => Memoize,
            0x95 => Frame,
            0x96 => ByteArray8,
            0x97 => NextBuffer,
            0x98 => ReadonlyBuffer,
            other => return Err(other),
        })
    }
}

/// A storage referenced by a persistent id, before any tensor is built on it.
#[derive(Debug, Clone)]
pub struct StorageRef {
    /// The container that holds the bytes.
    pub(crate) source: Arc<StorageSource>,
    /// Key of the storage within its container (`"0"`, `"1"`, ...).
    pub(crate) key: String,
    /// Element type of a typed storage; `None` for `torch.UntypedStorage`.
    pub(crate) dtype: Option<DType>,
    /// Bytes a tensor may reach in this storage: its declared size, or the end of the view
    /// when the pickle describes one.
    pub(crate) reachable_bytes: usize,
    /// Element offset into the root storage when the persistent id describes a view.
    /// Modern `torch.save` always writes `None` there; only early legacy files carry one.
    pub(crate) view_offset: usize,
}

#[derive(Debug, Clone)]
pub enum Object {
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
    /// A Python object this reader does not interpret: an unknown `REDUCE`, `NEWOBJ`,
    /// `NEWOBJ_EX` or `BUILD`. Nothing downstream looks inside one, so nothing is kept.
    Opaque,
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
            _ => 1,
        }
    }

    /// Bytes of string and bytes payload in the object tree, which a clone duplicates.
    fn payload_bytes(&self) -> usize {
        match self {
            Object::String(s) => s.len(),
            Object::Bytes(b) => b.len(),
            Object::Tuple(v) | Object::List(v) => v.iter().map(|o| o.payload_bytes()).sum(),
            Object::Dict(m) => m.values().map(|o| o.payload_bytes()).sum(),
            _ => 0,
        }
    }

    /// Whether the object carries a storage or tensor anywhere inside it.
    fn holds_tensor_data(&self) -> bool {
        match self {
            Object::Storage(_) | Object::Tensor(_) => true,
            Object::Tuple(v) | Object::List(v) => v.iter().any(Object::holds_tensor_data),
            Object::Dict(m) => m.values().any(Object::holds_tensor_data),
            _ => false,
        }
    }

    fn type_name(&self) -> &'static str {
        match self {
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
            Object::Opaque => "object",
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

/// Read a non-negative Python int as a `usize`.
pub(crate) fn non_negative(value: &Object, what: &str) -> Result<usize> {
    match value {
        Object::Int(i) => usize::try_from(*i)
            .map_err(|_| PickleError::InvalidData(format!("{what} must be non-negative, got {i}"))),
        other => Err(PickleError::InvalidData(format!(
            "{what} must be an int, got {}",
            other.type_name()
        ))),
    }
}

/// Read a Python str or int as a string key. PyTorch uses both for the same purpose.
pub(crate) fn key_string(value: &Object, what: &str) -> Result<String> {
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
fn resolve_storage_id(pid: &[Object], source: &Arc<StorageSource>) -> Result<StorageRef> {
    let [tag, storage_type, key, _location, numel, view @ ..] = pid else {
        return Err(PickleError::InvalidData(format!(
            "storage persistent id has {} fields, expected at least 5",
            pid.len()
        )));
    };
    if !matches!(tag, Object::String(tag) if tag == "storage") {
        return Err(PickleError::InvalidData(format!(
            "persistent id tag must be 'storage', got {}",
            tag.type_name()
        )));
    }

    let type_name = match storage_type {
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

    let key = key_string(key, "storage key")?;
    let numel = non_negative(numel, "storage element count")?;
    let element_size = dtype.map_or(1, |dtype| dtype.size());
    let byte_len = numel.checked_mul(element_size).ok_or_else(|| {
        PickleError::InvalidData(format!("storage '{key}' byte length overflows usize"))
    })?;

    // Early legacy files may describe a view of a root storage: (view_key, offset, view_size).
    // A tensor may reach only as far as the view does, which is where it ends rather than
    // where the root storage does.
    let (view_offset, reachable) = match view.first() {
        None | Some(Object::None) => (0, numel),
        Some(Object::Tuple(view)) if view.len() == 3 => {
            if dtype.is_none() {
                return Err(PickleError::InvalidData(format!(
                    "storage '{key}' is untyped but carries view metadata"
                )));
            }
            let offset = non_negative(&view[1], "storage view offset")?;
            let size = non_negative(&view[2], "storage view size")?;
            let end = offset
                .checked_add(size)
                .filter(|&end| end <= numel)
                .ok_or_else(|| {
                    PickleError::InvalidData(format!(
                        "storage '{key}' has a view of {size} elements at offset {offset}, past its {numel} elements"
                    ))
                })?;
            (offset, end)
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
        source: source.clone(),
        key,
        dtype,
        // `reachable` never exceeds `numel`, whose byte length is checked above.
        reachable_bytes: reachable * element_size,
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
/// is kept as an opaque object (see [`opaque`] for the one exception) so a checkpoint
/// carrying, say, numpy scalars or a device in its metadata still loads its tensors.
fn reduce(callable: Object, args: Object) -> Result<Object> {
    let (module_name, name) = match &callable {
        Object::Class { module_name, name } => (module_name.as_str(), name.as_str()),
        _ => return opaque(&callable, &args),
    };

    match (module_name, name) {
        ("collections", "OrderedDict") => ordered_dict(args),
        ("torch._utils", "_rebuild_tensor") => rebuild_tensor(args, TensorRebuild::Legacy),
        ("torch._utils", "_rebuild_tensor_v2") => rebuild_tensor(args, TensorRebuild::V2),
        ("torch._utils", "_rebuild_tensor_v3") => rebuild_tensor(args, TensorRebuild::V3),
        // _rebuild_parameter(data, requires_grad, backward_hooks[, state]): only the data
        // is kept, so state holding tensors would go missing and is refused instead.
        ("torch._utils", "_rebuild_parameter" | "_rebuild_parameter_with_state") => match args {
            Object::Tuple(mut fields) if !fields.is_empty() => {
                if fields[1..].iter().any(Object::holds_tensor_data) {
                    return Err(PickleError::UnsupportedType(format!(
                        "{name} with tensor-bearing state"
                    )));
                }
                Ok(fields.swap_remove(0))
            }
            other => Err(PickleError::InvalidData(format!(
                "{name}: expected a non-empty argument tuple, got {}",
                other.type_name()
            ))),
        },
        // _rebuild_from_type_v2(func, new_type, args, state): the tensor is func(*args).
        // The subclass itself is not represented: a weight loader wants the data. State
        // that holds tensors would be dropped, so that case is refused.
        ("torch._tensor", "_rebuild_from_type_v2") => match args {
            Object::Tuple(mut fields) if fields.len() >= 3 => {
                if fields.get(3).is_some_and(Object::holds_tensor_data) {
                    return Err(PickleError::UnsupportedType(
                        "tensor subclass with tensor-bearing state".to_string(),
                    ));
                }
                let inner_args = fields.swap_remove(2);
                let func = fields.swap_remove(0);
                reduce(func, inner_args)
            }
            other => Err(PickleError::InvalidData(format!(
                "_rebuild_from_type_v2: expected at least 3 arguments, got {}",
                other.type_name()
            ))),
        },
        _ => opaque(&callable, &args),
    }
}

/// Leave an uninterpreted call opaque, unless it consumed tensor data.
///
/// A call whose arguments hold a storage or tensor (sparse, quantized and nested tensors,
/// or a tensor subclass) is a tensor this reader cannot represent. That is worth an error
/// rather than an entry that silently goes missing. The same check guards `BUILD` state,
/// so a pickled module whose attributes hold tensors (a full model save) is refused rather
/// than loaded empty.
fn opaque(callable: &Object, args: &Object) -> Result<Object> {
    if callable.holds_tensor_data() || args.holds_tensor_data() {
        let name = match callable {
            Object::Class { module_name, name } => format!("{module_name}.{name}"),
            other => other.type_name().to_string(),
        };
        return Err(PickleError::UnsupportedType(name));
    }
    Ok(Object::Opaque)
}

/// `OrderedDict()` or `OrderedDict([(key, value), ...])`.
fn ordered_dict(args: Object) -> Result<Object> {
    let items = match args {
        Object::Tuple(fields) if fields.is_empty() => Vec::new(),
        Object::Tuple(mut fields) => match fields.swap_remove(0) {
            Object::List(items) | Object::Tuple(items) => items,
            other => {
                return Err(PickleError::InvalidData(format!(
                    "OrderedDict argument must be a list of pairs, got {}",
                    other.type_name()
                )));
            }
        },
        other => {
            return Err(PickleError::InvalidData(format!(
                "OrderedDict arguments must be a tuple, got {}",
                other.type_name()
            )));
        }
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
    /// `_rebuild_tensor(storage, storage_offset, size, stride)`, PyTorch before 0.4.
    Legacy,
    /// `_rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, hooks[, metadata])`.
    V2,
    /// `_rebuild_tensor_v3(storage, storage_offset, size, stride, requires_grad, hooks, dtype[, metadata])`,
    /// used for element types without a typed storage class (uint16/32/64, float8 and
    /// others). Only the unsigned ints have a burn dtype; the rest are unsupported.
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

fn rebuild_tensor(args: Object, kind: TensorRebuild) -> Result<Object> {
    let fn_name = kind.name();
    let mut fields = match args {
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

    let storage = match std::mem::replace(&mut fields[0], Object::None) {
        Object::Storage(storage) => storage,
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

    build_tensor(storage, dtype, storage_offset, shape, stride).map(Object::Tensor)
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
/// `data` begins at the tensor's storage offset, and the caller has validated that every
/// index the view touches lies within it.
fn gather_strided(
    data: &[u8],
    shape: &[usize],
    stride: &[usize],
    element_size: usize,
    byte_len: usize,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(byte_len);
    let num_elements = byte_len / element_size;
    for linear_index in 0..num_elements {
        let mut remaining = linear_index;
        let mut storage_index = 0;
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

/// Build a tensor that reads its bytes from the storage's container on demand.
///
/// Everything derivable from the metadata is validated here, so a file whose declarations are
/// inconsistent fails at parse time; the data itself is validated when it is read.
pub(crate) fn build_tensor(
    storage: StorageRef,
    dtype: DType,
    storage_offset: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,
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

    let declared_elements = storage.reachable_bytes / element_size;
    if extent > declared_elements {
        return Err(PickleError::InvalidData(format!(
            "Tensor with shape {shape:?} and stride {stride:?} at offset {storage_offset} needs {extent} elements, but storage '{}' declares {declared_elements}",
            storage.key
        )));
    }
    // A view can declare far more logical elements than its storage holds (an `expand` has
    // stride 0), so the file gives no bound on what a tensor materializes. Apply the
    // burn-pack ceiling that its own reader already enforces.
    if byte_len > MAX_TENSOR_SIZE {
        return Err(PickleError::InvalidData(format!(
            "Tensor with shape {shape:?} would materialize {byte_len} bytes, above the {MAX_TENSOR_SIZE} byte limit"
        )));
    }

    let StorageRef { source, key, .. } = storage;
    let provider_shape = shape.clone();
    // Read only the window the view can touch: a storage that reaches further, before the
    // tensor's offset or past its extent, is not paid for.
    let window_start = storage_offset.saturating_mul(element_size);
    let window_len = (extent - storage_offset).saturating_mul(element_size);
    let provider = move || -> std::result::Result<TensorData, PackError> {
        let shape = &provider_shape;
        let (mut data, skipped) = source.read(&key, window_start, window_len).map_err(|err| {
            PackError::ValidationError(format!(
                "Failed to read storage '{key}' for tensor with shape {shape:?}: {err}"
            ))
        })?;

        // `data` starts at the tensor's own offset, so every index below is relative to it.
        let available = (skipped + data.len()) / element_size;
        if extent > available {
            return Err(PackError::ValidationError(format!(
                "Tensor with shape {shape:?} requires {extent} elements from storage '{key}', but only {available} are available"
            )));
        }

        let mut bytes = if is_contiguous_stride(shape, &stride) {
            data.truncate(byte_len);
            data
        } else {
            gather_strided(&data, shape, &stride, element_size, byte_len)
        };

        to_native_endian(&mut bytes, element_size);
        if matches!(dtype, DType::Bool(_)) {
            // A well-formed file holds only 0 or 1; any other byte would be an invalid
            // `bool`, so normalize before reinterpreting.
            for byte in &mut bytes {
                *byte = u8::from(*byte != 0);
            }
        }

        Ok(TensorData::from_bytes_vec(bytes, shape.clone(), dtype))
    };

    // The tensor's name is a path through the pickle's dicts, assembled by the caller that
    // walks them. PyTorch carries no parameter identity, so `param_id` stays `None`.
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

/// Total nodes that may be copied through memo lookups and `DUP` before the pickle is
/// rejected as a memo bomb (each such opcode deep-copies an object).
const MAX_COPIED_NODES: usize = 1_000_000;

/// Total string and bytes payload those copies may duplicate. A pickle can memoize one
/// large value and fetch it many times, so node counts alone do not bound memory.
const MAX_COPIED_BYTES: usize = 64 << 20;

/// Deepest container nesting accepted.
///
/// Objects are walked recursively after parsing (extraction, conversion, drop), so a pickle
/// nested deeper than the thread's stack would overflow it instead of failing. Python's own
/// recursion limit is the same order of magnitude.
const MAX_NESTING_DEPTH: u32 = 1000;

/// A stack or memo slot: the object plus how deeply it nests containers (0 for a scalar).
#[derive(Clone)]
struct Entry {
    object: Object,
    depth: u32,
}

struct Unpickler<'a> {
    stack: Vec<Entry>,
    /// Stack heights at each open `MARK`, innermost last, as in CPython.
    marks: Vec<usize>,
    memo: HashMap<u32, Entry>,
    copied_nodes: usize,
    copied_bytes: usize,
    ids: &'a PersistentIds,
}

impl<'a> Unpickler<'a> {
    fn new(ids: &'a PersistentIds) -> Self {
        Self {
            stack: Vec::new(),
            marks: Vec::new(),
            memo: HashMap::new(),
            copied_nodes: 0,
            copied_bytes: 0,
            ids,
        }
    }

    fn push_scalar(&mut self, object: Object) {
        self.stack.push(Entry { object, depth: 0 });
    }

    fn push(&mut self, object: Object, depth: u32) -> Result<()> {
        if depth > MAX_NESTING_DEPTH {
            return Err(PickleError::InvalidData(format!(
                "pickle nesting exceeds {MAX_NESTING_DEPTH} levels"
            )));
        }
        self.stack.push(Entry { object, depth });
        Ok(())
    }

    /// Whether the innermost `MARK` sits on top of the stack, with no value above it.
    fn mark_on_top(&self) -> bool {
        self.marks.last() == Some(&self.stack.len())
    }

    fn pop(&mut self) -> Result<Entry> {
        if self.mark_on_top() {
            return Err(PickleError::InvalidData(
                "expected a value on the stack, found MARK".to_string(),
            ));
        }
        self.stack.pop().ok_or(PickleError::StackUnderflow)
    }

    fn pop_object(&mut self) -> Result<Object> {
        self.pop().map(|entry| entry.object)
    }

    fn top(&self) -> Result<&Entry> {
        if self.mark_on_top() {
            return Err(PickleError::InvalidData(
                "expected a value on the stack, found MARK".to_string(),
            ));
        }
        self.stack.last().ok_or(PickleError::StackUnderflow)
    }

    fn top_mut(&mut self) -> Result<&mut Entry> {
        if self.mark_on_top() {
            return Err(PickleError::InvalidData(
                "expected a value on the stack, found MARK".to_string(),
            ));
        }
        self.stack.last_mut().ok_or(PickleError::StackUnderflow)
    }

    /// Pop everything above the innermost `MARK`, and the mark itself.
    ///
    /// Returns the objects with the deepest nesting among them.
    fn pop_to_mark(&mut self) -> Result<(Vec<Object>, u32)> {
        let mark = self
            .marks
            .pop()
            .ok_or_else(|| PickleError::InvalidData("MARK not found on stack".to_string()))?;
        let entries = self.stack.split_off(mark);
        let depth = entries.iter().map(|entry| entry.depth).max().unwrap_or(0);
        Ok((
            entries.into_iter().map(|entry| entry.object).collect(),
            depth,
        ))
    }

    /// Account for a deep copy of `object` against the copy budgets.
    fn charge_copy(&mut self, object: &Object) -> Result<()> {
        self.copied_nodes += object.node_count();
        self.copied_bytes += object.payload_bytes();
        if self.copied_nodes > MAX_COPIED_NODES || self.copied_bytes > MAX_COPIED_BYTES {
            return Err(PickleError::InvalidData(format!(
                "Pickle memo bomb detected: copies exceeded {MAX_COPIED_NODES} nodes or {MAX_COPIED_BYTES} bytes"
            )));
        }
        Ok(())
    }

    fn memo_get(&mut self, idx: u32) -> Result<Entry> {
        let entry = self
            .memo
            .get(&idx)
            .ok_or(PickleError::MemoNotFound(idx))?
            .clone();
        self.charge_copy(&entry.object)?;
        Ok(entry)
    }

    fn memo_put(&mut self, idx: u32) -> Result<()> {
        let entry = self.top()?.clone();
        self.charge_copy(&entry.object)?;
        self.memo.insert(idx, entry);
        Ok(())
    }

    /// Extend the list on top of the stack.
    fn append(&mut self, items: Vec<Object>, items_depth: u32, op: OpCode) -> Result<()> {
        let top = self.top_mut()?;
        let Object::List(list) = &mut top.object else {
            return Err(PickleError::UnexpectedOpCode(op));
        };
        list.extend(items);
        let depth = top.depth.max(items_depth + 1);
        top.depth = depth;
        if depth > MAX_NESTING_DEPTH {
            return Err(PickleError::InvalidData(format!(
                "pickle nesting exceeds {MAX_NESTING_DEPTH} levels"
            )));
        }
        Ok(())
    }

    /// Insert key/value pairs into the dict on top of the stack.
    fn set_items(&mut self, items: Vec<Object>, items_depth: u32, op: OpCode) -> Result<()> {
        let top = self.top_mut()?;
        let Object::Dict(dict) = &mut top.object else {
            return Err(PickleError::UnexpectedOpCode(op));
        };
        insert_pairs(dict, items)?;
        let depth = top.depth.max(items_depth + 1);
        top.depth = depth;
        if depth > MAX_NESTING_DEPTH {
            return Err(PickleError::InvalidData(format!(
                "pickle nesting exceeds {MAX_NESTING_DEPTH} levels"
            )));
        }
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
                OpCode::None => self.push_scalar(Object::None),
                OpCode::NewTrue => self.push_scalar(Object::Bool(true)),
                OpCode::NewFalse => self.push_scalar(Object::Bool(false)),
                OpCode::Int => {
                    let text = read_line(r)?;
                    // Protocol 0 spells booleans as INT 00 / INT 01.
                    let value = match text.as_str() {
                        "00" => Object::Bool(false),
                        "01" => Object::Bool(true),
                        _ => Object::Int(parse_int(&text)?),
                    };
                    self.push_scalar(value);
                }
                OpCode::Long => {
                    let text = read_line(r)?;
                    let text = text.strip_suffix('L').unwrap_or(&text);
                    self.push_scalar(parse_long(text)?);
                }
                OpCode::Float => {
                    let text = read_line(r)?;
                    let value = text.parse::<f64>().map_err(|e| {
                        PickleError::InvalidData(format!("Invalid FLOAT value '{text}': {e}"))
                    })?;
                    self.push_scalar(Object::Float(value));
                }
                OpCode::Unicode => {
                    let text = read_raw_unicode_escape_line(r)?;
                    self.push_scalar(Object::String(text));
                }
                OpCode::BinInt => {
                    let v = r.read_i32::<LittleEndian>()?;
                    self.push_scalar(Object::Int(v as i64));
                }
                OpCode::BinInt1 => {
                    let v = r.read_u8()?;
                    self.push_scalar(Object::Int(v as i64));
                }
                OpCode::BinInt2 => {
                    let v = r.read_u16::<LittleEndian>()?;
                    self.push_scalar(Object::Int(v as i64));
                }
                OpCode::BinFloat => {
                    let v = r.read_f64::<BigEndian>()?;
                    self.push_scalar(Object::Float(v));
                }
                OpCode::Long1 => {
                    let len = r.read_u8()? as u64;
                    let bytes = read_exact_len(r, len, STRING_PREALLOC_BOUND)?;
                    self.push_scalar(long_object(&bytes));
                }
                OpCode::Long4 => {
                    let len = r.read_u32::<LittleEndian>()? as u64;
                    let bytes = read_exact_len(r, len, STRING_PREALLOC_BOUND)?;
                    self.push_scalar(long_object(&bytes));
                }

                // Strings and bytes
                OpCode::ShortBinUnicode => {
                    let len = r.read_u8()? as u64;
                    self.push_scalar(read_string(r, len)?);
                }
                OpCode::BinUnicode => {
                    let len = r.read_u32::<LittleEndian>()? as u64;
                    self.push_scalar(read_string(r, len)?);
                }
                OpCode::ShortBinString => {
                    let len = r.read_u8()? as u64;
                    self.push_scalar(read_py2_string(r, len)?);
                }
                OpCode::BinString => {
                    let len = r.read_u32::<LittleEndian>()? as u64;
                    self.push_scalar(read_py2_string(r, len)?);
                }
                OpCode::BinUnicode8 => {
                    let len = r.read_u64::<LittleEndian>()?;
                    self.push_scalar(read_string(r, len)?);
                }
                OpCode::ShortBinBytes => {
                    let len = r.read_u8()? as u64;
                    self.push_scalar(Object::Bytes(read_bytes(r, len)?));
                }
                OpCode::BinBytes => {
                    let len = r.read_u32::<LittleEndian>()? as u64;
                    self.push_scalar(Object::Bytes(read_bytes(r, len)?));
                }
                OpCode::BinBytes8 | OpCode::ByteArray8 => {
                    let len = r.read_u64::<LittleEndian>()?;
                    self.push_scalar(Object::Bytes(read_bytes(r, len)?));
                }

                // Containers
                OpCode::Mark => self.marks.push(self.stack.len()),
                OpCode::EmptyTuple => self.push_scalar(Object::Tuple(Vec::new())),
                OpCode::EmptyList | OpCode::EmptySet => self.push_scalar(Object::List(Vec::new())),
                OpCode::EmptyDict => self.push_scalar(Object::Dict(HashMap::new())),
                OpCode::Tuple => {
                    let (items, depth) = self.pop_to_mark()?;
                    self.push(Object::Tuple(items), depth + 1)?;
                }
                OpCode::Tuple1 => {
                    let a = self.pop()?;
                    self.push(Object::Tuple(vec![a.object]), a.depth + 1)?;
                }
                OpCode::Tuple2 => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    let depth = a.depth.max(b.depth) + 1;
                    self.push(Object::Tuple(vec![a.object, b.object]), depth)?;
                }
                OpCode::Tuple3 => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    let depth = a.depth.max(b.depth).max(c.depth) + 1;
                    self.push(Object::Tuple(vec![a.object, b.object, c.object]), depth)?;
                }
                OpCode::List | OpCode::FrozenSet => {
                    let (items, depth) = self.pop_to_mark()?;
                    self.push(Object::List(items), depth + 1)?;
                }
                OpCode::Append => {
                    let value = self.pop()?;
                    self.append(vec![value.object], value.depth, op)?;
                }
                OpCode::Appends | OpCode::AddItems => {
                    let (items, depth) = self.pop_to_mark()?;
                    self.append(items, depth, op)?;
                }
                OpCode::Dict => {
                    let (items, depth) = self.pop_to_mark()?;
                    let mut dict = HashMap::with_capacity(items.len() / 2);
                    insert_pairs(&mut dict, items)?;
                    self.push(Object::Dict(dict), depth + 1)?;
                }
                OpCode::SetItem => {
                    let value = self.pop()?;
                    let key = self.pop()?;
                    let depth = key.depth.max(value.depth);
                    self.set_items(vec![key.object, value.object], depth, op)?;
                }
                OpCode::SetItems => {
                    let (items, depth) = self.pop_to_mark()?;
                    self.set_items(items, depth, op)?;
                }

                // Stack manipulation
                OpCode::Pop => {
                    // As in CPython, POP discards a bare MARK when that is what is on top.
                    if self.mark_on_top() {
                        self.marks.pop();
                    } else {
                        self.pop()?;
                    }
                }
                OpCode::PopMark => {
                    self.pop_to_mark()?;
                }
                OpCode::Dup => {
                    let top = self.top()?.clone();
                    self.charge_copy(&top.object)?;
                    self.stack.push(top);
                }

                // Memo
                OpCode::Get => {
                    let idx = parse_index(&read_line(r)?)?;
                    let entry = self.memo_get(idx)?;
                    self.stack.push(entry);
                }
                OpCode::BinGet => {
                    let idx = r.read_u8()? as u32;
                    let entry = self.memo_get(idx)?;
                    self.stack.push(entry);
                }
                OpCode::LongBinGet => {
                    let idx = r.read_u32::<LittleEndian>()?;
                    let entry = self.memo_get(idx)?;
                    self.stack.push(entry);
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
                    self.push_scalar(Object::Class { module_name, name });
                }
                OpCode::StackGlobal => {
                    let name = self.pop_object()?;
                    let module_name = self.pop_object()?;
                    match (module_name, name) {
                        (Object::String(module_name), Object::String(name)) => {
                            self.push_scalar(Object::Class { module_name, name })
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
                    self.push_scalar(obj);
                }
                OpCode::BinPersId => {
                    let pid = self.pop_object()?;
                    let obj = resolve_persistent_id(pid, self.ids)?;
                    self.push_scalar(obj);
                }
                OpCode::Reduce => {
                    let args = self.pop()?;
                    let callable = self.pop_object()?;
                    // A call's result nests no deeper than its arguments did.
                    let obj = reduce(callable, args.object)?;
                    self.push(obj, args.depth)?;
                }
                OpCode::NewObj => {
                    let args = self.pop_object()?;
                    let cls = self.pop_object()?;
                    let obj = opaque(&cls, &args)?;
                    self.push_scalar(obj);
                }
                OpCode::NewObjEx => {
                    let kwargs = self.pop_object()?;
                    let args = self.pop_object()?;
                    let cls = self.pop_object()?;
                    opaque(&cls, &kwargs)?;
                    let obj = opaque(&cls, &args)?;
                    self.push_scalar(obj);
                }
                OpCode::Build => {
                    let state = self.pop()?;
                    let object = self.pop()?;
                    match (object.object, state.object) {
                        (Object::Dict(mut dict), Object::Dict(update)) => {
                            dict.extend(update);
                            self.push(Object::Dict(dict), object.depth.max(state.depth))?;
                        }
                        // A dict subclass with non-dict state: the items are what matter,
                        // but state holding tensor data must not vanish silently.
                        (dict @ Object::Dict(_), state) => {
                            if state.holds_tensor_data() {
                                return Err(PickleError::UnsupportedType(
                                    "dict with tensor-bearing state".to_string(),
                                ));
                            }
                            self.push(dict, object.depth)?;
                        }
                        (object, state) => {
                            let obj = opaque(&object, &state)?;
                            self.push_scalar(obj);
                        }
                    }
                }

                OpCode::Ext1
                | OpCode::Ext2
                | OpCode::Ext4
                | OpCode::NextBuffer
                | OpCode::ReadonlyBuffer => return Err(PickleError::UnsupportedOpCode(op)),
            }
        }

        self.pop_object()
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

/// Read a protocol 0 `UNICODE` payload: Latin-1 bytes with `\uXXXX` and `\UXXXXXXXX`
/// escapes for everything else, Python's `raw-unicode-escape` codec.
fn read_raw_unicode_escape_line<R: BufRead>(r: &mut R) -> Result<String> {
    let mut data = Vec::with_capacity(32);
    r.read_until(b'\n', &mut data)?;
    if data.pop() != Some(b'\n') {
        return Err(PickleError::Io(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "unterminated text field",
        )));
    }

    let mut text = String::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        let byte = data[i];
        let escape_len = match (byte, data.get(i + 1)) {
            (b'\\', Some(b'u')) => Some(4),
            (b'\\', Some(b'U')) => Some(8),
            _ => None,
        };
        match escape_len {
            Some(digits) => {
                let hex = data
                    .get(i + 2..i + 2 + digits)
                    .and_then(|hex| std::str::from_utf8(hex).ok())
                    .and_then(|hex| u32::from_str_radix(hex, 16).ok())
                    .and_then(char::from_u32)
                    .ok_or_else(|| {
                        PickleError::InvalidData("Invalid raw-unicode-escape sequence".to_string())
                    })?;
                text.push(hex);
                i += 2 + digits;
            }
            None => {
                text.push(char::from(byte));
                i += 1;
            }
        }
    }
    Ok(text)
}

fn read_bytes<R: BufRead>(r: &mut R, len: u64) -> Result<Vec<u8>> {
    Ok(read_exact_len(r, len, STRING_PREALLOC_BOUND)?)
}

fn read_string<R: BufRead>(r: &mut R, len: u64) -> Result<Object> {
    let s = String::from_utf8(read_bytes(r, len)?)
        .map_err(|e| PickleError::InvalidData(format!("Invalid UTF-8: {e}")))?;
    Ok(Object::String(s))
}

/// A Python 2 `str` is bytes that usually hold text, but Python 2 pickles of numpy arrays
/// put raw array data in them. Text is kept as a string and anything else as bytes.
fn read_py2_string<R: BufRead>(r: &mut R, len: u64) -> Result<Object> {
    Ok(match String::from_utf8(read_bytes(r, len)?) {
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

/// A Python long, kept opaque when it does not fit in an `i64`.
///
/// Python integers are unbounded and a checkpoint may carry a large one beside its
/// weights, such as a seed of `2**64`. No int this reader acts on (a shape, an offset, an
/// element count) can be that large, so an out-of-range value is metadata it never looks
/// at, and refusing the file over it would refuse an otherwise loadable checkpoint.
fn long_object(bytes: &[u8]) -> Object {
    int_from_le_bytes(bytes).map_or(Object::Opaque, Object::Int)
}

/// Protocol 0 writes a Python long in decimal. Out-of-range values stay opaque, as in
/// [`long_object`]; anything that is not an integer at all is an error.
fn parse_long(text: &str) -> Result<Object> {
    use std::num::IntErrorKind;

    match text.parse::<i64>() {
        Ok(value) => Ok(Object::Int(value)),
        Err(e)
            if matches!(
                e.kind(),
                IntErrorKind::PosOverflow | IntErrorKind::NegOverflow
            ) =>
        {
            Ok(Object::Opaque)
        }
        Err(e) => Err(PickleError::InvalidData(format!(
            "Invalid integer '{text}': {e}"
        ))),
    }
}

/// Decode a two's-complement little-endian integer, or `None` if it needs more than 64 bits.
fn int_from_le_bytes(bytes: &[u8]) -> Option<i64> {
    let negative = bytes.last().is_some_and(|b| b & 0x80 != 0);
    let fill = if negative { 0xff } else { 0x00 };
    // Beyond 8 bytes, every extra byte must be sign fill, and the sign bit of byte 7 must
    // agree with it: otherwise the value lies outside the i64 range.
    let fits = bytes.len() <= 8
        || (bytes[8..].iter().all(|&b| b == fill) && (bytes[7] & 0x80 != 0) == negative);
    if !fits {
        return None;
    }
    let mut buf = [fill; 8];
    let len = bytes.len().min(8);
    buf[..len].copy_from_slice(&bytes[..len]);
    Some(i64::from_le_bytes(buf))
}

fn dict_key(key: Object) -> Result<String> {
    match key {
        Object::String(s) => Ok(s),
        other => key_string(&other, "dict key"),
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

/// Walk a parsed dict, collecting each tensor found under a nested dict path.
///
/// Only dicts are descended into: a state_dict is one, and a tensor reached any other way is
/// ignored. A tensor is built without a name, its path being assembled only here, so this is
/// also where each one gets its final identity.
pub(crate) fn extract_tensors(dict: HashMap<String, Object>) -> HashMap<String, PackTensor> {
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
    let mut tensors = HashMap::new();
    walk(Object::Dict(dict), &mut Vec::new(), &mut tensors);
    tensors
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn fixture_source() -> Arc<StorageSource> {
        let path = crate::pytorch::tests::reader::test_data_path("non_contiguous.pt");
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

    fn rebuild(args: Object) -> Result<PackTensor> {
        match rebuild_tensor(args, TensorRebuild::Legacy)? {
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
        assert!(matches!(err, PickleError::InvalidData(msg) if msg.contains("memo bomb")));
    }

    #[test]
    fn memo_copies_of_large_payloads_are_bounded() {
        // A 2 MiB BINBYTES memoized once and fetched 40 times would copy 80 MiB.
        let payload = vec![0xabu8; 2 << 20];
        let mut bytes = vec![0x80, 0x02, b'B'];
        bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&payload);
        bytes.extend_from_slice(b"q\x00(");
        for _ in 0..40 {
            bytes.extend_from_slice(b"h\x00");
        }
        bytes.extend_from_slice(b"t.");
        assert!(matches!(
            plain(&bytes).unwrap_err(),
            PickleError::InvalidData(msg) if msg.contains("memo bomb")
        ));
    }

    #[test]
    fn dup_doubling_is_bounded() {
        // DUP then TUPLE2 doubles the object tree every two bytes.
        let mut bytes = vec![0x80, 0x02, b'K', 0x01];
        for _ in 0..40 {
            bytes.extend_from_slice(b"2\x86");
        }
        bytes.push(b'.');
        assert!(matches!(
            plain(&bytes).unwrap_err(),
            PickleError::InvalidData(msg) if msg.contains("memo bomb")
        ));
    }

    #[test]
    fn unicode_opcode_decodes_raw_unicode_escape() {
        // pickle.dumps("\u00e9t\u00e9 \u6a21", protocol=0): Latin-1 bytes plus a \u escape.
        let bytes = b"V\xe9t\xe9 \\u6a21\np0\n.";
        assert!(matches!(
            plain(bytes).unwrap(),
            Object::String(s) if s == "\u{e9}t\u{e9} \u{6a21}"
        ));
        assert!(plain(b"V\\u12.\n.").is_err());
    }

    #[test]
    fn rebuild_from_type_v2_rejects_tensor_bearing_state() {
        let source = fixture_source();
        let callable = Object::Class {
            module_name: "torch._tensor".to_string(),
            name: "_rebuild_from_type_v2".to_string(),
        };
        let mut state = HashMap::new();
        state.insert(
            "extra".to_string(),
            rebuild_args("FloatStorage", "0", 32, 0, &[2], &[1], &source),
        );
        let args = Object::Tuple(vec![
            Object::Class {
                module_name: "torch._utils".to_string(),
                name: "_rebuild_tensor".to_string(),
            },
            Object::Class {
                module_name: "torch".to_string(),
                name: "Tensor".to_string(),
            },
            rebuild_args("FloatStorage", "0", 32, 5, &[2, 3], &[3, 1], &source),
            Object::Dict(state),
        ]);
        assert!(matches!(
            reduce(callable, args).unwrap_err(),
            PickleError::UnsupportedType(_)
        ));
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
            int_from_le_bytes(&[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01]).is_none()
        );
    }

    #[test]
    fn oversized_int_beside_weights_stays_opaque() {
        // {'seed': 2**64, 'n': 7} in protocol 2 (LONG1) and protocol 0 (LONG).
        for bytes in [
            b"\x80\x02}q\x00(X\x04\x00\x00\x00seedq\x01\x8a\t\x00\x00\x00\x00\x00\x00\x00\x00\x01X\x01\x00\x00\x00nq\x02K\x07u.".as_slice(),
            b"(dp0\nVseed\np1\nL18446744073709551616L\nsVn\np2\nI7\ns.".as_slice(),
        ] {
            let Object::Dict(dict) = plain(bytes).unwrap_or_else(|e| panic!("{e}")) else {
                panic!("expected dict");
            };
            assert!(matches!(dict["seed"], Object::Opaque));
            assert!(matches!(dict["n"], Object::Int(7)));
        }
    }

    #[test]
    fn a_long_that_is_not_a_number_is_an_error() {
        // LONG carrying junk rather than digits.
        assert!(plain(b"Lnope\n.").is_err());
    }

    #[test]
    fn truncated_string_length_does_not_allocate() {
        // BINUNICODE claiming nearly 4 GiB followed by two bytes.
        let bytes: &[u8] = &[0x80, 0x02, b'X', 0xff, 0xff, 0xff, 0xff, b'h', b'i', b'.'];
        assert!(matches!(plain(bytes).unwrap_err(), PickleError::Io(_)));
    }

    #[test]
    fn unknown_reduce_is_kept_opaque() {
        // The shape of a pickled numpy scalar: REDUCE of numpy.core.multiarray.scalar.
        let bytes = b"\x80\x02cnumpy.core.multiarray\nscalar\nU\x02f8U\x08\x00\x00\x00\x00\x00\x00\xe0?\x86R.";
        assert!(matches!(plain(bytes).unwrap(), Object::Opaque));
    }

    #[test]
    fn unknown_call_consuming_tensor_data_is_an_error() {
        let source = fixture_source();
        let callable = Object::Class {
            module_name: "torch._utils".to_string(),
            name: "_rebuild_qtensor".to_string(),
        };
        let args = rebuild_args("FloatStorage", "0", 32, 0, &[2], &[1], &source);
        assert!(matches!(
            reduce(callable, args).unwrap_err(),
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
    fn strided_views_match_reference() {
        // Expected values computed independently as storage[offset + sum(i_k * stride_k)]
        // over the fixture storage, whose element k holds the value k. Covers the layouts
        // real checkpoints produce: channels_last conv weights, transposes, step slices,
        // overlapping as_strided windows, size-one dims with leftover strides, inner
        // expands, and empty views at the end of storage.
        type Case = (
            &'static str,
            &'static [usize],
            &'static [usize],
            usize,
            &'static [f32],
        );
        let cases: &[Case] = &[
            (
                "channels_last",
                &[2, 3, 2, 2][..],
                &[12, 1, 6, 3][..],
                0,
                &[
                    0.0, 3.0, 6.0, 9.0, 1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 12.0, 15.0, 18.0,
                    21.0, 13.0, 16.0, 19.0, 22.0, 14.0, 17.0, 20.0, 23.0,
                ][..],
            ),
            (
                "transposed",
                &[3, 4][..],
                &[1, 3][..],
                2,
                &[
                    2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0, 4.0, 7.0, 10.0, 13.0,
                ][..],
            ),
            (
                "step_slice",
                &[5][..],
                &[2][..],
                1,
                &[1.0, 3.0, 5.0, 7.0, 9.0][..],
            ),
            (
                "overlapping_windows",
                &[4, 3][..],
                &[1, 1][..],
                0,
                &[0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0][..],
            ),
            (
                "size_one_dim_odd_stride",
                &[1, 3][..],
                &[12, 1][..],
                4,
                &[4.0, 5.0, 6.0][..],
            ),
            (
                "size_one_inner_dim",
                &[3, 1][..],
                &[1, 7][..],
                0,
                &[0.0, 1.0, 2.0][..],
            ),
            (
                "expand_inner",
                &[2, 3, 2][..],
                &[0, 1, 0][..],
                3,
                &[3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0][..],
            ),
            ("empty_leading", &[0, 3][..], &[3, 1][..], 30, &[][..]),
            (
                "empty_trailing_at_end",
                &[2, 0][..],
                &[0, 1][..],
                32,
                &[][..],
            ),
        ];
        let source = fixture_source();
        for &(name, shape, stride, offset, expected) in cases {
            let to_i64 = |v: &[usize]| v.iter().map(|&x| x as i64).collect::<Vec<_>>();
            let tensor = rebuild(rebuild_args(
                "FloatStorage",
                "0",
                32,
                offset as i64,
                &to_i64(shape),
                &to_i64(stride),
                &source,
            ))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
            let data = bridge::into_data(tensor).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(data.as_slice::<f32>().unwrap(), expected, "{name}");
        }
    }

    #[test]
    fn strided_views_for_other_element_sizes() {
        // The same 128 storage bytes viewed as i16, u8 and f64 through non-contiguous
        // strides; expected values are the reinterpreted bytes gathered independently.
        let source = fixture_source();

        let tensor = rebuild(rebuild_args(
            "ShortStorage",
            "0",
            64,
            1,
            &[2, 3],
            &[1, 2],
            &source,
        ))
        .unwrap();
        let data = bridge::into_data(tensor).unwrap();
        assert_eq!(
            data.as_slice::<i16>().unwrap(),
            &[0, 16256, 16384, 0, 0, 0][..]
        );

        let tensor = rebuild(rebuild_args(
            "ByteStorage",
            "0",
            128,
            2,
            &[2, 4],
            &[1, 4],
            &source,
        ))
        .unwrap();
        let data = bridge::into_data(tensor).unwrap();
        assert_eq!(
            data.as_slice::<u8>().unwrap(),
            &[0, 128, 0, 64, 0, 63, 64, 64][..]
        );

        let tensor = rebuild(rebuild_args(
            "DoubleStorage",
            "0",
            16,
            0,
            &[2, 2],
            &[1, 2],
            &source,
        ))
        .unwrap();
        let data = bridge::into_data(tensor).unwrap();
        let bits: Vec<u64> = data
            .as_slice::<f64>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(
            bits,
            vec![
                4575657221408423936,
                4656722015783223296,
                4629700418010611712,
                4674736414296899584
            ]
        );
    }

    #[test]
    fn stride_rejects_non_integer_values() {
        let invalid_stride = Object::Tuple(vec![Object::String("one".to_string())]);
        assert!(matches!(
            parse_dims(&invalid_stride, "stride"),
            Err(PickleError::InvalidData(msg)) if msg == "stride must be an int, got str"
        ));
    }

    #[test]
    fn view_beyond_declared_storage_fails_at_parse_time() {
        // The persistent id's element count is authoritative even when the file holds
        // more bytes: PyTorch's own `set_` raises for a view past the declared storage.
        let source = fixture_source();
        let err = rebuild(rebuild_args(
            "FloatStorage",
            "0",
            32,
            30,
            &[2, 3],
            &[3, 1],
            &source,
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            PickleError::InvalidData(msg) if msg.contains("needs 36 elements, but storage '0' declares 32")
        ));
    }

    #[test]
    fn declared_storage_larger_than_file_fails_at_read_time() {
        let source = fixture_source();
        let tensor = rebuild(rebuild_args(
            "FloatStorage",
            "0",
            64,
            40,
            &[2, 3],
            &[3, 1],
            &source,
        ))
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
        let err = rebuild(rebuild_args(
            "FloatStorage",
            "0",
            32,
            0,
            &[1 << 21, 1 << 21],
            &[0, 0],
            &source,
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            PickleError::InvalidData(msg) if msg.contains("byte limit")
        ));
    }

    #[test]
    fn small_broadcast_view_loads() {
        let source = fixture_source();
        let tensor = rebuild(rebuild_args(
            "FloatStorage",
            "0",
            32,
            1,
            &[2, 3],
            &[0, 1],
            &source,
        ))
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
        let tensor = rebuild(rebuild_args(
            "FloatStorage",
            "0",
            32,
            5,
            &[2, 3],
            &[3, 1],
            &source,
        ))
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
        let tensor = rebuild(rebuild_args(
            "FloatStorage",
            "0",
            32,
            5,
            &[2, 4, 3],
            &[12, 1, 4],
            &source,
        ))
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
        let tensor = rebuild(rebuild_args("FloatStorage", "0", 32, 5, &[], &[], &source)).unwrap();
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
        let tensor = build_tensor(storage, DType::F32, 1, vec![3], vec![1]).unwrap();
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
        let Object::Tensor(tensor) = rebuild_tensor(args, TensorRebuild::V3).unwrap() else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.dtype, DType::U32);
        let data = bridge::into_data(tensor).unwrap();
        // The storage holds f32 0.0 and 1.0; read as u32 bit patterns.
        assert_eq!(data.as_slice::<u32>().unwrap(), &[0, 1.0f32.to_bits()]);
    }

    #[test]
    fn rebuild_from_type_v2_dispatches_inner_call() {
        // _rebuild_from_type_v2(func, new_type, args, state) for a tensor subclass.
        let source = fixture_source();
        let callable = Object::Class {
            module_name: "torch._tensor".to_string(),
            name: "_rebuild_from_type_v2".to_string(),
        };
        let args = Object::Tuple(vec![
            Object::Class {
                module_name: "torch._utils".to_string(),
                name: "_rebuild_tensor".to_string(),
            },
            Object::Class {
                module_name: "torch".to_string(),
                name: "Tensor".to_string(),
            },
            rebuild_args("FloatStorage", "0", 32, 5, &[2, 3], &[3, 1], &source),
            Object::Dict(HashMap::new()),
        ]);
        let Object::Tensor(tensor) = reduce(callable, args).unwrap() else {
            panic!("expected tensor");
        };
        let data = bridge::into_data(tensor).unwrap();
        assert_eq!(
            data.as_slice::<f32>().unwrap(),
            &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        );
    }

    #[test]
    fn bool_bytes_are_normalized() {
        // Read the f32 storage as bools: 0.0 is eight zero bytes, 1.0 is 00 00 80 3f.
        let source = fixture_source();
        let pid = storage_pid("BoolStorage", "0", 128);
        let Object::Tuple(pid) = pid else {
            unreachable!()
        };
        let storage = resolve_storage_id(&pid, &source).unwrap();
        let tensor =
            build_tensor(storage, DType::Bool(BoolStore::Native), 0, vec![8], vec![1]).unwrap();
        let data = bridge::into_data(tensor).unwrap();
        assert_eq!(
            data.as_slice::<bool>().unwrap(),
            &[false, false, false, false, false, false, true, true]
        );
    }

    #[test]
    fn untyped_storage_with_view_metadata_is_rejected() {
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
            Object::Tuple(vec![
                Object::String("7".to_string()),
                Object::Int(4),
                Object::Int(10),
            ]),
        ];
        assert!(matches!(
            resolve_storage_id(&pid, &source),
            Err(PickleError::InvalidData(msg)) if msg.contains("untyped but carries view metadata")
        ));
    }

    #[test]
    fn long_at_i64_boundary_does_not_fit() {
        // 2^63 and -2^63-1 need 9 bytes whose sign fill disagrees with byte 7.
        assert!(int_from_le_bytes(&[0, 0, 0, 0, 0, 0, 0, 0x80, 0]).is_none());
        assert!(
            int_from_le_bytes(&[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xff]).is_none()
        );
        assert_eq!(
            int_from_le_bytes(&[0, 0, 0, 0, 0, 0, 0, 0x80, 0xff]).unwrap(),
            i64::MIN
        );
    }

    #[test]
    fn deep_nesting_is_rejected() {
        // EMPTY_LIST then TUPLE1 repeated wraps the list ever deeper.
        let mut bytes = vec![0x80, 0x02, b']'];
        bytes.extend(std::iter::repeat_n(0x85, 1500));
        bytes.push(b'.');
        assert!(matches!(
            plain(&bytes).unwrap_err(),
            PickleError::InvalidData(msg) if msg.contains("nesting exceeds")
        ));

        // Just under the limit still parses.
        let mut bytes = vec![0x80, 0x02, b']'];
        bytes.extend(std::iter::repeat_n(0x85, 900));
        bytes.push(b'.');
        assert!(plain(&bytes).is_ok());
    }

    #[test]
    fn mark_never_becomes_a_value() {
        // MARK TUPLE1: nothing above the mark to wrap.
        assert!(matches!(
            plain(b"\x80\x02(\x85.").unwrap_err(),
            PickleError::InvalidData(msg) if msg.contains("found MARK")
        ));
        // MARK BINPUT 0: memoizing a mark.
        assert!(plain(b"\x80\x02(q\x00.").is_err());
        // A bare MARK on top at STOP.
        assert!(plain(b"\x80\x02K\x01(.").is_err());
        // POP discards a bare mark, as in CPython.
        assert!(matches!(
            plain(b"\x80\x02K\x01(0.").unwrap(),
            Object::Int(1)
        ));
    }

    #[test]
    fn build_on_a_tensor_is_an_error() {
        let source = fixture_source();
        let tensor = rebuild(rebuild_args(
            "FloatStorage",
            "0",
            32,
            0,
            &[2],
            &[1],
            &source,
        ))
        .unwrap();
        let err = opaque(&Object::Tensor(tensor), &Object::Dict(HashMap::new())).unwrap_err();
        assert!(matches!(err, PickleError::UnsupportedType(_)));
    }

    #[test]
    fn ordered_dict_rejects_non_list_argument() {
        let args = Object::Tuple(vec![Object::Int(3)]);
        assert!(matches!(
            ordered_dict(args),
            Err(PickleError::InvalidData(msg)) if msg.contains("list of pairs")
        ));
    }

    #[test]
    fn missing_storage_returns_contextual_error() {
        let source = fixture_source();
        let tensor = rebuild(rebuild_args(
            "FloatStorage",
            "missing",
            6,
            0,
            &[2, 3],
            &[3, 1],
            &source,
        ))
        .unwrap();
        let err = bridge::into_data(tensor).unwrap_err();
        assert!(matches!(
            err,
            PackError::ValidationError(msg)
                if msg.contains("Failed to read storage 'missing' for tensor with shape [2, 3]")
        ));
    }
}
