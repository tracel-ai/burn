//! Just enough pickle support to be able to read PyTorch checkpoints.
// This hardcodes objects that are required for tensor reading, we may want to make this a bit more
// composable/tensor agnostic at some point.
use crate::TensorSnapshot;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use burn_core::module::ParamId;
use burn_tensor::{DType, TensorData};
use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::io::{self, BufRead, Read};

const VERBOSE: bool = false;

/// Error type for pickle operations
#[derive(Debug)]
pub enum Error {
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

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "IO error: {}", e),
            Error::InvalidOpCode(code) => write!(f, "Invalid opcode: {}", code),
            Error::InvalidProtocol(proto) => write!(f, "Invalid protocol: {}", proto),
            Error::UnexpectedOpCode(op) => write!(f, "Unexpected opcode: {:?}", op),
            Error::UnsupportedType(ty) => write!(f, "Unsupported type: {}", ty),
            Error::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
            Error::StackUnderflow => write!(f, "Stack underflow"),
            Error::MemoNotFound(idx) => write!(f, "Memo index not found: {}", idx),
            Error::InvalidShapeOrType => write!(f, "Invalid tensor shape or type"),
        }
    }
}

impl std::error::Error for Error {}

type Result<T> = std::result::Result<T, Error>;

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
    BinInt = b'J',
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
    BinFloat = b'G',
    Append = b'a',
    Appends = b'e',
    Long1 = 0x8a,
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
            b'J' => Ok(Self::BinInt),
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
            b'd' => Ok(Self::EmptyDict),
            b'b' => Ok(Self::Build),
            b'.' => Ok(Self::Stop),
            0x81 => Ok(Self::NewObj),
            b']' => Ok(Self::EmptyList),
            b'G' => Ok(Self::BinFloat),
            b'a' => Ok(Self::Append),
            b'e' => Ok(Self::Appends),
            0x8a => Ok(Self::Long1),
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
    String::from_utf8(buf.to_vec()).map_err(|e| Error::InvalidData(format!("Invalid UTF-8: {}", e)))
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

fn rebuild_from_type_v2(o: Object, memo: &mut HashMap<u32, Object>) -> Result<Object> {
    let args = if let Object::Tuple(args) = o {
        if args.is_empty() {
            return Err(Error::InvalidData(
                "rebuild_from_type_v2: empty args".to_string(),
            ));
        }
        args
    } else {
        return Err(Error::InvalidData(format!(
            "rebuild_from_type_v2: expected tuple got {:?}",
            o
        )));
    };
    let func = &args[0];
    match func {
        Object::Class { module_name, name } => {
            let module_name = module_name.as_str();
            let name = name.as_str();
            let args = Object::Tuple(args[1..].to_vec());
            if module_name == "torch._utils" && name == "_rebuild_tensor_v2" {
                rebuild_tensor_v2(args, memo)
            } else if module_name == "torch._tensor" && name == "_rebuild_from_type_v2" {
                rebuild_from_type_v2(args, memo)
            } else if module_name == "torch._utils" && name == "_rebuild_parameter" {
                rebuild_parameter(args, memo)
            } else {
                Err(Error::UnsupportedType(format!("{}.{}", module_name, name)))
            }
        }
        _ => Err(Error::InvalidData(format!(
            "rebuild_from_type_v2: expected class got {:?}",
            func
        ))),
    }
}

fn rebuild_parameter(args: Object, memo: &mut HashMap<u32, Object>) -> Result<Object> {
    let args = if let Object::Tuple(args) = args {
        if args.is_empty() {
            return Err(Error::InvalidData(
                "rebuild_parameter: empty args".to_string(),
            ));
        }
        args
    } else {
        return Err(Error::InvalidData(format!(
            "rebuild_parameter: expected tuple got {:?}",
            args
        )));
    };
    let data = &args[0];
    let tensor = match data {
        Object::Reduce {
            callable: _,
            args: _,
        } => rebuild_from_type_v2(data.clone(), memo)?,
        _ => data.clone(),
    };
    Ok(tensor)
}

fn rebuild_tensor_v2(args: Object, _memo: &mut HashMap<u32, Object>) -> Result<Object> {
    // args is (layout, shape, stride, _requires_grad, _backward_hooks)
    let args = if let Object::Tuple(args) = args {
        args
    } else {
        return Err(Error::InvalidData(format!(
            "rebuild_tensor_v2: expected tuple got {:?}",
            args
        )));
    };

    if args.len() < 5 {
        return Err(Error::InvalidData(format!(
            "rebuild_tensor_v2: expected at least 5 args, got {}",
            args.len()
        )));
    }

    let persistent_id = match &args[0] {
        Object::Persistent(data) => data.clone(),
        _ => {
            return Err(Error::InvalidData(format!(
                "rebuild_tensor_v2: expected persistent id got {:?}",
                args[0]
            )));
        }
    };

    let shape = match &args[2] {
        Object::Tuple(shape) => shape
            .iter()
            .map(|x| match x {
                Object::Int(i) => Ok(*i as usize),
                _ => Err(Error::InvalidData("shape must contain ints".to_string())),
            })
            .collect::<Result<Vec<_>>>()?,
        _ => {
            return Err(Error::InvalidData(format!(
                "rebuild_tensor_v2: expected shape tuple got {:?}",
                args[2]
            )));
        }
    };

    // For now, we'll just store the persistent ID and shape
    // The actual data will need to be loaded from the zip file separately
    let shape_clone = shape.clone();
    Ok(Object::TorchParam(TensorSnapshot::from_closure(
        Rc::new(move || {
            // This will be filled in when we actually load the data
            // Create a dummy tensor data with correct shape
            let num_elements = shape_clone.iter().product::<usize>().max(1);
            TensorData::new(vec![0.0f32; num_elements], shape_clone.clone())
        }),
        DType::F32, // Default, will be updated when loading actual data
        shape,
        vec![],
        vec![],
        ParamId::new(),
    )))
}

pub struct Stack {
    stack: Vec<Object>,
    metastack: Vec<Vec<Object>>,
    memo: HashMap<u32, Object>,
}

impl Default for Stack {
    fn default() -> Self {
        Self::new()
    }
}

impl Stack {
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            metastack: Vec::new(),
            memo: HashMap::new(),
        }
    }

    fn push(&mut self, o: Object) {
        self.stack.push(o)
    }

    fn pop(&mut self) -> Result<Object> {
        match self.stack.pop() {
            None => Err(Error::StackUnderflow),
            Some(o) => Ok(o),
        }
    }

    fn top(&self) -> Result<Object> {
        match self.stack.last() {
            None => Err(Error::StackUnderflow),
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
            .ok_or(Error::InvalidData("marker not found".to_string()))?;

        let result = self.stack.split_off(marker_pos + 1);
        self.stack.pop(); // Remove the marker
        Ok(result)
    }

    fn last(&self) -> Result<&Object> {
        match self.stack.last() {
            None => Err(Error::StackUnderflow),
            Some(o) => Ok(o),
        }
    }

    fn last_mut(&mut self) -> Result<&mut Object> {
        match self.stack.last_mut() {
            None => Err(Error::StackUnderflow),
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
        self.memo.get(&idx).cloned().ok_or(Error::MemoNotFound(idx))
    }

    fn memo_put(&mut self, idx: u32, obj: Object) {
        self.memo.insert(idx, obj);
    }
}

fn read_global<R: BufRead>(r: &mut R, stack: &mut Stack) -> Result<()> {
    let module_name = buf_to_str(&read_to_newline(r)?)?;
    let name = buf_to_str(&read_to_newline(r)?)?;
    if VERBOSE {
        println!("  global {module_name} {name}");
    }
    stack.push(Object::Class { module_name, name });
    Ok(())
}

fn read_long1<R: BufRead>(r: &mut R, stack: &mut Stack) -> Result<()> {
    let len = r.read_u8()? as usize;
    let mut data = vec![0u8; len];
    r.read_exact(&mut data)?;
    // Handle little-endian signed integer
    let mut value = 0i64;
    for (i, &byte) in data.iter().enumerate() {
        value |= (byte as i64) << (i * 8);
    }
    // Handle sign extension for negative numbers
    if len < 8 && data.last().is_some_and(|&b| b & 0x80 != 0) {
        // Sign extend
        for i in len..8 {
            value |= 0xff << (i * 8);
        }
    }
    stack.push(Object::Int(value));
    Ok(())
}

fn read_string<R: BufRead>(r: &mut R, stack: &mut Stack, len: usize) -> Result<()> {
    let mut data = vec![0u8; len];
    r.read_exact(&mut data)?;
    let s = buf_to_str(&data)?;
    if VERBOSE {
        println!("  string {s:?}");
    }
    stack.push(Object::String(s));
    Ok(())
}

fn read_bin_int<R: BufRead>(r: &mut R, stack: &mut Stack) -> Result<()> {
    let v = r.read_i32::<LittleEndian>()?;
    if VERBOSE {
        println!("  int {v}");
    }
    stack.push(Object::Int(v as i64));
    Ok(())
}

fn read_bin_int1<R: BufRead>(r: &mut R, stack: &mut Stack) -> Result<()> {
    let v = r.read_u8()?;
    if VERBOSE {
        println!("  int {v}");
    }
    stack.push(Object::Int(v as i64));
    Ok(())
}

fn read_bin_int2<R: BufRead>(r: &mut R, stack: &mut Stack) -> Result<()> {
    let v = r.read_u16::<LittleEndian>()?;
    if VERBOSE {
        println!("  int {v}");
    }
    stack.push(Object::Int(v as i64));
    Ok(())
}

fn read_bin_float<R: BufRead>(r: &mut R, stack: &mut Stack) -> Result<()> {
    let v = r.read_f64::<LittleEndian>()?;
    if VERBOSE {
        println!("  float {v}");
    }
    stack.push(Object::Float(v));
    Ok(())
}

pub fn read_pickle<R: BufRead>(r: &mut R) -> Result<Object> {
    let mut stack = Stack::new();
    loop {
        let op_code = r.read_u8()?;
        if VERBOSE {
            println!("op_code {op_code}");
        }
        let op_code = OpCode::try_from(op_code).map_err(Error::InvalidOpCode)?;
        match op_code {
            OpCode::Proto => {
                let version = r.read_u8()?;
                if version > 5 {
                    return Err(Error::InvalidProtocol(version));
                }
                if VERBOSE {
                    println!("  proto {version}");
                }
            }
            OpCode::Global => read_global(r, &mut stack)?,
            OpCode::BinInt => read_bin_int(r, &mut stack)?,
            OpCode::BinInt1 => read_bin_int1(r, &mut stack)?,
            OpCode::BinInt2 => read_bin_int2(r, &mut stack)?,
            OpCode::BinFloat => read_bin_float(r, &mut stack)?,
            OpCode::BinUnicode => {
                let len = r.read_u32::<LittleEndian>()? as usize;
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
                    _ => return Err(Error::UnexpectedOpCode(op_code)),
                }
            }
            OpCode::Appends => {
                let objs = stack.pop_to_marker()?;
                match stack.last_mut()? {
                    Object::List(list) => list.extend(objs),
                    _ => return Err(Error::UnexpectedOpCode(op_code)),
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
                            return Err(Error::InvalidData(
                                "dict key must be a string".to_string(),
                            ));
                        }
                    }
                    _ => return Err(Error::UnexpectedOpCode(op_code)),
                }
            }
            OpCode::SetItems => {
                let mut objs = stack.pop_to_marker()?;
                if objs.len() % 2 != 0 {
                    return Err(Error::InvalidData(
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
                                return Err(Error::InvalidData(
                                    "dict key must be a string".to_string(),
                                ));
                            }
                        }
                    }
                    _ => return Err(Error::UnexpectedOpCode(op_code)),
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
                if let Object::String(s) = pid {
                    stack.push(Object::Persistent(s.into_bytes()));
                } else {
                    return Err(Error::InvalidData(
                        "persistent id must be a string".to_string(),
                    ));
                }
            }
            OpCode::Reduce => {
                let args = stack.pop()?;
                let callable = stack.pop()?;
                let obj = Object::Reduce {
                    callable: Box::new(callable.clone()),
                    args: Box::new(args.clone()),
                };
                let obj =
                    rebuild_from_type_v2(Object::Tuple(vec![callable, args]), &mut stack.memo)?;
                stack.push(obj);
            }
            OpCode::Build => {
                let args = stack.pop()?;
                let obj = stack.pop()?;
                match obj {
                    Object::Dict(_) => {
                        // For dicts, BUILD updates with the args
                        if let Object::Dict(update) = args
                            && let Object::Dict(dict) = stack.last_mut()?
                        {
                            dict.extend(update);
                        }
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
            OpCode::Stop => break,
            OpCode::Dict => {
                let objs = stack.pop_to_marker()?;
                let mut dict = HashMap::new();
                if objs.len() % 2 != 0 {
                    return Err(Error::InvalidData(
                        "dict requires even number of objects".to_string(),
                    ));
                }
                for chunk in objs.chunks(2) {
                    if let Object::String(key) = &chunk[0] {
                        dict.insert(key.clone(), chunk[1].clone());
                    } else {
                        return Err(Error::InvalidData("dict key must be a string".to_string()));
                    }
                }
                stack.push(Object::Dict(dict));
            }
        }
    }
    stack.pop()
}

/// Load tensors from a pickle file (PyTorch checkpoint format)
pub fn read_pickle_tensors<R: BufRead>(reader: &mut R) -> Result<HashMap<String, TensorSnapshot>> {
    let obj = read_pickle(reader)?;

    // Extract tensors from the loaded object
    let mut tensors = HashMap::new();
    extract_tensors(&obj, String::new(), &mut tensors);

    Ok(tensors)
}

fn extract_tensors(obj: &Object, path: String, tensors: &mut HashMap<String, TensorSnapshot>) {
    match obj {
        Object::Dict(dict) => {
            for (key, value) in dict {
                let new_path = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{}.{}", path, key)
                };
                extract_tensors(value, new_path, tensors);
            }
        }
        Object::TorchParam(snapshot) => {
            tensors.insert(path, snapshot.clone());
        }
        _ => {}
    }
}
