//! What the reader produces for each tensor in a checkpoint.

use std::fmt;
use std::io;
use std::sync::Arc;

/// Element type of a tensor.
///
/// The types `torch.save` writes as a plain storage. Complex, quantized and sparse tensors
/// are refused at parse time rather than given a variant here. Deliberately exhaustive: a
/// converter matching on it should fail to compile, not fall through, when a variant is
/// added, so adding one is a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    /// One byte per element, holding 0 or 1.
    Bool,
}

impl DType {
    /// Size of one element in bytes.
    pub const fn size(self) -> usize {
        match self {
            DType::F64 | DType::I64 | DType::U64 => 8,
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F16 | DType::BF16 | DType::I16 | DType::U16 => 2,
            DType::I8 | DType::U8 | DType::Bool => 1,
        }
    }
}

/// A tensor in a checkpoint, with its bytes not yet read.
///
/// Opening a checkpoint parses its metadata only. [`read`](Self::read) is what produces the
/// tensor's bytes, so a checkpoint can be inspected without loading it, or loaded one tensor
/// at a time. ZIP and legacy containers read the file at that point; a TAR container's
/// storages are held in memory from open.
///
/// The element type and shape are fixed at parse time, when the reader checks that the
/// tensor fits its storage, so they are read-only: [`byte_len`](Self::byte_len) is what
/// [`read`](Self::read) returns because nothing can move them apart.
#[derive(Clone)]
pub struct Tensor {
    /// Dotted path of the tensor through the checkpoint's dictionaries
    /// (`"encoder.0.weight"`). Free to rename; only error messages look at it.
    pub name: String,
    dtype: DType,
    shape: Vec<usize>,
    read: Arc<dyn Fn() -> io::Result<Vec<u8>> + Send + Sync>,
}

impl Tensor {
    pub(crate) fn new(
        name: String,
        dtype: DType,
        shape: Vec<usize>,
        read: impl Fn() -> io::Result<Vec<u8>> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name,
            dtype,
            shape,
            read: Arc::new(read),
        }
    }

    /// Element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Extents, outermost first. Empty for a scalar.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Number of bytes [`read`](Self::read) returns.
    pub fn byte_len(&self) -> usize {
        self.num_elements() * self.dtype.size()
    }

    /// Read the tensor's bytes from the checkpoint.
    ///
    /// The bytes are contiguous in row-major order and native-endian, whatever layout the
    /// file stored the tensor in: a strided or offset view is gathered, and a `bool` tensor
    /// holds only 0 and 1. Nothing is cached, so each call reads the storage again.
    ///
    /// # Errors
    ///
    /// A storage that is missing, corrupt or unreadable by its container fails with
    /// [`io::ErrorKind::InvalidData`], and one shorter than the pickle declared with
    /// [`io::ErrorKind::UnexpectedEof`]: the file disagrees with itself. Any other kind is
    /// the operating system's, and means the file could not be read at all. Every error
    /// names the tensor and its storage.
    pub fn read(&self) -> io::Result<Vec<u8>> {
        (self.read)().map_err(|err| named(&self.name, err))
    }

    /// Split the tensor into its metadata and its read handle.
    ///
    /// For a caller that keeps the metadata elsewhere and wants only the handle, without a
    /// second copy of the name and shape held alive with it. The handle reads and fails
    /// exactly as [`read`](Self::read) does.
    pub fn into_parts(
        self,
    ) -> (
        String,
        DType,
        Vec<usize>,
        impl Fn() -> io::Result<Vec<u8>> + Send + Sync + 'static,
    ) {
        let Self {
            name,
            dtype,
            shape,
            read,
        } = self;
        let context = name.clone();

        (name, dtype, shape, move || {
            read().map_err(|err| named(&context, err))
        })
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("name", &self.name)
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

/// Put the tensor's name in front of a read error, keeping its kind.
///
/// The reader builds a tensor before it knows the name (that is assembled from the dict
/// path afterwards), so the name is added here rather than inside the handle.
fn named(name: &str, err: io::Error) -> io::Error {
    if name.is_empty() {
        return err;
    }
    io::Error::new(err.kind(), format!("tensor '{name}': {err}"))
}
