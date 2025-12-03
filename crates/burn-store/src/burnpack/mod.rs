//! # Burnpack - Native Burn Model Storage Format
//!
//! Burnpack is the native binary storage format for Burn models, designed for efficient
//! serialization, fast loading, and cross-platform compatibility.
//!
//! ## Key Features
//!
//! - **CBOR Metadata**: Structured metadata with efficient binary encoding
//! - **Memory-Mapped Loading**: Zero-copy loading for optimal performance
//! - **256-byte Tensor Alignment**: Enables efficient mmap zero-copy access
//! - **No-std Support**: Works in embedded and WASM environments
//! - **ParamId Persistence**: Preserves parameter identities for stateful training
//! - **Lazy Tensor Loading**: Deferred data materialization for efficient memory usage
//!
//! ## File Format Structure
//!
//! ```text
//! ┌──────────────────────────────────┐
//! │  Header (10 bytes)               │
//! ├──────────────────────────────────┤
//! │  - Magic number (4 bytes)        │  0x4E525542 ("NRUB" in LE)
//! │  - Version (2 bytes)             │  Format version (0x0001)
//! │  - Metadata size (4 bytes)       │  Size of CBOR metadata (u32)
//! ├──────────────────────────────────┤
//! │  Metadata (CBOR)                 │
//! ├──────────────────────────────────┤
//! │  - Tensor descriptors (BTreeMap) │  Order-preserving map of tensor metadata
//! │    Key: tensor name (string)     │  e.g., "model.layer1.weight"
//! │    Value: TensorDescriptor       │
//! │      - dtype: DType              │  Data type (F32, F64, I32, etc.)
//! │      - shape: Vec<u64>           │  Tensor dimensions
//! │      - data_offsets: (u64, u64)  │  (start, end) byte offsets (256-byte aligned)
//! │      - param_id: Option<u64>     │  Parameter ID (for training state)
//! │  - Additional metadata(BTreeMap) │  User-defined key-value pairs
//! ├──────────────────────────────────┤
//! │  Tensor Data Section             │
//! ├──────────────────────────────────┤
//! │  [padding][tensor1][padding]...  │  Each tensor aligned to 256-byte boundary
//! │  Raw tensor bytes (little-endian)│  Enables mmap zero-copy loading
//! └──────────────────────────────────┘
//! ```
//!
//! ## Tensor Alignment
//!
//! All tensor data is aligned to 256-byte boundaries to enable efficient memory-mapped
//! (mmap) zero-copy loading. This alignment ensures:
//!
//! - Proper pointer alignment for all tensor element types (f64 requires 8 bytes)
//! - Cache-line friendly access (most CPUs use 64-byte cache lines)
//! - GPU memory alignment (CUDA prefers 256-byte for coalesced access)
//! - Future-proofing for wider SIMD instructions (AVX-512, future AVX-1024)
//!
//! The 256-byte alignment matches industry standards used by GGUF, MLX, ncnn, MNN,
//! and other major model formats.

pub mod base;
pub mod reader;
pub mod store;
pub mod writer;

#[cfg(test)]
mod tests;
