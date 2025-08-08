use core::fmt::Debug;

/// Helper trait for implementing indexing with support for negative indices.
///
/// # Example
/// ```rust
/// use burn_common::baselib::indexing::{AsIndex, canonicalize_dim};
///
/// fn example<I: AsIndex, const D: usize>(dim: I, size: usize) -> isize {
///    let dim: usize = canonicalize_dim(dim, D, false);
///    unimplemented!()
/// }
/// ```
pub trait AsIndex: Debug + Copy + Sized {
    /// Converts into a slice index.
    fn index(self) -> isize;
}

impl AsIndex for usize {
    fn index(self) -> isize {
        self as isize
    }
}

impl AsIndex for isize {
    fn index(self) -> isize {
        self
    }
}

impl AsIndex for i64 {
    fn index(self) -> isize {
        self as isize
    }
}

impl AsIndex for u64 {
    fn index(self) -> isize {
        self as isize
    }
}

// Default integer type
impl AsIndex for i32 {
    fn index(self) -> isize {
        self as isize
    }
}

impl AsIndex for u32 {
    fn index(self) -> isize {
        self as isize
    }
}

impl AsIndex for i16 {
    fn index(self) -> isize {
        self as isize
    }
}

impl AsIndex for u16 {
    fn index(self) -> isize {
        self as isize
    }
}

impl AsIndex for i8 {
    fn index(self) -> isize {
        self as isize
    }
}

impl AsIndex for u8 {
    fn index(self) -> isize {
        self as isize
    }
}
