use cubecl::{define_size, prelude::ElemExpand};

/// The element type ID to be used for dynamic element type while expanding a fused kernel.
pub(crate) const DYN_ELEM_ID: usize = usize::MAX;
pub(crate) type DynElem = ElemExpand<DYN_ELEM_ID>;

define_size!(DynSize);
