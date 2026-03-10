use cubecl::define_size;

/// The element type ID to be used for dynamic element type while expanding a fused kernel.
pub(crate) const DYN_ELEM_ID: usize = usize::MAX;
/// The element type ID to be used for the quantization store element type while expanding a fused kernel.
pub(crate) const Q_STORE_DYN_ELEM_ID: usize = usize::MAX - 1;
/// The element type ID to be used for the quantization param element type while expanding a fused kernel.
pub(crate) const Q_PARAM_DYN_ELEM_ID: usize = usize::MAX - 2;

define_size!(DynSize);
define_size!(DynQStoreSize);
define_size!(DynQParamSize);
