extern crate alloc;

#[cfg(feature = "std")]
pub use burn_tensor_testgen::might_panic;

/// Generate a test module with custom floating element types.
#[macro_export]
macro_rules! test_float_elem_variant {
    ($modname:ident, $float:ty, $module:literal, [$($feat:literal),* $(,)?]) => {
        #[cfg(all(test, any($(feature = $feat),*)))]
        mod $modname {
            pub type FloatElemType = $float;
            #[allow(unused)]
            pub use super::IntElemType;

            mod ty {
                include!("backend.rs");
                include!($module);
            }
        }
    };
}
