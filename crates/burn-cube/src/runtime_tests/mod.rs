pub mod subcube;

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_all {
    () => {
        use burn_cube::prelude::*;

        burn_cube::testgen_subcube!();
    };
}
