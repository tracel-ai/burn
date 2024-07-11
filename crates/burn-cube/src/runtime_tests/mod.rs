pub mod cmma;
pub mod launch;
pub mod slice;
pub mod subcube;

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_all {
    () => {
        use burn_cube::prelude::*;

        burn_cube::testgen_subcube!();
        burn_cube::testgen_launch!();
        burn_cube::testgen_cmma!();
        burn_cube::testgen_slice!();
    };
}
