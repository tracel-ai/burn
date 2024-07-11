use crate::frontend::CubeContext;
use crate::ir::Synchronization;

pub fn sync_units() {}

pub mod sync_units {
    use super::*;

    pub fn __expand(context: &mut CubeContext) {
        context.register(Synchronization::SyncUnits)
    }
}
