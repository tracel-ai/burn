use crate::frontend::CubeContext;
use crate::ir::Synchronization;

pub fn sync_units() {}

pub fn sync_units_expand(context: &mut CubeContext) {
    context.register(Synchronization::SyncUnits)
}
