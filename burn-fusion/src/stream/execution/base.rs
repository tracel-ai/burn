#[derive(Clone, Copy, Debug)]
pub(crate) enum ExecutionMode {
    // Signal that we execute the graph after a new ops is added to the graph.
    Lazy,
    // Signal that we execute the graph because of a sync without any new ops added to the graph.
    Sync,
}
