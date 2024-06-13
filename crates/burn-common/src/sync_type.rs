/// What kind of synchronization to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncType {
    /// Submit all outstanding tasks to the task queue if any.
    Flush,
    /// Submit all tasks to the task queue and wait for all of them to complete.
    Wait,
}
