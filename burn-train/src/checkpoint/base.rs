use burn_core::{
    module::StateError,
    record::{Record, RecorderError},
};

#[derive(Debug)]
pub enum CheckpointerError {
    IOError(std::io::Error),
    RecorderError(RecorderError),
    StateError(StateError),
    Unknown(String),
}

pub trait Checkpointer<R: Record> {
    fn save(&self, epoch: usize, record: R) -> Result<(), CheckpointerError>;
    fn restore(&self, epoch: usize) -> Result<R, CheckpointerError>;
}
