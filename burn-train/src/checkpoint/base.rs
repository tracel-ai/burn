use burn_core::record::{Record, RecorderError};

#[derive(Debug)]
pub enum CheckpointerError {
    IOError(std::io::Error),
    RecorderError(RecorderError),
    Unknown(String),
}

pub trait Checkpointer<R: Record> {
    fn save(&self, epoch: usize, record: R) -> Result<(), CheckpointerError>;
    fn restore(&self, epoch: usize) -> Result<R, CheckpointerError>;
}
