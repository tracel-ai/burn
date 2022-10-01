use crate::module::{State, StateError};

#[derive(Debug)]
pub enum CheckpointerError {
    IOError(std::io::Error),
    StateError(StateError),
}

pub trait Checkpointer<E> {
    fn save(&self, epoch: usize, state: State<E>) -> Result<(), CheckpointerError>;
    fn restore(&self, epoch: usize) -> Result<State<E>, CheckpointerError>;
}
