use super::ParamId;
use crate::module::{LoadingError, State, StateNamed};
use crate::tensor::Element;

/// Define a trainable parameter.
#[derive(Debug)]
pub struct Param<T> {
    pub(super) id: ParamId,
    pub(super) value: T,
}

impl<T> std::fmt::Display for Param<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("Param: {}", self.id).as_str())
    }
}

impl<T> Param<T> {
    pub fn new(value: T) -> Self {
        Self {
            id: ParamId::new(),
            value,
        }
    }
}

impl<T> std::ops::Deref for Param<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

pub(super) fn state_with_id<E: Element>(id: ParamId, state: State<E>) -> State<E> {
    let mut state_wrapper = StateNamed::new();

    state_wrapper.register_state("data", state);
    state_wrapper.register_state("id", State::ParamId(id));

    State::StateNamed(state_wrapper)
}

pub(super) fn load_with_id<E: Element>(
    state: &State<E>,
) -> Result<(&ParamId, &State<E>), LoadingError> {
    let state_wrapper = match state {
        State::StateNamed(state) => state,
        _ => {
            return Err(LoadingError::new(
                "Can't load state wrapper to fetch id and data".to_string(),
            ))
        }
    };

    let state = match state_wrapper.get("data") {
        Some(state) => state,
        None => {
            return Err(LoadingError::new(
                "Can't load state data from state wrapper".to_string(),
            ))
        }
    };

    let id = match state_wrapper.get("id") {
        Some(State::ParamId(id)) => id,
        _ => {
            return Err(LoadingError::new(
                "Can't load state id from state wrapper".to_string(),
            ))
        }
    };

    Ok((id, state))
}
