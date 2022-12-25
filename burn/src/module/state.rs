use super::ParamId;
use crate::tensor::{DataSerialize, Element};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs::File, path::Path};

#[derive(Debug, PartialEq, Eq, Clone, Default, Serialize, Deserialize)]
pub struct StateNamed<E> {
    pub values: HashMap<String, State<E>>,
}

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub enum State<E> {
    StateNamed(StateNamed<E>),
    Data(DataSerialize<E>),
    ParamId(ParamId),
}

#[derive(Debug)]
pub enum StateError {
    InvalidFormat(String),
    FileNotFound(String),
}

impl std::fmt::Display for StateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut message = "State error => ".to_string();

        match self {
            Self::InvalidFormat(err) => {
                message += format!("Invalid format: {err}").as_str();
            }
            Self::FileNotFound(err) => {
                message += format!("File not found: {err}").as_str();
            }
        };

        f.write_str(message.as_str())
    }
}

impl std::error::Error for StateError {}

impl<E: Element> StateNamed<E> {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    pub fn register_state(&mut self, name: &str, state: State<E>) {
        self.values.insert(name.to_string(), state);
    }
}

impl<E: Element> StateNamed<E> {
    pub fn get(&self, name: &str) -> Option<&State<E>> {
        self.values.get(name)
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn convert<O: Element>(self) -> StateNamed<O> {
        let mut values = HashMap::with_capacity(self.values.len());

        for (key, value) in self.values {
            values.insert(key, value.convert());
        }

        StateNamed { values }
    }
}

impl<E: Element> State<E> {
    pub fn get(&self, name: &str) -> Option<&Self> {
        match self {
            State::StateNamed(named) => named.get(name),
            _ => None,
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            State::StateNamed(named) => named.is_empty(),
            State::Data(_) => false,
            State::ParamId(_) => false,
        }
    }

    pub fn convert<O: Element>(self) -> State<O> {
        match self {
            State::StateNamed(named) => State::StateNamed(named.convert()),
            State::Data(data) => State::Data(data.convert()),
            State::ParamId(id) => State::ParamId(id),
        }
    }
}

impl<E: Element> State<E>
where
    E: serde::de::DeserializeOwned,
    E: serde::Serialize,
{
    pub fn save(self, file: &str) -> std::io::Result<()> {
        let path = Path::new(file);
        if path.exists() {
            log::info!("File exists, replacing");
            std::fs::remove_file(path).unwrap();
        }

        let writer = File::create(path)?;
        let writer = GzEncoder::new(writer, Compression::default());
        serde_json::to_writer(writer, &self).unwrap();

        Ok(())
    }

    pub fn load(file: &str) -> Result<Self, StateError> {
        let path = Path::new(file);
        let reader =
            File::open(path).map_err(|err| StateError::FileNotFound(format!("{err:?}")))?;
        let reader = GzDecoder::new(reader);
        let state = serde_json::from_reader(reader).unwrap();

        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::{list_param_ids, Module};
    use crate::tensor::backend::Backend;
    use crate::{nn, TestBackend};

    #[test]
    fn test_state_to_from_value() {
        let model = create_model();
        let state = model.state();
        let bytes = serde_json::to_vec(&state).unwrap();

        let state_from: State<<crate::TestBackend as Backend>::Elem> =
            serde_json::from_slice(&bytes).unwrap();

        assert_eq!(state, state_from);
    }

    #[test]
    fn test_can_save_and_load_from_file() {
        let model_before = create_model();
        let state_before = model_before.state();
        state_before.clone().save("/tmp/test.json").unwrap();

        let mut model_after = create_model();
        model_after
            .load(&State::load("/tmp/test.json").unwrap())
            .unwrap();

        let state_after = model_after.state();
        assert_eq!(state_before, state_after);
    }

    #[test]
    fn test_parameter_ids_are_loaded() {
        let model_1 = create_model();
        let mut model_2 = create_model();
        let params_before_1 = list_param_ids(&model_1);
        let params_before_2 = list_param_ids(&model_2);

        let state = model_1.state();
        model_2.load(&state).unwrap();
        let params_after_2 = list_param_ids(&model_2);

        assert_ne!(params_before_1, params_before_2);
        assert_eq!(params_before_1, params_after_2);
    }

    fn create_model() -> nn::Linear<TestBackend> {
        nn::Linear::<crate::TestBackend>::new(&nn::LinearConfig {
            d_input: 32,
            d_output: 32,
            bias: true,
        })
    }
}
