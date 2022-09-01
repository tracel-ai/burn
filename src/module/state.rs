use crate::tensor::{back, DataSerialize};
use std::collections::HashMap;

#[derive(Debug, PartialEq)]
pub struct StateNamed<B: back::Backend> {
    pub values: HashMap<String, State<B>>,
}

#[derive(Debug, PartialEq)]
pub enum State<B: back::Backend> {
    StateNamed(StateNamed<B>),
    Data(DataSerialize<B::Elem>),
}

#[derive(Debug)]
pub enum StateError {
    InvalidFormat(String),
    FileNotFound(String),
}

impl<B: back::Backend> Into<serde_json::Value> for State<B>
where
    B::Elem: serde::de::DeserializeOwned,
    B::Elem: serde::Serialize,
{
    fn into(self) -> serde_json::Value {
        match self {
            Self::StateNamed(state) => state.into(),
            Self::Data(data) => serde_json::to_value(data).unwrap(),
        }
    }
}

impl<B: back::Backend> Into<serde_json::Value> for StateNamed<B>
where
    B::Elem: serde::de::DeserializeOwned,
    B::Elem: serde::Serialize,
{
    fn into(self) -> serde_json::Value {
        let mut map = serde_json::Map::new();

        for (key, state) in self.values {
            map.insert(key, state.into());
        }

        serde_json::Value::Object(map)
    }
}

impl<B: back::Backend> TryFrom<serde_json::Value> for State<B>
where
    B::Elem: serde::de::DeserializeOwned,
    B::Elem: serde::Serialize,
{
    type Error = StateError;

    fn try_from(value: serde_json::Value) -> Result<Self, Self::Error> {
        match serde_json::from_value(value.clone()) {
            Ok(data) => Ok(State::Data(data)),
            Err(_) => Ok(State::StateNamed(StateNamed::try_from(value)?)),
        }
    }
}

impl<B: back::Backend> TryFrom<serde_json::Value> for StateNamed<B>
where
    B::Elem: serde::de::DeserializeOwned,
    B::Elem: serde::Serialize,
{
    type Error = StateError;

    fn try_from(value: serde_json::Value) -> Result<Self, Self::Error> {
        let map = match value {
            serde_json::Value::Object(map) => map,
            _ => {
                return Err(StateError::InvalidFormat(format!(
                    "Invalid value {:?}",
                    value
                )))
            }
        };

        let mut values = HashMap::new();
        for (key, value) in map {
            values.insert(key, State::try_from(value)?);
        }

        Ok(Self { values })
    }
}

impl<B: back::Backend> StateNamed<B> {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    pub fn register_state(&mut self, name: &str, state: State<B>) {
        self.values.insert(name.to_string(), state);
    }
}

impl<B: back::Backend> StateNamed<B> {
    pub fn get(&self, name: &str) -> &State<B> {
        self.values.get(name).unwrap()
    }
}

impl<B: back::Backend> State<B> {
    pub fn get(&self, name: &str) -> &Self {
        match self {
            State::StateNamed(named) => named.get(name),
            _ => panic!("Can't"),
        }
    }
}

impl<B: back::Backend> State<B>
where
    B::Elem: serde::de::DeserializeOwned,
    B::Elem: serde::Serialize,
{
    pub fn save(self, file: &str) -> std::io::Result<()> {
        let value: serde_json::Value = self.into();
        std::fs::write(file, value.to_string())
    }

    pub fn load(file: &str) -> Result<Self, StateError> {
        let value = std::fs::read_to_string(file)
            .map_err(|err| StateError::FileNotFound(format!("{:?}", err)))?;
        let value: serde_json::Value = serde_json::from_str(&value)
            .map_err(|err| StateError::InvalidFormat(format!("{:?}", err)))?;

        Self::try_from(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::Module;
    use crate::nn;

    #[test]
    fn test_state_to_from_value() {
        let linear = nn::Linear::<crate::TestBackend>::new(&nn::LinearConfig {
            d_input: 32,
            d_output: 32,
            bias: true,
        });

        let state = linear.state();
        let value: serde_json::Value = state.into();
        let state_from: State<crate::TestBackend> = State::try_from(value.clone()).unwrap();
        let value_from: serde_json::Value = state_from.into();

        assert_eq!(value, value_from);
    }

    #[test]
    fn test_can_save_and_load_from_file() {
        let mut linear = nn::Linear::<crate::TestBackend>::new(&nn::LinearConfig {
            d_input: 32,
            d_output: 32,
            bias: true,
        });
        linear.state().save("/tmp/test.json").unwrap();
        linear.load(&State::load("/tmp/test.json").unwrap())
    }
}
