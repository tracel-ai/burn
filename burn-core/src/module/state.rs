use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

use super::ParamId;
use crate::tensor::{DataSerialize, Element};
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;
#[cfg(feature = "std")]
use std::collections::HashMap;

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

/// All supported format.
///
/// # Notes
///
/// The default file format is compressed bincode for the smallest file size possible.
/// For `no_std` environments, you should use (StateFormat::Bin)[StateFormat::Bin] since compression isn't supported.
/// However, the bincode format alone is smaller than compressed `json` or `msgpack`.
#[derive(Default, Clone)]
pub enum StateFormat {
    #[default]
    BinGz,
    Bin,
    JsonGz,
    #[cfg(feature = "msgpack")]
    MpkGz,
}

impl core::fmt::Display for StateError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
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
    pub fn to_bin(&self) -> Result<Vec<u8>, StateError> {
        Ok(bincode::serde::encode_to_vec(self, Self::bin_config()).unwrap())
    }

    pub fn from_bin(data: &[u8]) -> Result<Self, StateError> {
        let state = bincode::serde::decode_borrowed_from_slice(data, Self::bin_config()).unwrap();
        Ok(state)
    }

    fn bin_config() -> bincode::config::Configuration {
        bincode::config::standard()
    }
}

#[cfg(feature = "std")]
mod std_enabled {
    use super::*;
    use flate2::{read::GzDecoder, write::GzEncoder, Compression};
    use std::{fs::File, path::Path};

    // TODO: Move from std to core after Error is core (see https://github.com/rust-lang/rust/issues/103765)
    impl std::error::Error for StateError {}

    macro_rules! str2reader {
        (
        $file:expr,
        $ext:expr
    ) => {{
            let path_ref = &format!("{}.{}", $file, $ext);
            let path = Path::new(path_ref);

            File::open(path).map_err(|err| StateError::FileNotFound(format!("{err:?}")))
        }};
    }

    macro_rules! str2writer {
        (
        $file:expr,
        $ext:expr
    ) => {{
            let path_ref = &format!("{}.{}", $file, $ext);
            let path = Path::new(path_ref);
            if path.exists() {
                log::info!("File exists, replacing");
                std::fs::remove_file(path).unwrap();
            }

            File::create(path)
        }};
    }
    impl<E: Element> State<E>
    where
        E: serde::de::DeserializeOwned,
        E: serde::Serialize,
    {
        /// Save the state to the provided file path using the given [StateFormat](StateFormat).
        ///
        /// # Notes
        ///
        /// The file extension will be added automatically depending on the state format.
        pub fn save(self, file: &str, format: &StateFormat) -> std::io::Result<()> {
            match format {
                StateFormat::BinGz => self.save_bingz(file),
                StateFormat::Bin => self.save_bin(file),
                StateFormat::JsonGz => self.save_jsongz(file),
                #[cfg(feature = "msgpack")]
                StateFormat::MpkGz => self.save_mpkgz(file),
            }
        }

        /// Load the state from the provided file path using the given [StateFormat](StateFormat).
        ///
        /// # Notes
        ///
        /// The file extension will be added automatically depending on the state format.
        pub fn load(file: &str, format: &StateFormat) -> Result<Self, StateError> {
            match format {
                StateFormat::BinGz => Self::load_bingz(file),
                StateFormat::Bin => Self::load_bin(file),
                StateFormat::JsonGz => Self::load_jsongz(file),
                #[cfg(feature = "msgpack")]
                StateFormat::MpkGz => Self::load_mpkgz(file),
            }
        }

        fn save_jsongz(self, file: &str) -> std::io::Result<()> {
            let writer = str2writer!(file, "json.gz")?;
            let writer = GzEncoder::new(writer, Compression::default());
            serde_json::to_writer(writer, &self).unwrap();

            Ok(())
        }

        fn load_jsongz(file: &str) -> Result<Self, StateError> {
            let reader = str2reader!(file, "json.gz")?;
            let reader = GzDecoder::new(reader);
            let state = serde_json::from_reader(reader).unwrap();

            Ok(state)
        }

        #[cfg(feature = "msgpack")]
        fn save_mpkgz(self, file: &str) -> std::io::Result<()> {
            let writer = str2writer!(file, "mpk.gz")?;
            let mut writer = GzEncoder::new(writer, Compression::default());
            rmp_serde::encode::write(&mut writer, &self).unwrap();

            Ok(())
        }

        #[cfg(feature = "msgpack")]
        fn load_mpkgz(file: &str) -> Result<Self, StateError> {
            let reader = str2reader!(file, "mpk.gz")?;
            let reader = GzDecoder::new(reader);
            let state = rmp_serde::decode::from_read(reader).unwrap();

            Ok(state)
        }

        fn save_bingz(self, file: &str) -> std::io::Result<()> {
            let config = Self::bin_config();
            let writer = str2writer!(file, "bin.gz")?;
            let mut writer = GzEncoder::new(writer, Compression::default());

            bincode::serde::encode_into_std_write(&self, &mut writer, config).unwrap();

            Ok(())
        }

        fn load_bingz(file: &str) -> Result<Self, StateError> {
            let reader = str2reader!(file, "bin.gz")?;
            let mut reader = GzDecoder::new(reader);
            let state =
                bincode::serde::decode_from_std_read(&mut reader, Self::bin_config()).unwrap();

            Ok(state)
        }

        fn save_bin(self, file: &str) -> std::io::Result<()> {
            let buf = bincode::serde::encode_to_vec(self, Self::bin_config()).unwrap();

            let mut writer = str2writer!(file, "bin")?;
            std::io::Write::write_all(&mut writer, &buf).unwrap();

            Ok(())
        }

        fn load_bin(file: &str) -> Result<Self, StateError> {
            let mut reader = str2reader!(file, "bin")?;
            let mut buf = Vec::new();
            std::io::Read::read_to_end(&mut reader, &mut buf).unwrap();
            let state =
                bincode::serde::decode_borrowed_from_slice(&buf, Self::bin_config()).unwrap();

            Ok(state)
        }
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

        let state_from: State<<crate::TestBackend as Backend>::FloatElem> =
            serde_json::from_slice(&bytes).unwrap();

        assert_eq!(state, state_from);
    }

    #[test]
    fn test_parameter_ids_are_loaded() {
        let model_1 = create_model();
        let mut model_2 = create_model();
        let params_before_1 = list_param_ids(&model_1);
        let params_before_2 = list_param_ids(&model_2);

        let state = model_1.state();
        model_2 = model_2.load(&state).unwrap();
        let params_after_2 = list_param_ids(&model_2);

        assert_ne!(params_before_1, params_before_2);
        assert_eq!(params_before_1, params_after_2);
    }

    #[test]
    fn test_from_to_binary() {
        let model_1 = create_model();
        let model_2 = create_model();
        let params_before_1 = list_param_ids(&model_1);
        let params_before_2 = list_param_ids(&model_2);

        // To & From Bytes
        let bytes = model_1.state().to_bin().unwrap();
        let model_2 = model_2.load(&State::from_bin(&bytes).unwrap()).unwrap();

        // Verify.
        let params_after_2 = list_param_ids(&model_2);
        assert_ne!(params_before_1, params_before_2);
        assert_eq!(params_before_1, params_after_2);
    }

    pub fn create_model() -> nn::Linear<TestBackend> {
        nn::Linear::<crate::TestBackend>::new(&nn::LinearConfig::new(32, 32).with_bias(true))
    }
}

#[cfg(all(test, feature = "std"))]
mod tests_save_load {
    use super::tests::create_model;
    use super::*;
    use crate::module::Module;

    static FILE_PATH: &str = "/tmp/test_state";

    #[test]
    fn test_can_save_and_load_from_file_jsongz_format() {
        test_can_save_and_load_from_file(StateFormat::JsonGz)
    }

    #[test]
    fn test_can_save_and_load_from_file_bin_format() {
        test_can_save_and_load_from_file(StateFormat::Bin)
    }

    #[test]
    fn test_can_save_and_load_from_file_bingz_format() {
        test_can_save_and_load_from_file(StateFormat::BinGz)
    }

    #[cfg(feature = "msgpack")]
    #[test]
    fn test_can_save_and_load_from_file_mpkgz_format() {
        test_can_save_and_load_from_file(StateFormat::MpkGz)
    }

    #[test]
    fn test_from_bin_on_disk() {
        let model = create_model();
        model
            .state()
            .save("/tmp/model_compare", &StateFormat::Bin)
            .unwrap();
        let bytes = std::fs::read("/tmp/model_compare.bin").unwrap();
        let state = State::from_bin(&bytes).unwrap();

        assert_eq!(state, model.state());
    }

    fn test_can_save_and_load_from_file(format: StateFormat) {
        let model_before = create_model();
        let state_before = model_before.state();
        state_before.clone().save(FILE_PATH, &format).unwrap();

        let model_after = create_model()
            .load(&State::load(FILE_PATH, &format).unwrap())
            .unwrap();

        let state_after = model_after.state();
        assert_eq!(state_before, state_after);
    }
}
