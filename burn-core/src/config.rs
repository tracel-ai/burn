use alloc::{format, string::String, string::ToString};
pub use burn_derive::Config;

#[derive(Debug)]
pub enum ConfigError {
    InvalidFormat(String),
    FileNotFound(String),
}

impl core::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut message = "Config error => ".to_string();

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

// TODO: Move from std to core after Error is core (see https://github.com/rust-lang/rust/issues/103765)
#[cfg(feature = "std")]
impl std::error::Error for ConfigError {}

pub trait Config: serde::Serialize + serde::de::DeserializeOwned {
    #[cfg(feature = "std")]
    fn save(&self, file: &str) -> std::io::Result<()> {
        std::fs::write(file, config_to_json(self))
    }

    #[cfg(feature = "std")]
    fn load(file: &str) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(file)
            .map_err(|_| ConfigError::FileNotFound(file.to_string()))?;
        config_from_str(&content)
    }

    fn load_binary(data: &[u8]) -> Result<Self, ConfigError> {
        let content = core::str::from_utf8(data).map_err(|_| {
            ConfigError::InvalidFormat("Could not parse data as utf-8.".to_string())
        })?;
        config_from_str(content)
    }
}

pub fn config_to_json<C: Config>(config: &C) -> String {
    serde_json::to_string_pretty(config).unwrap()
}

fn config_from_str<C: Config>(content: &str) -> Result<C, ConfigError> {
    serde_json::from_str(content).map_err(|err| ConfigError::InvalidFormat(format!("{err}")))
}
