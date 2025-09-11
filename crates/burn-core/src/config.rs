use alloc::{format, string::String, string::ToString};
pub use burn_derive::Config;
use core::fmt::Debug;

/// Configuration IO error.
#[derive(Debug)]
pub enum ConfigError {
    /// Invalid format.
    InvalidFormat(String),

    /// File not found.
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

impl core::error::Error for ConfigError {}

/// Configuration trait.
pub trait Config: Debug + serde::Serialize + serde::de::DeserializeOwned {
    /// Saves the configuration to a file.
    ///
    /// # Arguments
    ///
    /// * `file` - File to save the configuration to.
    ///
    /// # Returns
    ///
    /// The output of the save operation.
    #[cfg(feature = "std")]
    fn save<P: AsRef<std::path::Path>>(&self, file: P) -> std::io::Result<()> {
        std::fs::write(file, config_to_json(self))
    }

    /// Loads the configuration from a file.
    ///
    /// # Arguments
    ///
    /// * `file` - File to load the configuration from.
    ///
    /// # Returns
    ///
    /// The loaded configuration.
    #[cfg(feature = "std")]
    fn load<P: AsRef<std::path::Path>>(file: P) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(file.as_ref())
            .map_err(|_| ConfigError::FileNotFound(file.as_ref().to_string_lossy().to_string()))?;
        config_from_str(&content)
    }

    /// Loads the configuration from a binary buffer.
    ///
    /// # Arguments
    ///
    /// * `data` - Binary buffer to load the configuration from.
    ///
    /// # Returns
    ///
    /// The loaded configuration.
    fn load_binary(data: &[u8]) -> Result<Self, ConfigError> {
        let content = core::str::from_utf8(data).map_err(|_| {
            ConfigError::InvalidFormat("Could not parse data as utf-8.".to_string())
        })?;
        config_from_str(content)
    }
}

/// Converts a configuration to a JSON string.
///
/// # Arguments
///
/// * `config` - Configuration to convert.
///
/// # Returns
///
/// The JSON string.
pub fn config_to_json<C: Config>(config: &C) -> String {
    serde_json::to_string_pretty(config).unwrap()
}

fn config_from_str<C: Config>(content: &str) -> Result<C, ConfigError> {
    serde_json::from_str(content).map_err(|err| ConfigError::InvalidFormat(format!("{err}")))
}
