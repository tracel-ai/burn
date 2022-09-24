pub use burn_derive::Config;

#[derive(Debug)]
pub enum ConfigError {
    InvalidFormat(String),
    FileNotFound(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut message = "Config error => ".to_string();

        match self {
            Self::InvalidFormat(err) => {
                message += format!("Invalid format: {}", err).as_str();
            }
            Self::FileNotFound(err) => {
                message += format!("File not found: {}", err).as_str();
            }
        };

        f.write_str(message.as_str())
    }
}
impl std::error::Error for ConfigError {}

pub trait Config: serde::Serialize + serde::de::DeserializeOwned {
    fn save(&self, file: &str) -> std::io::Result<()> {
        std::fs::write(file, config_to_json(self))
    }

    fn load(file: &str) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(file)
            .map_err(|_| ConfigError::FileNotFound(file.to_string()))?;
        config_from_str(&content)
    }
}

pub fn config_to_json<C: Config>(config: &C) -> String {
    serde_json::to_string_pretty(config).unwrap()
}

fn config_from_str<C: Config>(content: &str) -> Result<C, ConfigError> {
    serde_json::from_str(content).map_err(|err| ConfigError::InvalidFormat(format!("{}", err)))
}
