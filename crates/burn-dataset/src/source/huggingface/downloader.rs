use std::fs::{self, create_dir_all};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::{SqliteDataset, SqliteDatasetError, SqliteDatasetStorage};

use sanitize_filename::sanitize;
use serde::de::DeserializeOwned;
use thiserror::Error;

const PYTHON_SOURCE: &str = include_str!("importer.py");
#[cfg(not(target_os = "windows"))]
const VENV_BIN_PYTHON: &str = "bin/python3";
#[cfg(target_os = "windows")]
const VENV_BIN_PYTHON: &str = "Scripts\\python";

/// Error type for [HuggingfaceDatasetLoader](HuggingfaceDatasetLoader).
#[derive(Error, Debug)]
pub enum ImporterError {
    /// Unknown error.
    #[error("unknown: `{0}`")]
    Unknown(String),

    /// Fail to download python dependencies.
    #[error("fail to download python dependencies: `{0}`")]
    FailToDownloadPythonDependencies(String),

    /// Fail to create sqlite dataset.
    #[error("sqlite dataset: `{0}`")]
    SqliteDataset(#[from] SqliteDatasetError),

    /// python3 is not installed.
    #[error("python3 is not installed")]
    PythonNotInstalled,

    /// venv environment is not initialized.
    #[error("venv environment is not initialized")]
    VenvNotInitialized,
}

/// Load a dataset from [huggingface datasets](https://huggingface.co/datasets).
///
/// The dataset with all splits is stored in a single sqlite database (see [SqliteDataset](SqliteDataset)).
///
/// # Example
/// ```no_run
///  use burn_dataset::HuggingfaceDatasetLoader;
///  use burn_dataset::SqliteDataset;
///  use serde::{Deserialize, Serialize};
///
/// #[derive(Deserialize, Debug, Clone)]
/// struct MnistItemRaw {
///     pub image_bytes: Vec<u8>,
///     pub label: usize,
/// }
///
///  let train_ds:SqliteDataset<MnistItemRaw> = HuggingfaceDatasetLoader::new("mnist")
///       .dataset("train")
///       .unwrap();
pub struct HuggingfaceDatasetLoader {
    name: String,
    subset: Option<String>,
    base_dir: Option<PathBuf>,
    huggingface_token: Option<String>,
    huggingface_cache_dir: Option<String>,
    huggingface_data_dir: Option<String>,
    trust_remote_code: bool,
}

impl HuggingfaceDatasetLoader {
    /// Create a huggingface dataset loader.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            subset: None,
            base_dir: None,
            huggingface_token: None,
            huggingface_cache_dir: None,
            huggingface_data_dir: None,
            trust_remote_code: false,
        }
    }

    /// Create a huggingface dataset loader for a subset of the dataset.
    ///
    /// The subset name must be one of the subsets listed in the dataset page.
    ///
    /// If no subset names are listed, then do not use this method.
    pub fn with_subset(mut self, subset: &str) -> Self {
        self.subset = Some(subset.to_string());
        self
    }

    /// Specify a base directory to store the dataset.
    ///
    /// If not specified, the dataset will be stored in `~/.cache/burn-dataset`.
    pub fn with_base_dir(mut self, base_dir: &str) -> Self {
        self.base_dir = Some(base_dir.into());
        self
    }

    /// Specify a huggingface token to download datasets behind authentication.
    ///
    /// You can get a token from [tokens settings](https://huggingface.co/settings/tokens)
    pub fn with_huggingface_token(mut self, huggingface_token: &str) -> Self {
        self.huggingface_token = Some(huggingface_token.to_string());
        self
    }

    /// Specify a huggingface cache directory to store the downloaded datasets.
    ///
    /// If not specified, the dataset will be stored in `~/.cache/huggingface/datasets`.
    pub fn with_huggingface_cache_dir(mut self, huggingface_cache_dir: &str) -> Self {
        self.huggingface_cache_dir = Some(huggingface_cache_dir.to_string());
        self
    }

    /// Specify a relative path to a subset of a dataset. This is used in some datasets for the
    /// manual steps of dataset download process.
    ///
    /// Unless you've encountered a ManualDownloadError
    /// when loading your dataset you probably don't have to worry about this setting.
    pub fn with_huggingface_data_dir(mut self, huggingface_data_dir: &str) -> Self {
        self.huggingface_data_dir = Some(huggingface_data_dir.to_string());
        self
    }

    /// Specify whether or not to trust remote code.
    ///
    /// If not specified, trust remote code is set to true.
    pub fn with_trust_remote_code(mut self, trust_remote_code: bool) -> Self {
        self.trust_remote_code = trust_remote_code;
        self
    }

    /// Load the dataset.
    pub fn dataset<I: DeserializeOwned + Clone>(
        self,
        split: &str,
    ) -> Result<SqliteDataset<I>, ImporterError> {
        let db_file = self.db_file()?;
        let dataset = SqliteDataset::from_db_file(db_file, split)?;
        Ok(dataset)
    }

    /// Get the path to the sqlite database file.
    ///
    /// If the database file does not exist, it will be downloaded and imported.
    pub fn db_file(self) -> Result<PathBuf, ImporterError> {
        // determine (and create if needed) the base directory
        let base_dir = SqliteDatasetStorage::base_dir(self.base_dir);

        if !base_dir.exists() {
            create_dir_all(&base_dir).expect("Failed to create base directory");
        }

        //sanitize the name and subset
        let name = sanitize(self.name.as_str());

        // create the db file path
        let db_file_name = if let Some(subset) = self.subset.clone() {
            format!("{}-{}.db", name, sanitize(subset.as_str()))
        } else {
            format!("{}.db", name)
        };

        let db_file = base_dir.join(db_file_name);

        // import the dataset if needed
        if !Path::new(&db_file).exists() {
            import(
                self.name,
                self.subset,
                db_file.clone(),
                base_dir,
                self.huggingface_token,
                self.huggingface_cache_dir,
                self.huggingface_data_dir,
                self.trust_remote_code,
            )?;
        }

        Ok(db_file)
    }
}

/// Import a dataset from huggingface. The transformed dataset is stored as sqlite database.
#[allow(clippy::too_many_arguments)]
fn import(
    name: String,
    subset: Option<String>,
    base_file: PathBuf,
    base_dir: PathBuf,
    huggingface_token: Option<String>,
    huggingface_cache_dir: Option<String>,
    huggingface_data_dir: Option<String>,
    trust_remote_code: bool,
) -> Result<(), ImporterError> {
    let venv_python_path = install_python_deps(&base_dir)?;

    let mut command = Command::new(venv_python_path);

    command.arg(importer_script_path(&base_dir));

    command.arg("--name");
    command.arg(name);

    command.arg("--file");
    command.arg(base_file);

    if let Some(subset) = subset {
        command.arg("--subset");
        command.arg(subset);
    }

    if let Some(huggingface_token) = huggingface_token {
        command.arg("--token");
        command.arg(huggingface_token);
    }

    if let Some(huggingface_cache_dir) = huggingface_cache_dir {
        command.arg("--cache_dir");
        command.arg(huggingface_cache_dir);
    }
    if let Some(huggingface_data_dir) = huggingface_data_dir {
        command.arg("--data_dir");
        command.arg(huggingface_data_dir);
    }
    if trust_remote_code {
        command.arg("--trust_remote_code");
        command.arg("True");
    }
    let mut handle = command.spawn().unwrap();
    handle
        .wait()
        .map_err(|err| ImporterError::Unknown(format!("{err:?}")))?;

    Ok(())
}

/// check python --version output is `Python 3.x.x`
fn check_python_version_is_3(python: &str) -> bool {
    let output = Command::new(python).arg("--version").output();
    match output {
        Ok(output) => {
            if output.status.success() {
                let version_string = String::from_utf8_lossy(&output.stdout);
                if let Some(index) = version_string.find(' ') {
                    let version = &version_string[index + 1..];
                    version.starts_with("3.")
                } else {
                    false
                }
            } else {
                false
            }
        }
        Err(_error) => false,
    }
}

/// get python3 name `python` `python3` or `py`
fn get_python_name() -> Result<&'static str, ImporterError> {
    let python_name_list = ["python3", "python", "py"];
    for python_name in python_name_list.iter() {
        if check_python_version_is_3(python_name) {
            return Ok(python_name);
        }
    }
    Err(ImporterError::PythonNotInstalled)
}

fn importer_script_path(base_dir: &Path) -> PathBuf {
    let path_file = base_dir.join("importer.py");

    fs::write(&path_file, PYTHON_SOURCE).expect("Write python dataset downloader");
    path_file
}

fn install_python_deps(base_dir: &Path) -> Result<PathBuf, ImporterError> {
    let venv_dir = base_dir.join("venv");
    let venv_python_path = venv_dir.join(VENV_BIN_PYTHON);
    // If the venv environment is already initialized, skip the initialization.
    if !check_python_version_is_3(venv_python_path.to_str().unwrap()) {
        let python_name = get_python_name()?;
        let mut command = Command::new(python_name);
        command.args([
            "-m",
            "venv",
            venv_dir
                .as_os_str()
                .to_str()
                .expect("Path utf8 conversion should not fail"),
        ]);

        // Spawn the venv creation process and wait for it to complete.
        let mut handle = command.spawn().unwrap();

        handle.wait().map_err(|err| {
            ImporterError::FailToDownloadPythonDependencies(format!(" error: {}", err))
        })?;
        // Check if the venv environment can be used successfully."
        if !check_python_version_is_3(venv_python_path.to_str().unwrap()) {
            return Err(ImporterError::VenvNotInitialized);
        }
    }

    let mut command = Command::new(&venv_python_path);
    command.args([
        "-m",
        "pip",
        "--quiet",
        "install",
        "pyarrow",
        "sqlalchemy",
        "Pillow",
        "soundfile",
        "datasets",
    ]);

    // Spawn the pip install process and wait for it to complete.
    let mut handle = command.spawn().unwrap();
    handle.wait().map_err(|err| {
        ImporterError::FailToDownloadPythonDependencies(format!(" error: {}", err))
    })?;

    Ok(venv_python_path)
}
