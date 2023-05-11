use crate::InMemDataset;
use dirs::home_dir;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::Hasher;
use std::process::Command;
use thiserror::Error;

const PYTHON: &str = "python3";

const PYTHON_SOURCE: &str = include_str!("dataset.py");

#[derive(Error, Debug)]
pub enum DownloaderError {
    #[error("unknown: `{0}`")]
    Unknown(String),
    #[error("fail to download python dependencies: `{0}`")]
    FailToDownloadPythonDependencies(String),
}

/// Load datasets from [huggingface datasets](https://huggingface.co/datasets).
pub struct HuggingfaceDatasetLoader {
    name: String,
    split: String,
    extractors: Vec<Extractor>,
    config: Vec<String>,
    config_named: Vec<(String, String)>,
    deps: Vec<String>,
}

impl HuggingfaceDatasetLoader {
    /// Create a huggingface dataset loader.
    pub fn new(name: &str, split: &str) -> Self {
        Self {
            name: name.to_string(),
            split: split.to_string(),
            extractors: Vec::new(),
            config: Vec::new(),
            config_named: Vec::new(),
            deps: Vec::new(),
        }
    }

    pub fn config(mut self, config: &str) -> Self {
        self.config.push(config.to_string());
        self
    }

    pub fn config_named(mut self, name: &str, config: &str) -> Self {
        self.config_named
            .push((name.to_string(), config.to_string()));
        self
    }

    pub fn deps(mut self, deps: &[&str]) -> Self {
        self.deps
            .append(&mut deps.iter().copied().map(String::from).collect());
        self
    }

    pub fn dep(mut self, dep: &str) -> Self {
        self.deps.push(dep.to_string());
        self
    }

    pub fn extract_image(mut self, field_name: &str) -> Self {
        self.extractors
            .push(Extractor::Image(field_name.to_string()));
        self
    }

    pub fn extract_number(self, field_name: &str) -> Self {
        self.extract_raw(field_name)
    }

    pub fn extract_string(self, field_name: &str) -> Self {
        self.extract_raw(field_name)
    }

    pub fn load_in_memory<I: serde::de::DeserializeOwned + Clone>(
        self,
    ) -> Result<InMemDataset<I>, DownloaderError> {
        let path_file = self.load_file()?;
        let dataset = InMemDataset::from_file(path_file.as_str()).unwrap();

        Ok(dataset)
    }

    pub fn load_file(self) -> Result<String, DownloaderError> {
        let mut hasher = DefaultHasher::new();
        hasher.write(format!("{:?}", self.extractors).as_bytes());
        hasher.write(format!("{:?}", self.config).as_bytes());
        hasher.write(format!("{:?}", self.config_named).as_bytes());
        let hash = hasher.finish();

        let base_file = format!("{}/{}-{}", cache_dir(), self.name, hash);
        let path_file = format!("{}-{}", base_file, self.split);

        if !std::path::Path::new(&path_file).exists() {
            download(
                self.name.clone(),
                vec![self.split],
                base_file,
                self.extractors,
                self.config,
                self.config_named,
                &self.deps,
            )?;
        }

        Ok(path_file)
    }

    fn extract_raw(mut self, field_name: &str) -> Self {
        self.extractors.push(Extractor::Raw(field_name.to_string()));
        self
    }
}

fn download(
    name: String,
    splits: Vec<String>,
    base_file: String,
    extractors: Vec<Extractor>,
    config: Vec<String>,
    config_named: Vec<(String, String)>,
    deps: &[String],
) -> Result<(), DownloaderError> {
    download_python_deps(deps)?;

    let mut command = Command::new(PYTHON);

    command.arg(dataset_downloader_file_path());

    command.arg("--file");
    command.arg(base_file);

    command.arg("--name");
    command.arg(name);

    command.arg("--split");
    for split in splits {
        command.arg(split);
    }

    let mut extracted_raw = Vec::new();
    let mut extracted_images = Vec::new();

    for extractor in extractors {
        match extractor {
            Extractor::Raw(field) => extracted_raw.push(field),
            Extractor::Image(field) => extracted_images.push(field),
        };
    }

    if !extracted_raw.is_empty() {
        command.arg("--extract-raw");
        for field in extracted_raw {
            command.arg(field);
        }
    }

    if !extracted_images.is_empty() {
        command.arg("--extract-image");
        for field in extracted_images {
            command.arg(field);
        }
    }

    if !config.is_empty() {
        command.arg("--config");
        for config in config {
            command.arg(config);
        }
    }
    if !config_named.is_empty() {
        command.arg("--config-named");
        for (key, value) in config_named {
            command.arg(format!("{key}={value}"));
        }
    }

    let mut handle = command.spawn().unwrap();
    handle
        .wait()
        .map_err(|err| DownloaderError::Unknown(format!("{err:?}")))?;

    Ok(())
}

fn cache_dir() -> String {
    let home_dir = home_dir().unwrap();
    let home_dir = home_dir.to_str().map(|s| s.to_string());
    let home_dir = home_dir.unwrap();
    let cache_dir = format!("{home_dir}/.cache/burn-dataset");
    std::fs::create_dir_all(&cache_dir).ok();
    cache_dir
}

fn dataset_downloader_file_path() -> String {
    let path_dir = cache_dir();
    let path_file = format!("{path_dir}/dataset.py");

    fs::write(path_file.as_str(), PYTHON_SOURCE).expect("Write python dataset downloader");
    path_file
}

fn download_python_deps(deps: &[String]) -> Result<(), DownloaderError> {
    let mut command = Command::new(PYTHON);

    command
        .args(["-m", "pip", "install", "datasets"])
        .args(deps);

    command
        .spawn()
        .map_err(|err| {
            DownloaderError::FailToDownloadPythonDependencies(format!(
                "{} | error: {}",
                deps.to_vec().join(", "),
                err
            ))
        })?
        .wait()
        .map_err(|err| {
            DownloaderError::FailToDownloadPythonDependencies(format!(
                "{} | error: {}",
                deps.to_vec().join(", "),
                err
            ))
        })?;

    Ok(())
}

#[derive(Debug)]
enum Extractor {
    Raw(String),
    Image(String),
}
