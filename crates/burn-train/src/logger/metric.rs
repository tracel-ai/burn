use super::{AsyncLogger, FileLogger, InMemoryLogger, Logger};
use crate::metric::{
    MetricDefinition, MetricEntry, NumericEntry,
    store::{EpochSummary, Split},
};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

const EPOCH_PREFIX: &str = "epoch-";

/// Metric logger.
pub trait MetricLogger: Send {
    /// Logs an item.
    ///
    /// # Arguments
    ///
    /// * `items` - List of items.
    /// * `epoch` - Current epoch.
    /// * `split` - Current dataset split.
    /// * `iteration` - Current iteration.
    fn log(&mut self, items: Vec<&MetricEntry>, epoch: usize, split: Split, iteration: usize);

    /// Read the logs for an epoch.
    fn read_numeric(
        &mut self,
        name: &str,
        epoch: usize,
        split: Split,
    ) -> Result<Vec<NumericEntry>, String>;

    /// Logs the metric definition information (name, description, unit, etc.)
    fn log_metric_definition(&self, definition: MetricDefinition);

    /// Logs summary of the epoch (duration, highest metric values reached, etc.)
    fn log_epoch_summary(&mut self, summary: EpochSummary);
}

/// The file metric logger.
pub struct FileMetricLogger {
    loggers: HashMap<String, AsyncLogger<String>>,
    directory: PathBuf,
    is_eval: bool,
}

impl FileMetricLogger {
    /// Create a new file metric logger.
    ///
    /// # Arguments
    ///
    /// * `directory` - The directory.
    ///
    /// # Returns
    ///
    /// The file metric logger.
    pub fn new(directory: impl AsRef<Path>) -> Self {
        Self {
            loggers: HashMap::new(),
            directory: directory.as_ref().to_path_buf(),
            is_eval: false,
        }
    }

    /// Create a new file metric logger.
    ///
    /// # Arguments
    ///
    /// * `directory` - The directory.
    ///
    /// # Returns
    ///
    /// The file metric logger.
    pub fn new_eval(directory: impl AsRef<Path>) -> Self {
        Self {
            loggers: HashMap::new(),
            directory: directory.as_ref().to_path_buf(),
            is_eval: true,
        }
    }

    pub(crate) fn split_exists(&self, split: Split) -> bool {
        let split_path = self.directory.join(split.to_string());
        split_path.exists() && split_path.is_dir()
    }

    /// Number of epochs recorded.
    pub(crate) fn epochs(&self) -> usize {
        if self.is_eval {
            log::warn!("Number of epochs not available when testing.");
            return 0;
        }

        let mut max_epoch = 0;

        // with split
        for path in fs::read_dir(&self.directory).unwrap() {
            let path = path.unwrap();

            if fs::metadata(path.path()).unwrap().is_dir() {
                for split_path in fs::read_dir(path.path()).unwrap() {
                    let split_path = split_path.unwrap();

                    if fs::metadata(split_path.path()).unwrap().is_dir() {
                        let dir_name = split_path.file_name().into_string().unwrap();

                        if !dir_name.starts_with(EPOCH_PREFIX) {
                            continue;
                        }

                        let epoch = dir_name.replace(EPOCH_PREFIX, "").parse::<usize>().ok();

                        if let Some(epoch) = epoch
                            && epoch > max_epoch
                        {
                            max_epoch = epoch;
                        }
                    }
                }
            }
        }

        max_epoch
    }

    fn train_directory(&self, tags: Option<&String>, epoch: usize, split: Split) -> PathBuf {
        let name = format!("{EPOCH_PREFIX}{epoch}");

        match tags {
            Some(tags) => self.directory.join(split.to_string()).join(tags).join(name),
            None => self.directory.join(split.to_string()).join(name),
        }
    }

    fn eval_directory(&self, tags: Option<&String>, split: Split) -> PathBuf {
        match tags {
            Some(tags) => self.directory.join(split.to_string()).join(tags),
            None => self.directory.clone(),
        }
    }

    fn file_path(
        &self,
        tags: Option<&String>,
        name: &str,
        epoch: Option<usize>,
        split: Split,
    ) -> PathBuf {
        let directory = match epoch {
            Some(epoch) => self.train_directory(tags, epoch, split),
            None => self.eval_directory(tags, split),
        };
        let name = name.replace(' ', "_");
        let name = format!("{name}.log");
        directory.join(name)
    }

    fn create_directory(&self, tags: Option<&String>, epoch: Option<usize>, split: Split) {
        let directory = match epoch {
            Some(epoch) => self.train_directory(tags, epoch, split),
            None => self.eval_directory(tags, split),
        };
        std::fs::create_dir_all(directory).ok();
    }
}

impl FileMetricLogger {
    fn log_item(
        &mut self,
        tags: Option<&String>,
        item: &MetricEntry,
        epoch: Option<usize>,
        split: Split,
    ) {
        let key = logger_key(&item.name, split);
        let value = &item.serialize;

        let logger = match self.loggers.get_mut(&key) {
            Some(val) => val,
            None => {
                self.create_directory(tags, epoch, split);

                let file_path = self.file_path(tags, &item.name, epoch, split);
                let logger = FileLogger::new(file_path);
                let logger = AsyncLogger::new(logger);

                self.loggers.insert(key.clone(), logger);
                self.loggers
                    .get_mut(&key)
                    .expect("Can get the previously saved logger.")
            }
        };

        logger.log(value.clone());
    }

    fn log_tags(&mut self, item: &MetricEntry, epoch: Option<usize>, split: Split) {
        let mut tags = String::new();
        item.tags.iter().for_each(|tag| tags += tag.as_str());
        let tags = tags.replace(" ", "-").trim().to_lowercase();
        self.log_item(Some(&tags), item, epoch, split);
    }
}

impl MetricLogger for FileMetricLogger {
    fn log(&mut self, items: Vec<&MetricEntry>, epoch: usize, split: Split, _iteration: usize) {
        for item in items.iter() {
            match item.tags.is_empty() {
                true => self.log_item(None, item, Some(epoch), split),
                false => self.log_tags(item, Some(epoch), split),
            }
        }
    }

    fn read_numeric(
        &mut self,
        name: &str,
        epoch: usize,
        split: Split,
    ) -> Result<Vec<NumericEntry>, String> {
        if let Some(value) = self.loggers.get(name) {
            value.sync()
        }

        let file_path = self.file_path(None, name, Some(epoch), split);

        let mut errors = false;

        let data = std::fs::read_to_string(file_path)
            .unwrap_or_default()
            .split('\n')
            .filter_map(|value| {
                if value.is_empty() {
                    None
                } else {
                    match NumericEntry::deserialize(value) {
                        Ok(value) => Some(value),
                        Err(err) => {
                            log::error!("{err}");
                            errors = true;
                            None
                        }
                    }
                }
            })
            .collect();

        if errors {
            Err("Parsing numeric entry errors".to_string())
        } else {
            Ok(data)
        }
    }

    fn log_metric_definition(&self, _definition: MetricDefinition) {}

    fn log_epoch_summary(&mut self, _summary: EpochSummary) {
        if !self.is_eval {
            self.loggers.clear();
        }
    }
}

fn logger_key(name: &str, split: Split) -> String {
    format!("{name}_{split}")
}

/// In memory metric logger, useful when testing and debugging.
#[derive(Default)]
pub struct InMemoryMetricLogger {
    values: HashMap<String, Vec<InMemoryLogger>>,
}

impl InMemoryMetricLogger {
    /// Create a new in-memory metric logger.
    pub fn new() -> Self {
        Self::default()
    }
}

impl MetricLogger for InMemoryMetricLogger {
    fn log(&mut self, items: Vec<&MetricEntry>, _epoch: usize, split: Split, _iteration: usize) {
        for item in items.iter() {
            let key = logger_key(&item.name, split);

            if !self.values.contains_key(&key) {
                self.values
                    .insert(key.to_string(), vec![InMemoryLogger::default()]);
            }

            let values = self.values.get_mut(&key).unwrap();

            values.last_mut().unwrap().log(item.serialize.clone());
        }
    }

    fn read_numeric(
        &mut self,
        name: &str,
        epoch: usize,
        split: Split,
    ) -> Result<Vec<NumericEntry>, String> {
        let key = logger_key(name, split);
        let values = match self.values.get(&key) {
            Some(values) => values,
            None => return Ok(Vec::new()),
        };

        match values.get(epoch - 1) {
            Some(logger) => Ok(logger
                .values
                .iter()
                .filter_map(|value| NumericEntry::deserialize(value).ok())
                .collect()),
            None => Ok(Vec::new()),
        }
    }

    fn log_metric_definition(&self, _definition: MetricDefinition) {}

    fn log_epoch_summary(&mut self, _summary: EpochSummary) {
        self.values
            .values_mut()
            .for_each(|loggers| loggers.push(InMemoryLogger::default()));
    }
}
