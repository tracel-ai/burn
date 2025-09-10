use super::{AsyncLogger, FileLogger, InMemoryLogger, Logger};
use crate::metric::{MetricEntry, NumericEntry};
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
    /// * `item` - The item.
    fn log(&mut self, item: &MetricEntry);

    /// Logs an epoch.
    ///
    /// # Arguments
    ///
    /// * `epoch` - The epoch.
    fn end_epoch(&mut self, epoch: usize);

    /// Read the logs for an epoch.
    fn read_numeric(&mut self, name: &str, epoch: usize) -> Result<Vec<NumericEntry>, String>;
}

/// The file metric logger.
pub struct FileMetricLogger {
    loggers: HashMap<String, AsyncLogger<String>>,
    directory: PathBuf,
    epoch: Option<usize>,
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
    pub fn new_train(directory: impl AsRef<Path>) -> Self {
        Self {
            loggers: HashMap::new(),
            directory: directory.as_ref().to_path_buf(),
            epoch: Some(1),
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
            epoch: None,
        }
    }

    /// Number of epochs recorded.
    pub(crate) fn epochs(&self) -> usize {
        if self.epoch.is_none() {
            log::warn!("Number of epochs not available when testing.");
            return 0;
        }

        let mut max_epoch = 0;

        for path in fs::read_dir(&self.directory).unwrap() {
            let path = path.unwrap();

            if fs::metadata(path.path()).unwrap().is_dir() {
                let dir_name = path.file_name().into_string().unwrap();

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

        max_epoch
    }

    fn train_directory(&self, tags: Option<&String>, epoch: usize) -> PathBuf {
        let name = format!("{EPOCH_PREFIX}{epoch}");

        match tags {
            Some(tags) => self.directory.join(tags).join(name),
            None => self.directory.join(name),
        }
    }

    fn eval_directory(&self, tags: Option<&String>) -> PathBuf {
        match tags {
            Some(tags) => self.directory.join(tags),
            None => self.directory.clone(),
        }
    }

    fn file_path(&self, tags: Option<&String>, name: &str, epoch: Option<usize>) -> PathBuf {
        let directory = match epoch {
            Some(epoch) => self.train_directory(tags, epoch),
            None => self.eval_directory(tags),
        };
        let name = name.replace(' ', "_");
        let name = format!("{name}.log");
        directory.join(name)
    }

    fn create_directory(&self, tags: Option<&String>, epoch: Option<usize>) {
        let directory = match epoch {
            Some(epoch) => self.train_directory(tags, epoch),
            None => self.eval_directory(tags),
        };
        std::fs::create_dir_all(directory).ok();
    }
}

impl FileMetricLogger {
    fn log_item(&mut self, tags: Option<&String>, item: &MetricEntry) {
        let key = &item.name;
        let value = &item.serialize;

        let logger = match self.loggers.get_mut(key.as_ref()) {
            Some(val) => val,
            None => {
                self.create_directory(tags, self.epoch);

                let file_path = self.file_path(tags, key, self.epoch);
                let logger = FileLogger::new(file_path);
                let logger = AsyncLogger::new(logger);

                self.loggers.insert(key.to_string(), logger);
                self.loggers
                    .get_mut(key.as_ref())
                    .expect("Can get the previously saved logger.")
            }
        };

        logger.log(value.clone());
    }

    fn log_tags(&mut self, item: &MetricEntry) {
        let mut tags = String::new();
        item.tags.iter().for_each(|tag| tags += tag.as_str());
        let tags = tags.replace(" ", "-").trim().to_lowercase();
        self.log_item(Some(&tags), item);
    }
}

impl MetricLogger for FileMetricLogger {
    fn log(&mut self, item: &MetricEntry) {
        match item.tags.is_empty() {
            true => self.log_item(None, item),
            false => self.log_tags(item),
        }
    }

    fn end_epoch(&mut self, epoch: usize) {
        self.loggers.clear();
        if self.epoch.is_none() {
            panic!("Only evaluation logger supported.");
        }
        self.epoch = Some(epoch + 1);
    }

    fn read_numeric(&mut self, name: &str, epoch: usize) -> Result<Vec<NumericEntry>, String> {
        if let Some(value) = self.loggers.get(name) {
            value.sync()
        }

        let file_path = self.file_path(None, name, Some(epoch));

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
    fn log(&mut self, item: &MetricEntry) {
        if !self.values.contains_key(item.name.as_ref()) {
            self.values
                .insert(item.name.to_string(), vec![InMemoryLogger::default()]);
        }

        let values = self.values.get_mut(item.name.as_ref()).unwrap();

        values.last_mut().unwrap().log(item.serialize.clone());
    }

    fn end_epoch(&mut self, _epoch: usize) {
        for (_, values) in self.values.iter_mut() {
            values.push(InMemoryLogger::default());
        }
    }

    fn read_numeric(&mut self, name: &str, epoch: usize) -> Result<Vec<NumericEntry>, String> {
        let values = match self.values.get(name) {
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
}
