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
    epoch: usize,
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
            epoch: 1,
        }
    }

    /// Number of epochs recorded.
    pub(crate) fn epochs(&self) -> usize {
        let mut max_epoch = 0;

        for path in fs::read_dir(&self.directory).unwrap() {
            let path = path.unwrap();

            if fs::metadata(path.path()).unwrap().is_dir() {
                let dir_name = path.file_name().into_string().unwrap();

                if !dir_name.starts_with(EPOCH_PREFIX) {
                    continue;
                }

                let epoch = dir_name.replace(EPOCH_PREFIX, "").parse::<usize>().ok();

                if let Some(epoch) = epoch {
                    if epoch > max_epoch {
                        max_epoch = epoch;
                    }
                }
            }
        }

        max_epoch
    }

    fn epoch_directory(&self, epoch: usize) -> PathBuf {
        let name = format!("{}{}", EPOCH_PREFIX, epoch);
        self.directory.join(name)
    }

    fn file_path(&self, name: &str, epoch: usize) -> PathBuf {
        let directory = self.epoch_directory(epoch);
        let name = name.replace(' ', "_");
        let name = format!("{name}.log");
        directory.join(name)
    }

    fn create_directory(&self, epoch: usize) {
        let directory = self.epoch_directory(epoch);
        std::fs::create_dir_all(directory).ok();
    }
}

impl MetricLogger for FileMetricLogger {
    fn log(&mut self, item: &MetricEntry) {
        let key = &item.name;
        let value = &item.serialize;

        let logger = match self.loggers.get_mut(key) {
            Some(val) => val,
            None => {
                self.create_directory(self.epoch);

                let file_path = self.file_path(key, self.epoch);
                let logger = FileLogger::new(file_path);
                let logger = AsyncLogger::new(logger);

                self.loggers.insert(key.clone(), logger);
                self.loggers
                    .get_mut(key)
                    .expect("Can get the previously saved logger.")
            }
        };

        logger.log(value.clone());
    }

    fn end_epoch(&mut self, epoch: usize) {
        self.loggers.clear();
        self.epoch = epoch + 1;
    }

    fn read_numeric(&mut self, name: &str, epoch: usize) -> Result<Vec<NumericEntry>, String> {
        if let Some(value) = self.loggers.get(name) {
            value.sync()
        }

        let file_path = self.file_path(name, epoch);

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
        if !self.values.contains_key(&item.name) {
            self.values
                .insert(item.name.clone(), vec![InMemoryLogger::default()]);
        }

        let values = self.values.get_mut(&item.name).unwrap();

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
