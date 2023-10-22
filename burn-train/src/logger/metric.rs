use super::{AsyncLogger, FileLogger, InMemoryLogger, Logger};
use crate::metric::MetricEntry;
use std::collections::HashMap;

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
    fn read_numeric(&mut self, name: &str, epoch: usize) -> Result<Vec<f64>, String>;
}

/// The file metric logger.
pub struct FileMetricLogger {
    loggers: HashMap<String, AsyncLogger<String>>,
    directory: String,
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
    pub fn new(directory: &str) -> Self {
        Self {
            loggers: HashMap::new(),
            directory: directory.to_string(),
            epoch: 1,
        }
    }

    fn file_path(&self, name: &str, epoch: usize) -> String {
        let directory = format!("{}/epoch-{}", self.directory, epoch);
        let name = name.replace(' ', "_");

        format!("{directory}/{name}.log")
    }
    fn create_directory(&self, epoch: usize) {
        let directory = format!("{}/epoch-{}", self.directory, epoch);
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
                let logger = FileLogger::new(&file_path);
                let logger = AsyncLogger::new(logger);

                self.loggers.insert(key.clone(), logger);
                self.loggers.get_mut(key).unwrap()
            }
        };

        logger.log(value.clone());
    }

    fn end_epoch(&mut self, epoch: usize) {
        self.loggers.clear();
        self.epoch = epoch + 1;
    }

    fn read_numeric(&mut self, name: &str, epoch: usize) -> Result<Vec<f64>, String> {
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
                    match value.parse::<f64>() {
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
            Err("Parsing float errors".to_string())
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

    fn read_numeric(&mut self, name: &str, epoch: usize) -> Result<Vec<f64>, String> {
        let values = match self.values.get(name) {
            Some(values) => values,
            None => return Ok(Vec::new()),
        };

        match values.get(epoch - 1) {
            Some(logger) => Ok(logger
                .values
                .iter()
                .filter_map(|value| value.parse::<f64>().ok())
                .collect()),
            None => Ok(Vec::new()),
        }
    }
}
