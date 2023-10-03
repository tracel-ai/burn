use super::{AsyncLogger, FileLogger, Logger};
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
    fn epoch(&mut self, epoch: usize);

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
        std::fs::create_dir_all(&directory).ok();
        let name = name.replace(' ', "_");

        format!("{directory}/{name}.log")
    }
}

impl MetricLogger for FileMetricLogger {
    fn log(&mut self, item: &MetricEntry) {
        let key = &item.name;
        let value = &item.serialize;

        let logger = match self.loggers.get_mut(key) {
            Some(val) => val,
            None => {
                let file_path = self.file_path(key, self.epoch);
                let logger = FileLogger::new(&file_path);
                let logger = AsyncLogger::new(logger);

                self.loggers.insert(key.clone(), logger);
                self.loggers.get_mut(key).unwrap()
            }
        };

        logger.log(value.clone());
    }

    fn epoch(&mut self, epoch: usize) {
        self.loggers.clear();
        self.epoch = epoch;
    }

    fn read_numeric(&mut self, name: &str, epoch: usize) -> Result<Vec<f64>, String> {
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
