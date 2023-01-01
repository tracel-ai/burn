use super::{AsyncLogger, FileLogger, Logger};
use crate::metric::MetricEntry;
use std::collections::HashMap;

pub trait MetricLogger: Send {
    fn log(&mut self, item: &MetricEntry);
    fn epoch(&mut self, epoch: usize);
}

pub struct FileMetricLogger {
    loggers: HashMap<String, Box<dyn Logger<String>>>,
    directory: String,
    epoch: usize,
}

impl FileMetricLogger {
    pub fn new(directory: &str) -> Self {
        Self {
            loggers: HashMap::new(),
            directory: directory.to_string(),
            epoch: 1,
        }
    }
}

impl MetricLogger for FileMetricLogger {
    fn log(&mut self, item: &MetricEntry) {
        let key = &item.name;
        let value = &item.serialize;

        let logger = match self.loggers.get_mut(key) {
            Some(val) => val,
            None => {
                let directory = format!("{}/epoch-{}", self.directory, self.epoch);
                let file_path = format!("{directory}/{key}.log");
                std::fs::create_dir_all(&directory).ok();

                let logger = FileLogger::new(&file_path);
                let logger = AsyncLogger::new(Box::new(logger));

                self.loggers.insert(key.clone(), Box::new(logger));
                self.loggers.get_mut(key).unwrap()
            }
        };

        logger.log(value.clone());
    }

    fn epoch(&mut self, epoch: usize) {
        self.loggers.clear();
        self.epoch = epoch;
    }
}
