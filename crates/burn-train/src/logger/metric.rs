use super::{AsyncLogger, FileLogger, InMemoryLogger, Logger};
use crate::metric::{
    MetricDefinition, MetricEntry, MetricId, NumericEntry,
    store::{EpochSummary, Split},
};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
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
    fn log(
        &mut self,
        items: Vec<&MetricEntry>,
        epoch: usize,
        split: Split,
        tag: Option<Arc<String>>,
    );

    /// Read the logs for an epoch.
    fn read_numeric(
        &mut self,
        name: &str,
        epoch: usize,
        split: Split,
    ) -> Result<Vec<NumericEntry>, String>;

    /// Logs the metric definition information (name, description, unit, etc.)
    fn log_metric_definition(&mut self, definition: MetricDefinition);

    /// Logs summary of the epoch (duration, highest metric values reached, etc.)
    fn log_epoch_summary(&mut self, summary: EpochSummary);
}

/// The file metric logger.
pub struct FileMetricLogger {
    loggers: HashMap<String, AsyncLogger<String>>,
    directory: PathBuf,
    metric_definitions: HashMap<MetricId, MetricDefinition>,
    is_eval: bool,
    last_epoch: Option<usize>,
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
            metric_definitions: HashMap::default(),
            is_eval: false,
            last_epoch: None,
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
            metric_definitions: HashMap::default(),
            is_eval: true,
            last_epoch: None,
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

    fn train_directory(&self, tag: Option<&String>, epoch: usize, split: Split) -> PathBuf {
        let name = format!("{EPOCH_PREFIX}{epoch}");

        match tag {
            Some(tag) => self.directory.join(split.to_string()).join(tag).join(name),
            None => self.directory.join(split.to_string()).join(name),
        }
    }

    fn eval_directory(&self, tag: Option<&String>, split: Split) -> PathBuf {
        match tag {
            Some(tag) => self.directory.join(split.to_string()).join(tag),
            None => self.directory.clone(),
        }
    }

    fn file_path(
        &self,
        tag: Option<&String>,
        name: &str,
        epoch: Option<usize>,
        split: Split,
    ) -> PathBuf {
        let directory = match epoch {
            Some(epoch) => self.train_directory(tag, epoch, split),
            None => self.eval_directory(tag, split),
        };
        let name = name.replace(' ', "_");
        let name = format!("{name}.log");
        directory.join(name)
    }

    fn create_directory(&self, tag: Option<&String>, epoch: Option<usize>, split: Split) {
        let directory = match epoch {
            Some(epoch) => self.train_directory(tag, epoch, split),
            None => self.eval_directory(tag, split),
        };
        std::fs::create_dir_all(directory).ok();
    }
}

impl FileMetricLogger {
    fn log_item(
        &mut self,
        tag: Option<&String>,
        item: &MetricEntry,
        epoch: Option<usize>,
        split: Split,
    ) {
        let name = &self.metric_definitions.get(&item.metric_id).unwrap().name;
        let key = logger_key(name, split);
        let value = &item.serialized_entry.serialized;

        let logger = match self.loggers.get_mut(&key) {
            Some(val) => val,
            None => {
                self.create_directory(tag, epoch, split);

                let file_path = self.file_path(tag, name, epoch, split);
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
}

impl MetricLogger for FileMetricLogger {
    fn log(
        &mut self,
        items: Vec<&MetricEntry>,
        epoch: usize,
        split: Split,
        tag: Option<Arc<String>>,
    ) {
        if !self.is_eval && self.last_epoch != Some(epoch) {
            self.loggers.clear();
            self.last_epoch = Some(epoch);
        }
        for item in items.iter() {
            match tag {
                Some(ref tag) => {
                    let tag = tag.trim().replace(' ', "-").to_lowercase();
                    self.log_item(Some(&tag), item, Some(epoch), split);
                }
                None => self.log_item(None, item, Some(epoch), split),
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

    fn log_metric_definition(&mut self, definition: MetricDefinition) {
        self.metric_definitions
            .insert(definition.metric_id.clone(), definition);
    }

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
    last_epoch: Option<usize>,
    metric_definitions: HashMap<MetricId, MetricDefinition>,
}

impl InMemoryMetricLogger {
    /// Create a new in-memory metric logger.
    pub fn new() -> Self {
        Self::default()
    }
}

impl MetricLogger for InMemoryMetricLogger {
    fn log(
        &mut self,
        items: Vec<&MetricEntry>,
        epoch: usize,
        split: Split,
        _tag: Option<Arc<String>>,
    ) {
        if self.last_epoch != Some(epoch) {
            self.values
                .values_mut()
                .for_each(|loggers| loggers.push(InMemoryLogger::default()));
            self.last_epoch = Some(epoch);
        }
        for item in items.iter() {
            let name = &self.metric_definitions.get(&item.metric_id).unwrap().name;
            let key = logger_key(name, split);

            if !self.values.contains_key(&key) {
                self.values
                    .insert(key.to_string(), vec![InMemoryLogger::default()]);
            }

            let values = self.values.get_mut(&key).unwrap();

            values
                .last_mut()
                .unwrap()
                .log(item.serialized_entry.serialized.clone());
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

    fn log_metric_definition(&mut self, definition: MetricDefinition) {
        self.metric_definitions
            .insert(definition.metric_id.clone(), definition);
    }

    fn log_epoch_summary(&mut self, _summary: EpochSummary) {}
}
