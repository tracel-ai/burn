use super::{AsyncLogger, FileLogger, InMemoryLogger, Logger};
use crate::metric::{
    MetricDefinition, MetricEntry, MetricId, NumericEntry,
    store::{EpochSummary, MetricsUpdate, Split},
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
    /// * `update` - Update information for all registered metrics.
    /// * `epoch` - Current epoch.
    /// * `split` - Current dataset split.
    fn log(&mut self, update: MetricsUpdate, epoch: usize, split: &Split);

    /// Read the logs for an epoch.
    fn read_numeric(
        &mut self,
        name: &str,
        epoch: usize,
        split: &Split,
    ) -> Result<Vec<NumericEntry>, String>;

    /// Logs the metric definition information (name, description, unit, etc.)
    fn log_metric_definition(&mut self, definition: MetricDefinition);

    /// Logs summary at the end of the epoch.
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

    pub(crate) fn split_exists(&self, split: &Split) -> bool {
        self.split_dir(split).is_some()
    }

    pub(crate) fn split_dir(&self, split: &Split) -> Option<PathBuf> {
        let split_path = match split {
            Split::Test(Some(tag)) => self.directory.join(split.to_string()).join(tag.as_str()),
            other => self.directory.join(other.to_string()),
        };
        (split_path.exists() && split_path.is_dir()).then_some(split_path)
    }

    pub(crate) fn is_epoch_dir<P: AsRef<str>>(dirname: P) -> bool {
        dirname.as_ref().starts_with(EPOCH_PREFIX)
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

    fn train_directory(&self, epoch: usize, split: &Split) -> PathBuf {
        let name = format!("{EPOCH_PREFIX}{epoch}");

        match split {
            Split::Train | Split::Valid | Split::Test(None) => {
                self.directory.join(split.to_string()).join(name)
            }
            Split::Test(Some(tag)) => {
                let tag = format_tag(tag);
                self.directory.join(split.to_string()).join(tag).join(name)
            }
        }
    }

    fn eval_directory(&self, split: &Split) -> PathBuf {
        match split {
            Split::Train | Split::Valid | Split::Test(None) => self.directory.clone(),
            Split::Test(Some(tag)) => self.directory.join(split.to_string()).join(format_tag(tag)),
        }
    }

    fn file_path(&self, name: &str, epoch: Option<usize>, split: &Split) -> PathBuf {
        let directory = match epoch {
            Some(epoch) => self.train_directory(epoch, split),
            None => self.eval_directory(split),
        };
        let name = name.replace(' ', "_");
        let name = format!("{name}.log");
        directory.join(name)
    }

    fn create_directory(&self, epoch: Option<usize>, split: &Split) {
        let directory = match epoch {
            Some(epoch) => self.train_directory(epoch, split),
            None => self.eval_directory(split),
        };
        std::fs::create_dir_all(directory).ok();
    }
}

impl FileMetricLogger {
    fn log_item(&mut self, item: &MetricEntry, epoch: Option<usize>, split: &Split) {
        let name = &self.metric_definitions.get(&item.metric_id).unwrap().name;
        let key = logger_key(name, split);
        let value = &item.serialized_entry.serialized;

        let logger = match self.loggers.get_mut(&key) {
            Some(val) => val,
            None => {
                self.create_directory(epoch, split);

                let file_path = self.file_path(name, epoch, split);
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

fn format_tag(tag: &str) -> String {
    tag.trim().replace(' ', "-").to_lowercase()
}

impl MetricLogger for FileMetricLogger {
    fn log(&mut self, update: MetricsUpdate, epoch: usize, split: &Split) {
        if !self.is_eval && self.last_epoch != Some(epoch) {
            self.loggers.clear();
            self.last_epoch = Some(epoch);
        }

        let entries: Vec<_> = update
            .entries
            .iter()
            .chain(
                update
                    .entries_numeric
                    .iter()
                    .map(|numeric_update| &numeric_update.entry),
            )
            .cloned()
            .collect();

        for item in entries.iter() {
            self.log_item(item, Some(epoch), split);
        }
    }

    fn read_numeric(
        &mut self,
        name: &str,
        epoch: usize,
        split: &Split,
    ) -> Result<Vec<NumericEntry>, String> {
        if let Some(value) = self.loggers.get(name) {
            value.sync()
        }

        let file_path = self.file_path(name, Some(epoch), split);

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

fn logger_key(name: &str, split: &Split) -> String {
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
    fn log(&mut self, update: MetricsUpdate, epoch: usize, split: &Split) {
        if self.last_epoch != Some(epoch) {
            self.values
                .values_mut()
                .for_each(|loggers| loggers.push(InMemoryLogger::default()));
            self.last_epoch = Some(epoch);
        }

        let entries: Vec<_> = update
            .entries
            .iter()
            .chain(
                update
                    .entries_numeric
                    .iter()
                    .map(|numeric_update| &numeric_update.entry),
            )
            .cloned()
            .collect();

        for item in entries.iter() {
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
        split: &Split,
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
