use super::Logger;
use std::{fs::File, io::Write};

/// File logger.
pub struct FileLogger {
    file: File,
}

impl FileLogger {
    /// Create a new file logger.
    ///
    /// # Arguments
    ///
    /// * `path` - The path.
    ///
    /// # Returns
    ///
    /// The file logger.
    pub fn new(path: &str) -> Self {
        let mut options = std::fs::File::options();
        let file = options
            .write(true)
            .truncate(true)
            .create(true)
            .open(path)
            .unwrap();

        Self { file }
    }
}

impl<T> Logger<T> for FileLogger
where
    T: std::fmt::Display,
{
    fn log(&mut self, item: T) {
        writeln!(&mut self.file, "{item}").unwrap();
    }
}
