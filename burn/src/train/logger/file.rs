use super::Logger;
use std::{fs::File, io::Write};

pub struct FileLogger {
    file: File,
}

impl FileLogger {
    pub fn new(path: &str) -> Self {
        let mut options = std::fs::File::options();
        let file = options
            .write(true)
            .append(true)
            .create(true)
            .open(&path)
            .unwrap();

        Self { file }
    }
}

impl<T> Logger<T> for FileLogger
where
    T: std::fmt::Display,
{
    fn log(&mut self, item: T) {
        write!(&mut self.file, "{}", item.to_string()).unwrap();
    }
}
