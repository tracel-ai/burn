use super::Logger;

/// In memory logger.
#[derive(Default)]
pub struct InMemoryLogger {
    pub(crate) values: Vec<String>,
}

impl<T> Logger<T> for InMemoryLogger
where
    T: std::fmt::Display,
{
    fn log(&mut self, item: T) {
        self.values.push(item.to_string());
    }
}
