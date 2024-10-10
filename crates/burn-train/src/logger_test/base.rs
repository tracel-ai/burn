/// The logger trait.
pub trait Logger<T>: Send {
    /// Logs an item.
    ///
    /// # Arguments
    ///
    /// * `item` - The item.
    fn log(&mut self, item: T);
}

/// The logger backend trait.
pub trait LoggerBackend {
    /// The logger type.
    type Logger<T>: Logger<T>;

    /// Create a new logger.
    ///
    /// # Arguments
    ///
    /// * `epoch` - The epoch.
    ///
    /// # Returns
    ///
    /// The logger.
    fn create<T>(&self, epoch: usize) -> Self::Logger<T>;
}
