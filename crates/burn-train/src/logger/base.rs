/// The logger trait.
pub trait Logger<T>: Send {
    /// Logs an item.
    ///
    /// # Arguments
    ///
    /// * `item` - The item.
    fn log(&mut self, item: T);
}
