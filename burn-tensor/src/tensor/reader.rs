#[cfg(feature = "async-read")]
#[async_trait::async_trait]
/// Allows to create async data reader for backends.
pub trait AsyncReader<T> {
    /// Read the data asynchronously.
    async fn read(self: Box<Self>) -> T;
}

/// Define how data is read, sync or async.
pub enum Reader<T> {
    /// Sync data variant.
    Sync(T),
    #[cfg(feature = "async-read")]
    /// Async data variant.
    Async(Box<dyn AsyncReader<T>>),
}

impl<T> Reader<T> {
    #[cfg(feature = "async-read")]
    /// Read the data.
    pub async fn read(self) -> T {
        match self {
            Self::Sync(data) => data,
            Self::Async(func) => func.read().await,
        }
    }

    #[cfg(not(feature = "async-read"))]
    /// Read the data.
    pub fn read(self) -> T {
        match self {
            Self::Sync(data) => data,
        }
    }

    /// Force reading the data synchronously.
    pub fn read_force_sync(self) -> T {
        match self {
            Self::Sync(data) => data,
            #[cfg(feature = "async-read")] // TODO: Maybe block_on here instead.
            Self::Async(_) => panic!("Force sync, but got async function"),
        }
    }
}
