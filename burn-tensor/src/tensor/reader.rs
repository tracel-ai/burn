use crate::Data;

#[cfg(feature = "async-read")]
#[async_trait::async_trait]
/// Allows to create async data reader for backends.
pub trait AsyncDataReader<E, const D: usize> {
    /// Read the data asynchronously.
    async fn read(self: Box<Self>) -> Data<E, D>;
}

/// Define how data is read, sync or async.
pub enum DataReader<E, const D: usize> {
    /// Sync data variant.
    Sync(Data<E, D>),
    #[cfg(feature = "async-read")]
    /// Async data variant.
    Async(Box<dyn AsyncDataReader<E, D>>),
}

impl<E, const D: usize> DataReader<E, D> {
    #[cfg(feature = "async-read")]
    /// Read the data.
    pub async fn read(self) -> Data<E, D> {
        match self {
            DataReader::Sync(data) => data,
            DataReader::Async(func) => func.read().await,
        }
    }

    #[cfg(not(feature = "async-read"))]
    /// Read the data.
    pub fn read(self) -> Data<E, D> {
        match self {
            DataReader::Sync(data) => data,
        }
    }

    /// Force reading the data synchronously.
    pub fn read_force_sync(self) -> Data<E, D> {
        match self {
            DataReader::Sync(data) => data,
            #[cfg(feature = "async-read")] // TODO: Maybe block_on here instead.
            DataReader::Async(_) => panic!("Force sync, but got async function"),
        }
    }
}
