use alloc::boxed::Box;
use core::marker::PhantomData;

#[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
#[async_trait::async_trait]
/// Allows to create async reader.
pub trait AsyncReader<T>: Send {
    /// Read asynchronously.
    async fn read(self: Box<Self>) -> T;
}

/// Define how data is read, sync or async.
pub enum Reader<T> {
    /// Concrete variant.
    Concrete(T),
    /// Sync data variant.
    Sync(Box<dyn SyncReader<T>>),
    #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
    /// Async data variant.
    Async(Box<dyn AsyncReader<T>>),
    #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
    /// Future data variant.
    Future(core::pin::Pin<Box<dyn core::future::Future<Output = T> + Send>>),
}

/// Allows to create sync reader.
pub trait SyncReader<T>: Send {
    /// Read synchronously.
    fn read(self: Box<Self>) -> T;
}

#[derive(new)]
struct MappedReader<I, O, F> {
    reader: Reader<I>,
    mapper: F,
    _output: PhantomData<O>,
}

impl<I, O, F> SyncReader<O> for MappedReader<I, O, F>
where
    I: Send,
    O: Send,
    F: Send + FnOnce(I) -> O,
{
    fn read(self: Box<Self>) -> O {
        let input = self
            .reader
            .read_sync()
            .expect("Only sync data supported in a sync reader.");

        (self.mapper)(input)
    }
}

#[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
#[async_trait::async_trait]
impl<I, O, F> AsyncReader<O> for MappedReader<I, O, F>
where
    I: Send,
    O: Send,
    F: Send + FnOnce(I) -> O,
{
    async fn read(self: Box<Self>) -> O {
        let input = self.reader.read().await;
        (self.mapper)(input)
    }
}

impl<T> Reader<T> {
    #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
    /// Read the data.
    pub async fn read(self) -> T {
        match self {
            Self::Concrete(data) => data,
            Self::Sync(reader) => reader.read(),
            Self::Async(func) => func.read().await,
            Self::Future(future) => future.await,
        }
    }

    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    /// Read the data.
    pub fn read(self) -> T {
        match self {
            Self::Concrete(data) => data,
            Self::Sync(reader) => reader.read(),
        }
    }

    /// Read the data only if sync, returns None if an async reader.
    pub fn read_sync(self) -> Option<T> {
        match self {
            Self::Concrete(data) => Some(data),
            Self::Sync(reader) => Some(reader.read()),
            #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
            Self::Async(_func) => return None,
            #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
            Self::Future(_future) => return None,
        }
    }

    /// Map the current reader to another type.
    pub fn map<O, F: FnOnce(T) -> O>(self, mapper: F) -> Reader<O>
    where
        T: 'static + Send,
        O: 'static + Send,
        F: 'static + Send,
    {
        #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
        return Reader::Async(Box::new(MappedReader::new(self, mapper)));

        #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
        Reader::Sync(Box::new(MappedReader::new(self, mapper)))
    }
}
