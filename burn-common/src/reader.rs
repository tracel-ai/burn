use alloc::boxed::Box;
use core::marker::PhantomData;

#[cfg(target_family = "wasm")]
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
    #[cfg(target_family = "wasm")]
    /// Async data variant.
    Async(Box<dyn AsyncReader<T>>),
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
    F: Send + Fn(I) -> O,
{
    fn read(self: Box<Self>) -> O {
        let input = match self.reader {
            Reader::Concrete(input) => input,
            Reader::Sync(func) => func.read(),
            #[cfg(target_family = "wasm")]
            Reader::Async(_func) => panic!("Can't read async function with sync reader"),
        };

        (self.mapper)(input)
    }
}

#[cfg(target_family = "wasm")]
#[async_trait::async_trait]
impl<I, O, F> AsyncReader<O> for MappedReader<I, O, F>
where
    I: Send,
    O: Send,
    F: Send + Fn(I) -> O,
{
    async fn read(self: Box<Self>) -> O {
        let input = self.reader.read().await;
        (self.mapper)(input)
    }
}

impl<T> Reader<T> {
    #[cfg(target_family = "wasm")]
    /// Read the data.
    pub async fn read(self) -> T {
        match self {
            Self::Concrete(data) => data,
            Self::Sync(reader) => reader.read(),
            Self::Async(func) => func.read().await,
        }
    }

    #[cfg(not(target_family = "wasm"))]
    /// Read the data.
    pub fn read(self) -> T {
        match self {
            Self::Concrete(data) => data,
            Self::Sync(reader) => reader.read(),
        }
    }

    /// Map the current reader to another type.
    pub fn map<O, F: Fn(T) -> O>(self, mapper: F) -> Reader<O>
    where
        T: 'static + Send,
        O: 'static + Send,
        F: 'static + Send,
    {
        #[cfg(target_family = "wasm")]
        return Reader::Async(Box::new(MappedReader::new(self, mapper)));

        #[cfg(not(target_family = "wasm"))]
        Reader::Sync(Box::new(MappedReader::new(self, mapper)))
    }
}
