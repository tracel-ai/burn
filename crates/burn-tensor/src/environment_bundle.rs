use crate::{BuildStamp, EnvironmentBundle, EnvironmentError, EnvironmentSummary};
use alloc::string::String;
use burn_std::environment::ReadBundle as ReadStoredBundle;
use std::path::Path;
use std::time::SystemTime;

/// What a saved [`EnvironmentBundle`] says of itself, read without loading
/// it, under the error every other compute environment call answers with.
///
/// ```rust,ignore
/// use burn::tensor::{BuildStamp, EnvironmentBundle, ReadBundle};
///
/// let bundle = EnvironmentBundle::read(path)?;
/// let ours = bundle.saved_by()? == Some(BuildStamp::current());
/// println!("{} holds {:?}", bundle.name()?, bundle.summary()?);
/// ```
pub trait ReadBundle: Sized {
    /// The bundle saved at `path`. Its bytes are read from the file when
    /// something uses them, and the readers below open the file in place, so
    /// the file must stay as it is while the bundle is held.
    ///
    /// # Errors
    ///
    /// [`Storage`](EnvironmentError::Storage): the file cannot be read.
    fn read(path: impl AsRef<Path>) -> Result<Self, EnvironmentError>;

    /// The name the bundle's manifest gives it, as
    /// [`save_named`](crate::ComputeEnvironment::save_named) set it.
    ///
    /// # Errors
    ///
    /// [`Storage`](EnvironmentError::Storage): the bytes are not a bundle, or
    /// do not read.
    fn name(&self) -> Result<String, EnvironmentError>;

    /// The executable that saved the bundle, when its manifest names one:
    /// [`BuildStamp::current`] when this executable did.
    ///
    /// # Errors
    ///
    /// As [`name`](Self::name).
    fn saved_by(&self) -> Result<Option<BuildStamp>, EnvironmentError>;

    /// When the bundle was saved, when its manifest says.
    ///
    /// # Errors
    ///
    /// As [`name`](Self::name).
    fn saved_at(&self) -> Result<Option<SystemTime>, EnvironmentError>;

    /// What the bundle holds, counted.
    ///
    /// # Errors
    ///
    /// As [`name`](Self::name).
    fn summary(&self) -> Result<EnvironmentSummary, EnvironmentError>;
}

/// The storage's reads, each error taken into this crate's.
impl ReadBundle for EnvironmentBundle {
    fn read(path: impl AsRef<Path>) -> Result<Self, EnvironmentError> {
        Ok(<Self as ReadStoredBundle>::read(path)?)
    }

    fn name(&self) -> Result<String, EnvironmentError> {
        Ok(ReadStoredBundle::name(self)?)
    }

    fn saved_by(&self) -> Result<Option<BuildStamp>, EnvironmentError> {
        Ok(ReadStoredBundle::saved_by(self)?)
    }

    fn saved_at(&self) -> Result<Option<SystemTime>, EnvironmentError> {
        Ok(ReadStoredBundle::saved_at(self)?)
    }

    fn summary(&self) -> Result<EnvironmentSummary, EnvironmentError> {
        Ok(ReadStoredBundle::summary(self)?)
    }
}
