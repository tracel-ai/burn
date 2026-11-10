use crate::{BuildStamp, EnvironmentBundle, EnvironmentError, EnvironmentSummary};
use alloc::string::String;
use std::path::Path;
use std::time::SystemTime;

pub use burn_std::environment::BundleDescription;

/// What a saved [`EnvironmentBundle`] says of itself, read without loading
/// it, under the error every other compute environment call answers with.
///
/// [`describe`](Self::describe) reads every answer in one opening of the
/// bundle; each other reader is a view of it, for a caller that wants one.
///
/// ```rust,ignore
/// use burn::tensor::{BuildStamp, EnvironmentBundle, ReadBundle};
///
/// let bundle = EnvironmentBundle::read(path)?;
/// let description = bundle.describe()?;
/// let ours = description.saved_by == Some(BuildStamp::current());
/// println!("{} holds {:?}", description.name, description.summary);
/// ```
pub trait ReadBundle: Sized {
    /// The bundle saved at `path`. The file is opened now and held, and its
    /// bytes read when something uses them: a file renamed over `path`
    /// meanwhile is not the one read.
    ///
    /// # Errors
    ///
    /// [`Storage`](EnvironmentError::Storage): the file does not open.
    fn read(path: impl AsRef<Path>) -> Result<Self, EnvironmentError>;

    /// What the bundle says of itself: its name, which executable saved it
    /// and when, and its counts.
    ///
    /// # Errors
    ///
    /// [`Storage`](EnvironmentError::Storage): the bytes do not read, they
    /// are not a bundle, or its entries cannot be counted.
    fn describe(&self) -> Result<BundleDescription, EnvironmentError>;

    /// The name the bundle's manifest gives it, as
    /// [`save_named`](crate::ComputeEnvironment::save_named) set it.
    ///
    /// # Errors
    ///
    /// As [`describe`](Self::describe).
    fn name(&self) -> Result<String, EnvironmentError> {
        Ok(self.describe()?.name)
    }

    /// The executable that saved the bundle, when its manifest names one:
    /// [`BuildStamp::current`] when this executable did.
    ///
    /// # Errors
    ///
    /// As [`describe`](Self::describe).
    fn saved_by(&self) -> Result<Option<BuildStamp>, EnvironmentError> {
        Ok(self.describe()?.saved_by)
    }

    /// When the bundle was saved, when its manifest says.
    ///
    /// # Errors
    ///
    /// As [`describe`](Self::describe).
    fn saved_at(&self) -> Result<Option<SystemTime>, EnvironmentError> {
        Ok(self.describe()?.saved_at)
    }

    /// What the bundle holds, counted.
    ///
    /// # Errors
    ///
    /// As [`describe`](Self::describe).
    fn summary(&self) -> Result<EnvironmentSummary, EnvironmentError> {
        Ok(self.describe()?.summary)
    }
}

/// The storage's reads, each error taken into this crate's.
impl ReadBundle for EnvironmentBundle {
    fn read(path: impl AsRef<Path>) -> Result<Self, EnvironmentError> {
        Ok(Self::open(path)?)
    }

    fn describe(&self) -> Result<BundleDescription, EnvironmentError> {
        Ok(self.read_description()?)
    }
}
