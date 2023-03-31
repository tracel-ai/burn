use alloc::string::String;
use alloc::vec::Vec;

use crate::module::State;
use burn_tensor::Element;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

#[cfg(feature = "std")]
use self::std_enabled::FileBinGzRecorder;

#[derive(Debug)]
pub enum RecordError {
    FileNotFound(String),
    Unknown(String),
}

impl core::fmt::Display for RecordError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(format!("{self:?}").as_str())
    }
}

pub trait Recorder {
    type SaveArgs;
    type SaveOutput;
    type LoadArgs;

    fn save<Obj: Serialize + DeserializeOwned>(
        obj: Obj,
        args: Self::SaveArgs,
    ) -> Result<Self::SaveOutput, RecordError>;
    fn load<Obj: Serialize + DeserializeOwned>(args: Self::LoadArgs) -> Result<Obj, RecordError>;
}

pub trait FileRecorder: Recorder<SaveArgs = String, SaveOutput = (), LoadArgs = String> {}

pub trait RecordSettings {
    type FloatElem: Element + Serialize + DeserializeOwned;
    type IntElem: Element + Serialize + DeserializeOwned;
    type Recorder: Recorder;

    fn default() -> DefaultRecordSettings {
        DefaultRecordSettings {}
    }
}

pub struct DefaultRecordSettings;

impl RecordSettings for DefaultRecordSettings {
    type FloatElem = half::f16;
    type IntElem = i16;
    #[cfg(feature = "std")]
    type Recorder = FileBinGzRecorder;
    #[cfg(not(feature = "std"))]
    type Recorder = InMemoryBinRecorder;
}

pub type RecorderSaveArgs<S> = <<S as RecordSettings>::Recorder as Recorder>::SaveArgs;
pub type RecorderLoadArgs<S> = <<S as RecordSettings>::Recorder as Recorder>::LoadArgs;
pub type RecorderSaveResult<S> =
    Result<<<S as RecordSettings>::Recorder as Recorder>::SaveOutput, RecordError>;

pub trait Record {
    type Item<S: RecordSettings>;

    fn into_item<S: RecordSettings>(self) -> Self::Item<S>;
    fn from_item<S: RecordSettings>(item: Self::Item<S>) -> Self;

    /// Save the record to the provided file path using the given [StateFormat](StateFormat).
    ///
    /// # Notes
    ///
    /// The file extension will be added automatically depending on the state format.
    fn save<S>(self, args: RecorderSaveArgs<S>) -> RecorderSaveResult<S>
    where
        Self: Sized,
        S: RecordSettings,
        Self::Item<S>: Serialize + DeserializeOwned,
    {
        RecordWrapper::<Self>::save(self, args)
    }

    /// Load the record from the provided file path using the given [StateFormat](StateFormat).
    ///
    /// # Notes
    ///
    /// The file extension will be added automatically depending on the state format.
    fn load<S>(args: RecorderLoadArgs<S>) -> Result<Self, RecordError>
    where
        Self: Sized,
        S: RecordSettings,
        Self::Item<S>: Serialize + DeserializeOwned,
    {
        RecordWrapper::load(args)
    }
}

#[derive(new, Serialize, Deserialize)]
struct Metadata {
    elem_float: String,
    elem_int: String,
    format: String,
    version: String,
}

#[derive(Serialize, Deserialize)]
struct RecordWrapper<R> {
    record: R,
    metadata: Metadata,
}

impl<R> RecordWrapper<R>
where
    R: Record,
{
    fn save<S>(record: R, args: RecorderSaveArgs<S>) -> RecorderSaveResult<S>
    where
        S: RecordSettings,
        R::Item<S>: Serialize + DeserializeOwned,
    {
        let metadata = Metadata::new(
            core::any::type_name::<S::FloatElem>().to_string(),
            core::any::type_name::<S::IntElem>().to_string(),
            core::any::type_name::<S::Recorder>().to_string(),
            env!("CARGO_PKG_VERSION").to_string(),
        );
        let item = record.into_item::<S>();
        let record = RecordWrapper {
            record: item,
            metadata,
        };

        <S::Recorder as Recorder>::save(record, args)
    }

    fn load<S>(args: RecorderLoadArgs<S>) -> Result<R, RecordError>
    where
        S: RecordSettings,
        R::Item<S>: Serialize + DeserializeOwned,
    {
        let record: RecordWrapper<R::Item<S>> = <S::Recorder as Recorder>::load(args)?;

        Ok(R::from_item(record.record))
    }
}

fn bin_config() -> bincode::config::Configuration {
    bincode::config::standard()
}

impl<T: Element> Record for State<T> {
    type Item<S: RecordSettings> = State<S::FloatElem>;

    fn into_item<S: RecordSettings>(self) -> Self::Item<S> {
        self.convert::<S::FloatElem>()
    }

    fn from_item<S: RecordSettings>(item: Self::Item<S>) -> Self {
        item.convert()
    }
}

pub struct InMemoryBinRecorder;

impl Recorder for InMemoryBinRecorder {
    type SaveArgs = ();
    type SaveOutput = Vec<u8>;
    type LoadArgs = Vec<u8>;

    fn save<Obj: Serialize + DeserializeOwned>(
        obj: Obj,
        _args: Self::SaveArgs,
    ) -> Result<Vec<u8>, RecordError> {
        Ok(bincode::serde::encode_to_vec(obj, bin_config()).unwrap())
    }

    fn load<Obj: Serialize + DeserializeOwned>(args: Self::LoadArgs) -> Result<Obj, RecordError> {
        let state = bincode::serde::decode_borrowed_from_slice(&args, bin_config()).unwrap();
        Ok(state)
    }
}

#[cfg(feature = "std")]
mod std_enabled {
    use super::*;
    use flate2::{read::GzDecoder, write::GzEncoder, Compression};
    use std::{fs::File, path::Path};

    // TODO: Move from std to core after Error is core (see https://github.com/rust-lang/rust/issues/103765)
    impl std::error::Error for RecordError {}

    macro_rules! str2reader {
        (
        $file:expr,
        $ext:expr
    ) => {{
            let path_ref = &format!("{}.{}", $file, $ext);
            let path = Path::new(path_ref);

            File::open(path).map_err(|err| match err.kind() {
                std::io::ErrorKind::NotFound => RecordError::FileNotFound(err.to_string()),
                _ => RecordError::Unknown(err.to_string()),
            })
        }};
    }

    macro_rules! str2writer {
        (
        $file:expr,
        $ext:expr
    ) => {{
            let path_ref = &format!("{}.{}", $file, $ext);
            let path = Path::new(path_ref);
            if path.exists() {
                log::info!("File exists, replacing");
                std::fs::remove_file(path).unwrap();
            }

            File::create(path).map_err(|err| match err.kind() {
                std::io::ErrorKind::NotFound => RecordError::FileNotFound(err.to_string()),
                _ => RecordError::Unknown(err.to_string()),
            })
        }};
    }

    pub struct FileBinGzRecorder;

    impl Recorder for FileBinGzRecorder {
        type SaveArgs = String;
        type SaveOutput = ();
        type LoadArgs = String;

        fn save<Obj: Serialize + DeserializeOwned>(
            obj: Obj,
            file: String,
        ) -> Result<(), RecordError> {
            let config = bin_config();
            let writer = str2writer!(file, "bin.gz")?;
            let mut writer = GzEncoder::new(writer, Compression::default());

            bincode::serde::encode_into_std_write(&obj, &mut writer, config).unwrap();

            Ok(())
        }

        fn load<Obj: Serialize + DeserializeOwned>(file: String) -> Result<Obj, RecordError> {
            let reader = str2reader!(file, "bin.gz")?;
            let mut reader = GzDecoder::new(reader);
            let state = bincode::serde::decode_from_std_read(&mut reader, bin_config()).unwrap();

            Ok(state)
        }
    }

    pub struct FileBinRecorder;

    impl Recorder for FileBinRecorder {
        type SaveArgs = String;
        type SaveOutput = ();
        type LoadArgs = String;

        fn save<Obj: Serialize + DeserializeOwned>(
            obj: Obj,
            file: String,
        ) -> Result<(), RecordError> {
            let config = bin_config();
            let writer = str2writer!(file, "bin.gz")?;
            let mut writer = GzEncoder::new(writer, Compression::default());

            bincode::serde::encode_into_std_write(&obj, &mut writer, config).unwrap();

            Ok(())
        }

        fn load<Obj: Serialize + DeserializeOwned>(file: String) -> Result<Obj, RecordError> {
            let reader = str2reader!(file, "bin.gz")?;
            let mut reader = GzDecoder::new(reader);
            let state = bincode::serde::decode_from_std_read(&mut reader, bin_config()).unwrap();

            Ok(state)
        }
    }

    impl FileRecorder for FileBinGzRecorder {}
    impl FileRecorder for FileBinRecorder {}
}
