use core::marker::PhantomData;

use burn_tensor::Element;
use serde::{de::DeserializeOwned, Serialize};

use super::Recorder;

pub trait RecordSettings {
    type FloatElem: Element + Serialize + DeserializeOwned;
    type IntElem: Element + Serialize + DeserializeOwned;
    type Recorder: Recorder;
}

#[cfg(feature = "std")]
pub struct Settings<Float = half::f16, Int = i16, Recorder = crate::record::FileBinGzRecorder> {
    float: PhantomData<Float>,
    int: PhantomData<Int>,
    recorder: PhantomData<Recorder>,
}
#[cfg(not(feature = "std"))]
pub struct Settings<Float = half::f16, Int = i16, Recorder = crate::record::InMemoryBinRecorder> {
    float: PhantomData<Float>,
    int: PhantomData<Int>,
    recorder: PhantomData<Recorder>,
}

impl<Float, Int, Recorder> RecordSettings for Settings<Float, Int, Recorder>
where
    Float: Element + Serialize + DeserializeOwned,
    Int: Element + Serialize + DeserializeOwned,
    Recorder: crate::record::Recorder,
{
    type FloatElem = Float;
    type IntElem = Int;
    type Recorder = Recorder;
}
