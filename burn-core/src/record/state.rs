use super::{Record, RecordSettings};
use crate::module::State;
use burn_tensor::Element;

impl<T: Element> Record for State<T> {
    type Item<S: RecordSettings> = State<S::FloatElem>;

    fn into_item<S: RecordSettings>(self) -> Self::Item<S> {
        self.convert::<S::FloatElem>()
    }

    fn from_item<S: RecordSettings>(item: Self::Item<S>) -> Self {
        item.convert()
    }
}
