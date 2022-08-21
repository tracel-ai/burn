use crate::{Dataset, DatasetIterator};

pub struct InMemDataset<I> {
    items: Vec<I>,
}

impl<I> InMemDataset<I> {
    pub fn new(items: Vec<I>) -> Self {
        InMemDataset { items }
    }
}

impl<I> Dataset<I> for InMemDataset<I>
where
    I: Clone,
{
    fn get(&self, index: usize) -> Option<I> {
        match self.items.get(index) {
            Some(item) => Some(item.clone()),
            None => None,
        }
    }
    fn iter<'a>(&'a self) -> DatasetIterator<'a, I> {
        DatasetIterator::new(self)
    }
    fn len(&self) -> usize {
        self.items.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_data;

    #[test]
    pub fn given_in_memory_dataset_when_iterate_should_iterate_though_all_items() {
        let items_original = test_data::string_items();
        let dataset = InMemDataset::new(items_original.clone());

        let items: Vec<String> = dataset.iter().collect();

        assert_eq!(items_original, items);
    }
}
