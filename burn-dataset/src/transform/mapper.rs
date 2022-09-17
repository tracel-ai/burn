use crate::Dataset;

pub trait Mapper<I, O> {
    fn map(&self, item: &I) -> O;
}

pub struct MapperDataset<M, I> {
    dataset: Box<dyn Dataset<I>>,
    mapper: M,
}

impl<M, I> MapperDataset<M, I> {
    pub fn new(dataset: Box<dyn Dataset<I>>, mapper: M) -> Self {
        Self { dataset, mapper }
    }
}

impl<M, I, O> Dataset<O> for MapperDataset<M, I>
where
    M: Mapper<I, O> + Send + Sync,
    I: Send + Sync,
    O: Send + Sync,
{
    fn get(&self, index: usize) -> Option<O> {
        let item = self.dataset.get(index);
        item.map(|item| self.mapper.map(&item))
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{test_data, InMemDataset};

    #[test]
    pub fn given_mapper_dataset_when_iterate_should_iterate_though_all_map_items() {
        struct StringToFirstChar {}
        impl Mapper<String, String> for StringToFirstChar {
            fn map(&self, item: &String) -> String {
                let mut item = item.clone();
                item.truncate(1);
                item
            }
        }
        let items_original = test_data::string_items();
        let dataset = InMemDataset::new(items_original);
        let dataset = MapperDataset::new(Box::new(dataset), StringToFirstChar {});

        let items: Vec<String> = dataset.iter().collect();

        assert_eq!(vec!["1", "2", "3", "4"], items);
    }
}
