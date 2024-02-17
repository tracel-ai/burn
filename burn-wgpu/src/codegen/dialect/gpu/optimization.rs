use super::{Operation, ScopeProcessing};

impl ScopeProcessing {
    pub fn optimize(self) -> Self {
        self.merge_read_global()
    }

    fn merge_read_global(mut self) -> Self {
        // TODO: Now we only support merging the first found global read, but we should merged all
        // the ones that can be merged.
        let mut main_read = None;
        let mut main_position = 0;
        let mut merged_position = Vec::new();

        for (position, operation) in self.operations.iter().enumerate() {
            let algo = match operation {
                Operation::Algorithm(algo) => algo,
                _ => continue,
            };

            let read = match algo {
                super::Algorithm::ReadGlobalWithLayout(algo) => algo,
                _ => continue,
            };

            if main_read.is_none() {
                main_read = Some(read.clone());
                main_position = position;
                continue;
            }

            if let Some(main) = &main_read {
                if let Some(merged) = main.try_merge(read) {
                    main_read = Some(merged);
                    merged_position.push(position);
                    continue;
                }
            }
        }

        if merged_position.is_empty() {
            return self;
        }

        if let Some(main) = main_read {
            let mut operations = Vec::with_capacity(self.operations.len());

            for (position, operation) in self.operations.into_iter().enumerate() {
                if position == main_position {
                    operations.push(Operation::Algorithm(
                        super::Algorithm::ReadGlobalWithLayout(main.clone()),
                    ));
                    continue;
                }

                if !merged_position.contains(&position) {
                    operations.push(operation);
                }
            }

            self.operations = operations;
        }

        self
    }
}
