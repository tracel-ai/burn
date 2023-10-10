use crate::{logger::MetricLogger, Aggregate, Direction};
use std::collections::HashMap;

/// Type that can be used to fetch and use numeric metric aggregates.
#[derive(Default, Debug)]
pub(crate) struct NumericMetricsAggregate {
    mean_for_each_epoch: HashMap<Key, f64>,
}

#[derive(new, Hash, PartialEq, Eq, Debug)]
struct Key {
    name: String,
    epoch: usize,
}

impl NumericMetricsAggregate {
    pub(crate) fn mean(
        &mut self,
        name: &str,
        epoch: usize,
        loggers: &mut [Box<dyn MetricLogger>],
    ) -> Option<f64> {
        let key = Key::new(name.to_string(), epoch);

        if let Some(value) = self.mean_for_each_epoch.get(&key) {
            return Some(*value);
        }

        let points = || {
            let mut errors = Vec::new();
            for logger in loggers {
                match logger.read_numeric(name, epoch) {
                    Ok(points) => return Ok(points),
                    Err(err) => errors.push(err),
                };
            }

            Err(errors.join(" "))
        };

        let points = points().expect("Can read values");

        if points.is_empty() {
            return None;
        }

        let num_points = points.len();
        let mean = points.into_iter().sum::<f64>() / num_points as f64;

        self.mean_for_each_epoch.insert(key, mean);
        Some(mean)
    }

    pub(crate) fn find_epoch(
        &mut self,
        name: &str,
        aggregate: Aggregate,
        direction: Direction,
        loggers: &mut [Box<dyn MetricLogger>],
    ) -> Option<usize> {
        let mut data = Vec::new();
        let mut current_epoch = 1;

        loop {
            match aggregate {
                Aggregate::Mean => match self.mean(name, current_epoch, loggers) {
                    Some(value) => {
                        data.push(value);
                    }
                    None => break,
                },
            };

            current_epoch += 1;
        }

        if data.is_empty() {
            return None;
        }

        let mut current_value = match &direction {
            Direction::Lowest => f64::MAX,
            Direction::Highest => f64::MIN,
        };

        for (i, value) in data.into_iter().enumerate() {
            match &direction {
                Direction::Lowest => {
                    if value < current_value {
                        current_value = value;
                        current_epoch = i + 1;
                    }
                }
                Direction::Highest => {
                    if value > current_value {
                        current_value = value;
                        current_epoch = i + 1;
                    }
                }
            }
        }

        Some(current_epoch)
    }
}

#[cfg(test)]
mod tests {
    use crate::{logger::FileMetricLogger, metric::MetricEntry};

    use super::*;

    struct TestLogger {
        logger: FileMetricLogger,
        epoch: usize,
    }
    const NAME: &str = "test-logger";

    impl TestLogger {
        fn new() -> Self {
            Self {
                logger: FileMetricLogger::new("/tmp"),
                epoch: 1,
            }
        }
        fn log(&mut self, num: f64) {
            self.logger.log(&MetricEntry::new(
                NAME.into(),
                num.to_string(),
                num.to_string(),
            ));
        }
        fn new_epoch(&mut self) {
            self.epoch += 1;
            self.logger.epoch(self.epoch);
        }
    }

    #[test]
    fn should_find_epoch() {
        let mut logger = TestLogger::new();
        let mut aggregate = NumericMetricsAggregate::default();

        logger.log(500.); // Epoch 1
        logger.log(1000.); // Epoch 1
        logger.new_epoch();
        logger.log(200.); // Epoch 2
        logger.log(1000.); // Epoch 2
        logger.new_epoch();
        logger.log(10000.); // Epoch 3

        let value = aggregate
            .find_epoch(
                NAME,
                Aggregate::Mean,
                Direction::Lowest,
                &mut [Box::new(logger.logger)],
            )
            .unwrap();

        assert_eq!(value, 2);
    }
}
