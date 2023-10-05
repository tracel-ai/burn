use crate::{logger::MetricLogger, Aggregate, Direction};
use std::collections::HashMap;

/// Type that can be used to fetch and use numeric metric aggregates.
#[derive(Default)]
pub struct NumericMetricsAggregate {
    mean_for_each_epoch: HashMap<Key, f64>,
}

#[derive(new, Hash, PartialEq, Eq)]
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
    ) -> Result<f64, String> {
        let key = Key::new(name.to_string(), epoch);

        if let Some(value) = self.mean_for_each_epoch.get(&key) {
            return Ok(*value);
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

        let points = points()?;
        let num_points = points.len();
        let mean = points.into_iter().sum::<f64>() / num_points as f64;

        self.mean_for_each_epoch.insert(key, mean);
        Ok(mean)
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
                    Ok(value) => data.push(value),
                    Err(_) => break,
                },
            };

            current_epoch += 1;
        }

        if data.is_empty() {
            return None;
        }

        let mut current_value = match &direction {
            Direction::Lowest => f64::MAX,
            Direction::Hightest => f64::MIN,
        };

        for (epoch, value) in data.into_iter().enumerate() {
            match &direction {
                Direction::Lowest => {
                    if value < current_value {
                        current_value = value;
                        current_epoch = epoch;
                    }
                }
                Direction::Hightest => {
                    if value > current_value {
                        current_value = value;
                        current_epoch = epoch;
                    }
                }
            }
        }

        Some(current_epoch)
    }
}
