pub(crate) struct BarChartPlot {
    train: MinMaxPoints,
    valid: MinMaxPoints,
    test: MinMaxPoints,
}

#[derive(new)]
struct MinMaxPoints {
    min: f64,
    max: f64,
}

impl BarChartPlot {
    /// Create a new bar chart plot.
    pub(crate) fn new() -> Self {
        Self {
            train: MinMaxPoints::new(0.0, 1.0),
            valid: MinMaxPoints::new(0.0, 1.0),
            test: MinMaxPoints::new(0.0, 1.0),
        }
    }

    /// Register a training data point.
    pub(crate) fn push_train(&mut self, data: f64) {
        self.train.min = f64::min(data, self.train.min);
        self.train.max = f64::min(data, self.train.max);
    }

    /// Register a valid data point.
    pub(crate) fn push_valid(&mut self, data: f64) {
        self.valid.min = f64::min(data, self.valid.min);
        self.valid.max = f64::max(data, self.valid.max);
    }

    /// Register a test data point.
    pub(crate) fn push_test(&mut self, data: f64) {
        self.test.min = f64::min(data, self.test.min);
        self.test.max = f64::max(data, self.test.max);
    }
}
