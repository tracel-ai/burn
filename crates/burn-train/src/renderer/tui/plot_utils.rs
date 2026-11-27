use crate::metric::format_float;

const AXIS_TITLE_PRECISION: usize = 2;

/// The data describing both X and Y axes.
pub(crate) struct PlotAxes {
    pub(crate) labels_x: Vec<String>,
    pub(crate) labels_y: Vec<String>,
    pub(crate) bounds_x: [f64; 2],
    pub(crate) bounds_y: [f64; 2],
}

impl Default for PlotAxes {
    fn default() -> Self {
        Self {
            bounds_x: [f64::MAX, f64::MIN],
            bounds_y: [f64::MAX, f64::MIN],
            labels_x: Vec::new(),
            labels_y: Vec::new(),
        }
    }
}

impl PlotAxes {
    /// Update the bounds based on the min max of each X and Y axes with both train and valid data.
    pub(crate) fn update_bounds(&mut self, (x_min, x_max): (f64, f64), (y_min, y_max): (f64, f64)) {
        self.bounds_x = [x_min, x_max];
        self.bounds_y = [y_min, y_max];

        // We know x are integers.
        self.labels_x = vec![format!("{x_min}"), format!("{x_max}")];
        self.labels_y = vec![
            format_float(y_min, AXIS_TITLE_PRECISION),
            format_float(y_max, AXIS_TITLE_PRECISION),
        ];
    }
}
