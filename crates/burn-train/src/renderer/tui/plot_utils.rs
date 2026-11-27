use crate::metric::format_float;
use ratatui::{
    layout::Rect,
    style::{Color, Style, Stylize},
    widgets::{Axis, Chart, Dataset, LegendPosition},
};

const AXIS_TITLE_PRECISION: usize = 2;

/// The data describing both X and Y axes.
pub(crate) struct PlotAxes {
    labels_x: Vec<String>,
    labels_y: Vec<String>,
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
    pub(crate) fn chart<'a>(&'a self, datasets: Vec<Dataset<'a>>) -> Chart<'a> {
        let axis = |labels: &'a [String], bounds| {
            Axis::default()
                .style(Style::default().fg(Color::DarkGray))
                .labels(labels.iter().map(|label| label.as_str().bold()))
                .bounds(bounds)
        };
        Chart::new(datasets)
            .x_axis(axis(&self.labels_x, self.bounds_x))
            .y_axis(axis(&self.labels_y, self.bounds_y))
            .legend_position(Some(LegendPosition::Right))
    }

    /// Ratatui's chart reserves up to a third of its width for left-aligned tick
    /// labels, then one column for the y axis and two rows for the x axis and labels.
    pub(crate) fn graph_area(&self, area: Rect) -> Rect {
        if self.labels_y.is_empty() {
            return area;
        }
        // Tick labels are ASCII numbers. The first x label ends beneath the y axis.
        let first_x = self.labels_x[0].len().saturating_sub(1);
        let label_width = self
            .labels_y
            .iter()
            .map(String::len)
            .max()
            .unwrap_or(0)
            .max(first_x);
        let labels = label_width.min(usize::from(area.width / 3)) as u16;
        let left = labels + u16::from(area.width - labels > 1);
        let bottom = 2.min(area.height.saturating_sub(1));
        Rect::new(
            area.x + left,
            area.y,
            area.width - left,
            area.height - bottom,
        )
    }

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
