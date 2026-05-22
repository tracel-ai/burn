use crate::{
    metric::{MetricName, NumericEntry},
    renderer::{EvaluationProgress, TrainingProgress, tui::TuiTag},
};

use super::{FullHistoryPlot, RecentHistoryPlot, TerminalFrame, TuiSplit};
use ratatui::{
    crossterm::event::{Event, KeyCode, KeyEventKind},
    prelude::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::Line,
    widgets::{
        Axis, BarChart, BarGroup, Block, Borders, Chart, LegendPosition, Padding, Paragraph, Tabs,
    },
};
use std::collections::BTreeMap;
use unicode_width::UnicodeWidthStr;

/// 1 cell of padding on each side of a tab title, matching ratatui's default `Tabs` widget.
const TAB_PADDING: u16 = 2;
/// 1-cell `│` divider between adjacent tabs in ratatui's default `Tabs` widget.
const TAB_DIVIDER: u16 = 1;

/// 1000 seems to be required to see some improvement.
const MAX_NUM_SAMPLES_RECENT: usize = 1000;
/// 250 seems to be the right resolution when plotting all history.
/// Otherwise, there is too much points and the lines arent't smooth enough.
const MAX_NUM_SAMPLES_FULL: usize = 250;

/// Numeric metrics state that handles creating plots.
#[derive(Default)]
pub(crate) struct NumericMetricsState {
    data: BTreeMap<MetricName, (RecentHistoryPlot, FullHistoryPlot)>,
    names: Vec<MetricName>,
    selected: usize,
    kind: PlotKind,
    num_samples_train: Option<usize>,
    num_samples_valid: Option<usize>,
    num_samples_test: Option<usize>,
    epoch: usize,
}

/// The kind of plot to display.
#[derive(Default, Clone, Copy)]
pub(crate) enum PlotKind {
    /// Display the full history of the metric with reduced resolution.
    #[default]
    Full,
    /// Display only the recent history of the metric, but with more resolution.
    Recent,
    Summary,
}

impl NumericMetricsState {
    /// Register a new training value for the metric with the given name.
    pub(crate) fn push(&mut self, tag: TuiTag, name: MetricName, data: NumericEntry) {
        if let Some((recent, full)) = self.data.get_mut(name.as_ref()) {
            recent.push(tag.clone(), data.current());
            full.push(tag, data);
        } else {
            let mut recent = RecentHistoryPlot::new(MAX_NUM_SAMPLES_RECENT);
            let mut full = FullHistoryPlot::new(MAX_NUM_SAMPLES_FULL);

            recent.push(tag.clone(), data.current());
            full.push(tag, data);

            self.names.push(name.clone());
            self.data.insert(name, (recent, full));
        }
    }

    /// Update the state with the training progress.
    pub(crate) fn update_progress_train(&mut self, progress: &TrainingProgress) {
        self.epoch = progress.global_progress.items_processed;

        if self.num_samples_train.is_some() {
            return;
        }

        // If the training only has the notion of global progress, num_samples_train remains None.
        self.num_samples_train = progress.progress.as_ref().map(|p| p.items_total);
    }

    /// Update the state with the validation progress.
    pub(crate) fn update_progress_valid(&mut self, progress: &TrainingProgress) {
        if self.num_samples_valid.is_some() {
            return;
        }

        // If num_samples_train is None, keep the default max_samples for validation.
        if let Some(num_sample_train) = self.num_samples_train {
            for (_, (_recent, full)) in self.data.iter_mut() {
                let ratio = match &progress.progress {
                    Some(p) => p.items_total as f64 / num_sample_train as f64,
                    None => progress.global_progress.items_total as f64 / num_sample_train as f64,
                };

                full.update_max_sample(TuiSplit::Valid, ratio);
            }
        }

        self.epoch = progress.global_progress.items_processed;
        self.num_samples_valid = progress.progress.as_ref().map(|p| p.items_total);
    }

    /// Update the state with the testing progress.
    pub(crate) fn update_progress_test(&mut self, progress: &EvaluationProgress) {
        if self.num_samples_test.is_some() {
            return;
        }

        if let Some(num_sample_train) = self.num_samples_train {
            for (_, (_recent, full)) in self.data.iter_mut() {
                let ratio = progress.progress.items_total as f64 / num_sample_train as f64;
                full.update_max_sample(TuiSplit::Test, ratio);
            }
        }

        self.num_samples_test = Some(progress.progress.items_total);
    }

    /// Create a view to display the numeric metrics.
    pub(crate) fn view(&self) -> NumericMetricView<'_> {
        match self.names.is_empty() {
            true => NumericMetricView::None,
            false => match self.kind {
                PlotKind::Summary => {
                    NumericMetricView::BarPlots(&self.names, self.selected, self.bar_chart())
                }
                _ => NumericMetricView::LinePlots(
                    &self.names,
                    self.selected,
                    self.line_chart(),
                    self.kind,
                ),
            },
        }
    }

    /// Handle the current event.
    pub(crate) fn on_event(&mut self, event: &Event) {
        if let Event::Key(key) = event {
            match key.kind {
                KeyEventKind::Release | KeyEventKind::Repeat => (),
                #[cfg(target_os = "windows")] // Fix the double toggle on Windows.
                KeyEventKind::Press => return,
                #[cfg(not(target_os = "windows"))]
                KeyEventKind::Press => (),
            }
            match key.code {
                KeyCode::Right => self.next_metric(),
                KeyCode::Left => self.previous_metric(),
                KeyCode::Up => self.switch_kind(),
                KeyCode::Down => self.switch_kind(),
                _ => {}
            }
        }
    }

    fn switch_kind(&mut self) {
        self.kind = match self.kind {
            PlotKind::Full => PlotKind::Recent,
            PlotKind::Recent => PlotKind::Summary,
            PlotKind::Summary => PlotKind::Full,
        };
    }

    fn next_metric(&mut self) {
        let len = self.data.len();
        if len == 0 {
            return;
        }
        self.selected = (self.selected + 1) % len;
    }

    fn previous_metric(&mut self) {
        let len = self.data.len();
        if len == 0 {
            return;
        }
        if self.selected > 0 {
            self.selected -= 1;
        } else {
            self.selected = len - 1;
        }
    }

    fn line_chart<'a>(&'a self) -> Chart<'a> {
        let name = self.names.get(self.selected).unwrap();
        let (recent, full) = self.data.get(name).unwrap();

        let (datasets, axes) = match self.kind {
            PlotKind::Full => (full.datasets(), &full.axes),
            PlotKind::Recent => (recent.datasets(), &recent.axes),
            _ => unreachable!(),
        };

        Chart::<'a>::new(datasets)
            .block(Block::default())
            .x_axis(
                Axis::default()
                    .style(Style::default().fg(Color::DarkGray))
                    .title("Iteration")
                    .labels(axes.labels_x.clone().into_iter().map(|s| s.bold()))
                    .bounds(axes.bounds_x),
            )
            .y_axis(
                Axis::default()
                    .style(Style::default().fg(Color::DarkGray))
                    .labels(axes.labels_y.clone().into_iter().map(|s| s.bold()))
                    .bounds(axes.bounds_y),
            )
            .legend_position(Some(LegendPosition::Right))
    }

    fn bar_chart<'a>(&'a self) -> BarChart<'a> {
        let name = self.names.get(self.selected).unwrap();
        let (_recent, full) = self.data.get(name).unwrap();
        let mut bar_width = 0;
        let bars = full.bars(100, &mut bar_width);

        let data = BarGroup::default().bars(&bars);
        BarChart::default()
            .block(Block::default().padding(Padding::new(2, 2, 2, 0)))
            .bar_width(bar_width as u16)
            .bar_gap(2)
            .data(data)
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(new)]
pub(crate) enum NumericMetricView<'a> {
    LinePlots(&'a [MetricName], usize, Chart<'a>, PlotKind),
    BarPlots(&'a [MetricName], usize, BarChart<'a>),
    None,
}

impl NumericMetricView<'_> {
    pub(crate) fn render(self, frame: &mut TerminalFrame<'_>, size: Rect) {
        match self {
            Self::LinePlots(titles, selected, chart, kind) => {
                let plot_title = match kind {
                    PlotKind::Full => "Full History",
                    PlotKind::Recent => "Recent History",
                    _ => unreachable!(),
                };
                render_plot_panel(
                    frame,
                    size,
                    "Plots",
                    plot_title,
                    titles,
                    selected,
                    |f, a| f.render_widget(chart, a),
                );
            }
            Self::BarPlots(titles, selected, chart) => {
                render_plot_panel(
                    frame,
                    size,
                    "Summary",
                    "Summary",
                    titles,
                    selected,
                    |f, a| f.render_widget(chart, a),
                );
            }
            Self::None => {}
        }
    }
}

/// Draw the bordered plot panel: tab strip on top, centered plot title, then the chart.
fn render_plot_panel(
    frame: &mut TerminalFrame<'_>,
    size: Rect,
    block_title: &str,
    plot_title: &str,
    titles: &[MetricName],
    selected: usize,
    render_chart: impl FnOnce(&mut TerminalFrame<'_>, Rect),
) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(block_title)
        .title_alignment(Alignment::Left);
    let inner = block.inner(size);
    frame.render_widget(block, size);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Length(1),
            Constraint::Min(0),
        ])
        .split(inner);

    render_tab_strip(frame, chunks[0], titles, selected);
    let title = Paragraph::new(Line::from(plot_title.bold())).alignment(Alignment::Center);
    frame.render_widget(title, chunks[1]);
    render_chart(frame, chunks[2]);
}

/// Render the metric tabs in `area`, scrolling horizontally so the `selected` tab is always
/// visible. A `‹` / `›` indicator is drawn in a reserved cell on each side when tabs are
/// hidden off that edge.
fn render_tab_strip(
    frame: &mut TerminalFrame<'_>,
    area: Rect,
    titles: &[MetricName],
    selected: usize,
) {
    if titles.is_empty() || area.width == 0 {
        return;
    }

    let titles_str: Vec<String> = titles.iter().map(|t| t.to_string()).collect();
    let widths: Vec<u16> = titles_str.iter().map(|s| tab_cell_width(s)).collect();

    let inner_width = area.width.saturating_sub(2);
    let (start, end) = visible_tab_window(&widths, selected, inner_width);

    let edge_style = Style::default().fg(Color::DarkGray);
    if start > 0 {
        let left = Rect { width: 1, ..area };
        frame.render_widget(Paragraph::new("‹").style(edge_style), left);
    }
    if end < titles.len() {
        let right = Rect {
            x: area.x + area.width - 1,
            width: 1,
            ..area
        };
        frame.render_widget(Paragraph::new("›").style(edge_style), right);
    }

    let tabs_area = Rect {
        x: area.x + 1,
        width: inner_width,
        ..area
    };
    let tabs = Tabs::new(
        titles_str[start..end]
            .iter()
            .map(|s| Line::from(vec![s.clone().yellow()])),
    )
    .select(selected - start)
    .highlight_style(
        Style::default()
            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
            .fg(Color::LightYellow),
    );
    frame.render_widget(tabs, tabs_area);
}

/// Cells consumed by one tab. Title display width plus ratatui's default padding.
fn tab_cell_width(title: &str) -> u16 {
    u16::try_from(UnicodeWidthStr::width(title) + TAB_PADDING as usize).unwrap_or(u16::MAX)
}

/// Pick the `[start, end)` slice of `widths` to render so the tab at `selected` is visible
/// inside `available` cells. The selected tab is pinned as far right as fits. `end` is then
/// grown rightward as far as the remaining space allows. Always returns
/// `start <= selected < end` when `widths` is non-empty. If a single tab exceeds
/// `available`, clipping is delegated to ratatui's `Tabs`.
fn visible_tab_window(widths: &[u16], selected: usize, available: u16) -> (usize, usize) {
    if widths.is_empty() {
        return (0, 0);
    }
    let selected = selected.min(widths.len() - 1);
    let available = available as u32;
    let divider = TAB_DIVIDER as u32;

    // Width of titles[start..=selected] including dividers between them. Maintained
    // incrementally so the windowing loop is O(N) over the full title list.
    let mut width: u32 =
        widths[..=selected].iter().map(|&w| w as u32).sum::<u32>() + selected as u32 * divider;

    let mut start = 0;
    while width > available && start < selected {
        width -= widths[start] as u32 + divider;
        start += 1;
    }

    let mut end = selected + 1;
    while end < widths.len() {
        let next = width + widths[end] as u32 + divider;
        if next > available {
            break;
        }
        width = next;
        end += 1;
    }

    (start, end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_navigation_on_empty_state_is_a_no_op() {
        // Pressing Right or Left before any metric has been recorded must not
        // panic. Previously this triggered `(0 + 1) % 0` and `0usize - 1` on
        // the empty `data` map.
        let mut state = NumericMetricsState::default();

        state.next_metric();
        state.previous_metric();

        assert_eq!(state.selected, 0);
        assert!(state.data.is_empty());
    }

    fn cells(titles: &[&str]) -> Vec<u16> {
        titles.iter().copied().map(tab_cell_width).collect()
    }

    #[test]
    fn tab_cell_width_includes_padding() {
        assert_eq!(tab_cell_width("Loss"), 4 + TAB_PADDING);
        assert_eq!(tab_cell_width(""), TAB_PADDING);
    }

    #[test]
    fn visible_window_is_empty_when_no_tabs() {
        assert_eq!(visible_tab_window(&[], 0, 80), (0, 0));
    }

    #[test]
    fn visible_window_returns_full_range_when_everything_fits() {
        let widths = cells(&["Loss", "Acc", "F1"]);
        assert_eq!(visible_tab_window(&widths, 1, 80), (0, widths.len()));
    }

    #[test]
    fn visible_window_pins_selected_to_right_edge_when_growing_from_left() {
        // Each tab cell is 2 + TAB_PADDING = 4. One divider between = 5 per added tab.
        // available = 9 means exactly two tabs fit ([start..=selected] = 4 + 1 + 4 = 9).
        let widths = cells(&["AA", "BB", "CC", "DD", "EE"]);
        assert_eq!(visible_tab_window(&widths, 3, 9), (2, 4));
    }

    #[test]
    fn visible_window_does_not_panic_when_single_tab_is_wider_than_available() {
        let widths = cells(&["this title is much wider than the cell", "B"]);
        let (start, end) = visible_tab_window(&widths, 0, 4);
        assert_eq!(start, 0);
        assert!(end >= 1);
    }

    #[test]
    fn visible_window_clamps_out_of_range_selected() {
        let widths = cells(&["A", "B", "C"]);
        let (start, end) = visible_tab_window(&widths, 99, 80);
        assert!(start <= 2 && 2 < end);
    }
}
