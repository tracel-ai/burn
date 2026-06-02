use crate::{
    logger::ProgressSnapshot,
    metric::{MetricName, NumericEntry},
    renderer::tui::TuiTag,
};

use super::{FullHistoryPlot, RecentHistoryPlot, TerminalFrame, TuiSplit};
use ratatui::{
    crossterm::event::{
        Event, KeyCode, KeyEvent, KeyEventKind, MouseButton, MouseEvent, MouseEventKind,
    },
    prelude::{Alignment, Constraint, Direction, Layout, Position, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::Line,
    widgets::{
        Axis, BarChart, BarGroup, Block, Borders, Chart, LegendPosition, Padding, Paragraph, Tabs,
        Widget,
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ChevronSide {
    Left,
    Right,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum HoverTarget {
    Tab(usize),
    Chevron(ChevronSide),
}

/// Hit-test geometry and hover state for the tab strip, populated by `render_tab_strip`
/// on every frame and consumed by `on_mouse_event`.
#[derive(Default)]
pub(crate) struct TabStripState {
    hovered: Option<HoverTarget>,
    tab_rects: Vec<Rect>,
    chevron_left: Option<Rect>,
    chevron_right: Option<Rect>,
}

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
    strip: TabStripState,
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
    pub(crate) fn update_progress_train(&mut self, progress: &ProgressSnapshot) {
        self.epoch = progress.global.items_processed;

        if self.num_samples_train.is_some() {
            return;
        }

        self.num_samples_train = Some(progress.split.items_total);
    }

    /// Update the state with the validation progress.
    pub(crate) fn update_progress_valid(&mut self, progress: &ProgressSnapshot) {
        if self.num_samples_valid.is_some() {
            return;
        }

        // If num_samples_train is None, keep the default max_samples for validation.
        if let Some(num_sample_train) = self.num_samples_train {
            for (_, (_recent, full)) in self.data.iter_mut() {
                let ratio = progress.split.items_total as f64 / num_sample_train as f64;
                full.update_max_sample(TuiSplit::Valid, ratio);
            }
        }

        self.epoch = progress.global.items_processed;
        self.num_samples_valid = Some(progress.split.items_total);
    }

    /// Update the state with the testing progress.
    pub(crate) fn update_progress_test(&mut self, progress: &ProgressSnapshot) {
        if self.num_samples_test.is_some() {
            return;
        }

        if let Some(num_sample_train) = self.num_samples_train {
            for (_, (_recent, full)) in self.data.iter_mut() {
                let ratio = progress.split.items_total as f64 / num_sample_train as f64;
                full.update_max_sample(TuiSplit::Test, ratio);
            }
        }

        self.num_samples_test = Some(progress.split.items_total);
    }

    /// Create a view to display the numeric metrics.
    pub(crate) fn view(&mut self) -> NumericMetricView<'_> {
        if self.names.is_empty() {
            return NumericMetricView::None;
        }
        match self.kind {
            PlotKind::Summary => {
                let chart = Self::bar_chart(&self.names, &self.data, self.selected);
                NumericMetricView::BarPlots {
                    titles: &self.names,
                    selected: self.selected,
                    chart,
                    strip: &mut self.strip,
                }
            }
            kind => {
                let chart = Self::line_chart(&self.names, &self.data, self.selected, kind);
                NumericMetricView::LinePlots {
                    titles: &self.names,
                    selected: self.selected,
                    chart,
                    kind,
                    strip: &mut self.strip,
                }
            }
        }
    }

    /// Handle the current event. Returns `true` when visible state changed and a
    /// redraw is warranted, `false` for events that produced no observable change
    /// (so the caller can skip an unnecessary redraw).
    pub(crate) fn on_event(&mut self, event: &Event) -> bool {
        match event {
            Event::Key(key) => self.on_key_event(key),
            Event::Mouse(mouse) => self.on_mouse_event(mouse),
            _ => false,
        }
    }

    fn on_key_event(&mut self, key: &KeyEvent) -> bool {
        match key.kind {
            KeyEventKind::Release | KeyEventKind::Repeat => (),
            #[cfg(target_os = "windows")] // Fix the double toggle on Windows.
            KeyEventKind::Press => return false,
            #[cfg(not(target_os = "windows"))]
            KeyEventKind::Press => (),
        }
        match key.code {
            KeyCode::Right => {
                self.next_metric();
                true
            }
            KeyCode::Left => {
                self.previous_metric();
                true
            }
            KeyCode::Up | KeyCode::Down => {
                self.switch_kind();
                true
            }
            _ => false,
        }
    }

    fn on_mouse_event(&mut self, mouse: &MouseEvent) -> bool {
        let pos = Position::new(mouse.column, mouse.row);
        let target = hover_target_at(&self.strip, pos);
        match mouse.kind {
            MouseEventKind::Moved => {
                if self.strip.hovered == target {
                    false
                } else {
                    self.strip.hovered = target;
                    true
                }
            }
            MouseEventKind::Down(MouseButton::Left) => match target {
                Some(HoverTarget::Tab(idx)) => {
                    self.selected = idx;
                    true
                }
                Some(HoverTarget::Chevron(ChevronSide::Left)) => {
                    self.previous_metric();
                    true
                }
                Some(HoverTarget::Chevron(ChevronSide::Right)) => {
                    self.next_metric();
                    true
                }
                None => false,
            },
            _ => false,
        }
    }

    pub(crate) fn select_by_name(&mut self, name: &MetricName) {
        if let Some(idx) = self.names.iter().position(|n| n == name) {
            self.selected = idx;
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

    fn line_chart<'a>(
        names: &'a [MetricName],
        data: &'a BTreeMap<MetricName, (RecentHistoryPlot, FullHistoryPlot)>,
        selected: usize,
        kind: PlotKind,
    ) -> Chart<'a> {
        let name = names.get(selected).unwrap();
        let (recent, full) = data.get(name).unwrap();

        let (datasets, axes) = match kind {
            PlotKind::Full => (full.datasets(), &full.axes),
            PlotKind::Recent => (recent.datasets(), &recent.axes),
            _ => unreachable!(),
        };

        Chart::<'a>::new(datasets)
            .block(Block::default())
            .x_axis(
                Axis::default()
                    .style(Style::default().fg(Color::DarkGray))
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

    fn bar_chart<'a>(
        names: &'a [MetricName],
        data: &'a BTreeMap<MetricName, (RecentHistoryPlot, FullHistoryPlot)>,
        selected: usize,
    ) -> BarChart<'a> {
        let name = names.get(selected).unwrap();
        let (_recent, full) = data.get(name).unwrap();
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
pub(crate) enum NumericMetricView<'a> {
    LinePlots {
        titles: &'a [MetricName],
        selected: usize,
        chart: Chart<'a>,
        kind: PlotKind,
        strip: &'a mut TabStripState,
    },
    BarPlots {
        titles: &'a [MetricName],
        selected: usize,
        chart: BarChart<'a>,
        strip: &'a mut TabStripState,
    },
    None,
}

impl NumericMetricView<'_> {
    pub(crate) fn render(self, frame: &mut TerminalFrame<'_>, size: Rect) {
        match self {
            Self::LinePlots {
                titles,
                selected,
                chart,
                kind,
                strip,
            } => {
                let plot_title = match kind {
                    PlotKind::Full => "Full History",
                    PlotKind::Recent => "Recent History",
                    _ => unreachable!(),
                };
                render_plot_panel(
                    frame, size, "Plots", plot_title, titles, selected, strip, chart,
                );
            }
            Self::BarPlots {
                titles,
                selected,
                chart,
                strip,
            } => {
                render_plot_panel(
                    frame, size, "Summary", "Summary", titles, selected, strip, chart,
                );
            }
            Self::None => {}
        }
    }
}

/// Draw the bordered plot panel: tab strip on top, centered plot title, then the chart.
#[allow(clippy::too_many_arguments)]
fn render_plot_panel<W: Widget>(
    frame: &mut TerminalFrame<'_>,
    size: Rect,
    block_title: &str,
    plot_title: &str,
    titles: &[MetricName],
    selected: usize,
    strip: &mut TabStripState,
    chart: W,
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

    render_tab_strip(frame, chunks[0], titles, selected, strip);
    let title = Paragraph::new(Line::from(plot_title.bold())).alignment(Alignment::Center);
    frame.render_widget(title, chunks[1]);
    frame.render_widget(chart, chunks[2]);
}

/// Render the metric tabs in `area`, scrolling horizontally so the `selected` tab is always
/// visible. A `‹` / `›` indicator is drawn in a reserved cell on each side when tabs are
/// hidden off that edge. The hovered tab gets an extra underline, a hovered chevron gets a
/// brighter foreground. Hit-test rects for the visible tabs and the two chevrons are written
/// back into `strip` so `on_mouse_event` can route clicks.
fn render_tab_strip(
    frame: &mut TerminalFrame<'_>,
    area: Rect,
    titles: &[MetricName],
    selected: usize,
    strip: &mut TabStripState,
) {
    strip.tab_rects.clear();
    strip.tab_rects.resize(titles.len(), Rect::default());
    strip.chevron_left = None;
    strip.chevron_right = None;

    if titles.is_empty() || area.width == 0 {
        return;
    }

    let titles_str: Vec<String> = titles.iter().map(|t| t.to_string()).collect();
    let widths: Vec<u16> = titles_str.iter().map(|s| tab_cell_width(s)).collect();

    let inner_width = area.width.saturating_sub(2);
    let (start, end) = visible_tab_window(&widths, selected, inner_width);

    if start > 0 {
        let left = Rect { width: 1, ..area };
        let color = chevron_color(strip.hovered, ChevronSide::Left);
        frame.render_widget(Paragraph::new("‹").style(Style::default().fg(color)), left);
        strip.chevron_left = Some(left);
    }
    if end < titles.len() {
        let right = Rect {
            x: area.x + area.width - 1,
            width: 1,
            ..area
        };
        let color = chevron_color(strip.hovered, ChevronSide::Right);
        frame.render_widget(Paragraph::new("›").style(Style::default().fg(color)), right);
        strip.chevron_right = Some(right);
    }

    let tabs_area = Rect {
        x: area.x + 1,
        width: inner_width,
        ..area
    };

    // Hit-test rects mirror the on-screen layout of the visible window. Tabs scrolled
    // off either edge keep the zero-sized default from `resize` above so the mouse test
    // misses them.
    let mut x = tabs_area.x;
    let tabs_end = tabs_area.x.saturating_add(tabs_area.width);
    for (i, &w) in (start..end).zip(&widths[start..end]) {
        let remaining = tabs_end.saturating_sub(x);
        strip.tab_rects[i] = Rect {
            x: x.min(tabs_end),
            y: tabs_area.y,
            width: w.min(remaining),
            height: tabs_area.height,
        };
        x = x.saturating_add(w).saturating_add(TAB_DIVIDER);
    }

    let tabs = Tabs::new(titles_str[start..end].iter().enumerate().map(|(local, s)| {
        let span = s.clone().yellow();
        let span = if strip.hovered == Some(HoverTarget::Tab(start + local)) {
            span.underlined()
        } else {
            span
        };
        Line::from(vec![span])
    }))
    .select(selected - start)
    .highlight_style(
        Style::default()
            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
            .fg(Color::LightYellow),
    );
    frame.render_widget(tabs, tabs_area);
}

fn chevron_color(hovered: Option<HoverTarget>, side: ChevronSide) -> Color {
    if hovered == Some(HoverTarget::Chevron(side)) {
        Color::Gray
    } else {
        Color::DarkGray
    }
}

fn hover_target_at(strip: &TabStripState, pos: Position) -> Option<HoverTarget> {
    if strip.chevron_left.is_some_and(|r| r.contains(pos)) {
        return Some(HoverTarget::Chevron(ChevronSide::Left));
    }
    if strip.chevron_right.is_some_and(|r| r.contains(pos)) {
        return Some(HoverTarget::Chevron(ChevronSide::Right));
    }
    strip
        .tab_rects
        .iter()
        .position(|r| r.contains(pos))
        .map(HoverTarget::Tab)
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
    use ratatui::crossterm::event::{KeyModifiers, MouseEvent};
    use std::sync::Arc;

    fn name(s: &str) -> MetricName {
        Arc::new(s.to_string())
    }

    fn mouse(kind: MouseEventKind, column: u16, row: u16) -> Event {
        Event::Mouse(MouseEvent {
            kind,
            column,
            row,
            modifiers: KeyModifiers::NONE,
        })
    }

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

    #[test]
    fn select_by_name_sets_index_on_match_and_no_ops_otherwise() {
        let mut state = NumericMetricsState {
            names: vec![name("loss"), name("acc"), name("lr")],
            ..NumericMetricsState::default()
        };

        state.select_by_name(&name("acc"));
        assert_eq!(state.selected, 1);

        state.select_by_name(&name("missing"));
        assert_eq!(state.selected, 1);
    }

    fn strip_with_tabs(tabs: Vec<Rect>) -> TabStripState {
        TabStripState {
            tab_rects: tabs,
            ..TabStripState::default()
        }
    }

    #[test]
    fn mouse_click_on_tab_selects_it() {
        let mut state = NumericMetricsState {
            names: vec![name("loss"), name("acc")],
            strip: strip_with_tabs(vec![Rect::new(0, 0, 6, 2), Rect::new(7, 0, 5, 2)]),
            ..NumericMetricsState::default()
        };

        state.on_event(&mouse(MouseEventKind::Down(MouseButton::Left), 8, 0));

        assert_eq!(state.selected, 1);
    }

    #[test]
    fn mouse_click_outside_tabs_does_not_change_selection() {
        let mut state = NumericMetricsState {
            names: vec![name("loss"), name("acc")],
            selected: 0,
            strip: strip_with_tabs(vec![Rect::new(0, 0, 6, 2), Rect::new(7, 0, 5, 2)]),
            ..NumericMetricsState::default()
        };

        state.on_event(&mouse(MouseEventKind::Down(MouseButton::Left), 50, 50));

        assert_eq!(state.selected, 0);
    }

    #[test]
    fn mouse_moved_updates_hovered() {
        let mut state = NumericMetricsState {
            names: vec![name("loss"), name("acc")],
            strip: strip_with_tabs(vec![Rect::new(0, 0, 6, 2), Rect::new(7, 0, 5, 2)]),
            ..NumericMetricsState::default()
        };

        state.on_event(&mouse(MouseEventKind::Moved, 2, 0));
        assert_eq!(state.strip.hovered, Some(HoverTarget::Tab(0)));

        state.on_event(&mouse(MouseEventKind::Moved, 50, 50));
        assert_eq!(state.strip.hovered, None);
    }

    #[test]
    fn mouse_click_on_left_chevron_goes_to_previous_metric() {
        // Three metrics with the second one selected. A click on the left chevron should
        // wrap-decrement the selection just like KeyCode::Left does.
        let mut state = NumericMetricsState {
            names: vec![name("a"), name("b"), name("c")],
            data: BTreeMap::from_iter([
                (
                    name("a"),
                    (RecentHistoryPlot::new(1), FullHistoryPlot::new(1)),
                ),
                (
                    name("b"),
                    (RecentHistoryPlot::new(1), FullHistoryPlot::new(1)),
                ),
                (
                    name("c"),
                    (RecentHistoryPlot::new(1), FullHistoryPlot::new(1)),
                ),
            ]),
            selected: 1,
            strip: TabStripState {
                chevron_left: Some(Rect::new(0, 0, 1, 2)),
                ..TabStripState::default()
            },
            ..NumericMetricsState::default()
        };

        state.on_event(&mouse(MouseEventKind::Down(MouseButton::Left), 0, 0));
        assert_eq!(state.selected, 0);
    }

    #[test]
    fn mouse_click_on_right_chevron_goes_to_next_metric() {
        let mut state = NumericMetricsState {
            names: vec![name("a"), name("b"), name("c")],
            data: BTreeMap::from_iter([
                (
                    name("a"),
                    (RecentHistoryPlot::new(1), FullHistoryPlot::new(1)),
                ),
                (
                    name("b"),
                    (RecentHistoryPlot::new(1), FullHistoryPlot::new(1)),
                ),
                (
                    name("c"),
                    (RecentHistoryPlot::new(1), FullHistoryPlot::new(1)),
                ),
            ]),
            selected: 1,
            strip: TabStripState {
                chevron_right: Some(Rect::new(20, 0, 1, 2)),
                ..TabStripState::default()
            },
            ..NumericMetricsState::default()
        };

        state.on_event(&mouse(MouseEventKind::Down(MouseButton::Left), 20, 0));
        assert_eq!(state.selected, 2);
    }

    #[test]
    fn mouse_moved_over_chevron_updates_hover_target() {
        let mut state = NumericMetricsState {
            strip: TabStripState {
                chevron_left: Some(Rect::new(0, 0, 1, 2)),
                chevron_right: Some(Rect::new(20, 0, 1, 2)),
                ..TabStripState::default()
            },
            ..NumericMetricsState::default()
        };

        state.on_event(&mouse(MouseEventKind::Moved, 0, 0));
        assert_eq!(
            state.strip.hovered,
            Some(HoverTarget::Chevron(ChevronSide::Left))
        );

        state.on_event(&mouse(MouseEventKind::Moved, 20, 0));
        assert_eq!(
            state.strip.hovered,
            Some(HoverTarget::Chevron(ChevronSide::Right))
        );
    }

    #[test]
    fn on_event_returns_false_for_mouse_move_that_does_not_change_hover() {
        let mut state = NumericMetricsState {
            names: vec![name("loss"), name("acc")],
            strip: TabStripState {
                tab_rects: vec![Rect::new(0, 0, 6, 2), Rect::new(7, 0, 5, 2)],
                ..TabStripState::default()
            },
            ..NumericMetricsState::default()
        };

        // First move onto tab 0: hover changes from None to Some(Tab(0)).
        assert!(state.on_event(&mouse(MouseEventKind::Moved, 2, 0)));
        // Second move still on tab 0: no change, no redraw needed.
        assert!(!state.on_event(&mouse(MouseEventKind::Moved, 3, 0)));
        // Move off any tab: hover changes back to None.
        assert!(state.on_event(&mouse(MouseEventKind::Moved, 50, 50)));
        // Move outside again: still None, no change.
        assert!(!state.on_event(&mouse(MouseEventKind::Moved, 60, 60)));
    }

    #[test]
    fn on_event_returns_true_for_tab_click_false_for_missed_click() {
        let mut state = NumericMetricsState {
            names: vec![name("loss"), name("acc")],
            strip: TabStripState {
                tab_rects: vec![Rect::new(0, 0, 6, 2), Rect::new(7, 0, 5, 2)],
                ..TabStripState::default()
            },
            ..NumericMetricsState::default()
        };

        assert!(state.on_event(&mouse(MouseEventKind::Down(MouseButton::Left), 8, 0)));
        assert!(!state.on_event(&mouse(MouseEventKind::Down(MouseButton::Left), 50, 50)));
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
