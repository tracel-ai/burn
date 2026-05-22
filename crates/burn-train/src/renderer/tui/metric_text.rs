use super::TerminalFrame;
use crate::{
    metric::{MetricEntry, MetricName},
    renderer::tui::{TuiGroup, TuiSplit},
};
use ratatui::{
    crossterm::event::{Event, MouseButton, MouseEventKind},
    prelude::{Alignment, Position, Rect},
    style::{Color, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};
use std::{collections::BTreeMap, ops::Range, sync::Arc};

#[derive(Default)]
pub(crate) struct TextMetricsState {
    data: BTreeMap<String, MetricGroup>,
    names: Vec<MetricName>,
    hovered: Option<MetricName>,
    last_pane_rect: Option<Rect>,
    last_header_rows: Vec<(MetricName, Range<u16>)>,
}

struct MetricGroup {
    groups: BTreeMap<TuiGroup, MetricSplits>,
}

impl MetricGroup {
    fn new(group: TuiGroup, metric: MetricSplits) -> Self {
        Self {
            groups: BTreeMap::from_iter(Some((group, metric))),
        }
    }
    fn update(&mut self, split: TuiSplit, group: TuiGroup, metric: MetricEntry) {
        match self.groups.get_mut(&group) {
            Some(value) => value.update(split, metric),
            None => {
                let value = MetricSplits::new(split, metric);

                self.groups.insert(group, value);
            }
        }
    }
}

struct MetricSplits {
    splits: BTreeMap<TuiSplit, MetricEntry>,
}

impl MetricSplits {
    fn new(split: TuiSplit, metric: MetricEntry) -> Self {
        Self {
            splits: BTreeMap::from_iter(Some((split, metric))),
        }
    }

    fn update(&mut self, split: TuiSplit, metric: MetricEntry) {
        self.splits.insert(split, metric);
    }
}

impl TextMetricsState {
    pub(crate) fn update(
        &mut self,
        split: TuiSplit,
        group: TuiGroup,
        metric: MetricEntry,
        name: Arc<String>,
    ) {
        if let Some(existing) = self.data.get_mut(name.as_ref()) {
            existing.update(split, group, metric);
        } else {
            let key = name.clone();
            let value = MetricSplits::new(split, metric);

            self.names.push(key.clone());
            self.data
                .insert(key.to_string(), MetricGroup::new(group, value));
        }
    }

    pub(crate) fn view(&mut self) -> TextMetricView<'_> {
        TextMetricView::new(
            &self.names,
            &self.data,
            self.hovered.as_ref(),
            &mut self.last_pane_rect,
            &mut self.last_header_rows,
        )
    }

    /// Updates hover state and returns the clicked metric name, if any.
    pub(crate) fn on_event(&mut self, event: &Event) -> Option<MetricName> {
        let Event::Mouse(mouse) = event else {
            return None;
        };
        let pos = Position::new(mouse.column, mouse.row);
        let hit = if self.last_pane_rect.is_some_and(|pane| pane.contains(pos)) {
            self.last_header_rows
                .iter()
                .find(|(_, rows)| rows.contains(&pos.y))
                .map(|(name, _)| name.clone())
        } else {
            None
        };

        match mouse.kind {
            MouseEventKind::Moved => {
                self.hovered = hit;
                None
            }
            MouseEventKind::Down(MouseButton::Left) => hit,
            _ => None,
        }
    }
}

pub(crate) struct TextMetricView<'a> {
    lines: Vec<Vec<Span<'static>>>,
    /// Index into `lines` of each metric's header row, in display order.
    header_line_indices: Vec<(MetricName, usize)>,
    pane_rect_out: &'a mut Option<Rect>,
    header_rows_out: &'a mut Vec<(MetricName, Range<u16>)>,
}

impl<'a> TextMetricView<'a> {
    fn new(
        names: &[MetricName],
        data: &BTreeMap<String, MetricGroup>,
        hovered: Option<&MetricName>,
        pane_rect_out: &'a mut Option<Rect>,
        header_rows_out: &'a mut Vec<(MetricName, Range<u16>)>,
    ) -> Self {
        let mut lines = Vec::with_capacity(names.len() * 4);
        let mut header_line_indices = Vec::with_capacity(names.len());

        let start_line = |title: &str, is_hovered: bool| {
            let span = Span::from(format!(" {title} ")).bold().yellow();
            let span = if is_hovered { span.underlined() } else { span };
            vec![span]
        };
        let format_line = |group: &TuiGroup, split: &TuiSplit, formatted: &str| {
            vec![
                Span::from(format!(" {group}{split} ")).bold(),
                Span::from(formatted.to_string()).italic(),
            ]
        };

        for name in names {
            let is_hovered = hovered.is_some_and(|h| h.as_ref() == name.as_ref());
            header_line_indices.push((name.clone(), lines.len()));
            lines.push(start_line(name, is_hovered));

            let entry = data.get(name.as_ref()).unwrap();

            for (name, group) in entry.groups.iter() {
                for (split, entry) in group.splits.iter() {
                    lines.push(format_line(name, split, &entry.serialized_entry.formatted));
                }
            }

            lines.push(vec![Span::from("")]);
        }

        Self {
            lines,
            header_line_indices,
            pane_rect_out,
            header_rows_out,
        }
    }

    pub(crate) fn render(self, frame: &mut TerminalFrame<'_>, size: Rect) {
        let Self {
            lines,
            header_line_indices,
            pane_rect_out,
            header_rows_out,
        } = self;

        // Skip the 1-cell top border. Header lines longer than the pane will
        // wrap and misplace the hit zone, accepted as a known limitation.
        let text_origin_y = size.y.saturating_add(1);
        *pane_rect_out = Some(size);
        *header_rows_out = header_line_indices
            .into_iter()
            .map(|(name, line_idx)| {
                let row = text_origin_y.saturating_add(line_idx as u16);
                (name, row..row.saturating_add(1))
            })
            .collect();

        let paragraph = Paragraph::new(lines.into_iter().map(Line::from).collect::<Vec<_>>())
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: false })
            .block(Block::default().borders(Borders::ALL).title("Metrics"))
            .style(Style::default().fg(Color::Gray));

        frame.render_widget(paragraph, size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::crossterm::event::{KeyModifiers, MouseEvent};

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
    fn click_on_header_row_returns_metric_name() {
        let mut state = TextMetricsState {
            last_pane_rect: Some(Rect::new(0, 0, 20, 10)),
            last_header_rows: vec![(name("Loss"), 1..2), (name("Accuracy"), 5..6)],
            ..TextMetricsState::default()
        };

        let clicked = state.on_event(&mouse(MouseEventKind::Down(MouseButton::Left), 3, 5));

        assert_eq!(clicked.as_deref().map(|s| s.as_str()), Some("Accuracy"));
    }

    #[test]
    fn click_off_any_header_returns_none() {
        let mut state = TextMetricsState {
            last_pane_rect: Some(Rect::new(0, 0, 20, 10)),
            last_header_rows: vec![(name("Loss"), 1..2)],
            ..TextMetricsState::default()
        };

        let on_data_row = state.on_event(&mouse(MouseEventKind::Down(MouseButton::Left), 3, 3));
        let outside_pane = state.on_event(&mouse(MouseEventKind::Down(MouseButton::Left), 50, 50));

        assert!(on_data_row.is_none());
        assert!(outside_pane.is_none());
    }

    #[test]
    fn moved_event_updates_hovered_and_returns_none() {
        let mut state = TextMetricsState {
            last_pane_rect: Some(Rect::new(0, 0, 20, 10)),
            last_header_rows: vec![(name("Loss"), 1..2)],
            ..TextMetricsState::default()
        };

        let result = state.on_event(&mouse(MouseEventKind::Moved, 3, 1));
        assert!(result.is_none());
        assert_eq!(state.hovered.as_deref().map(|s| s.as_str()), Some("Loss"));

        let result = state.on_event(&mouse(MouseEventKind::Moved, 3, 8));
        assert!(result.is_none());
        assert!(state.hovered.is_none());
    }
}
