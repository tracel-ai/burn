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

/// Hit-test geometry and hover state for the metrics pane, populated by
/// `TextMetricView::render` on every frame and consumed by `on_event`.
#[derive(Default)]
pub(crate) struct TextHitState {
    hovered: Option<MetricName>,
    rect: Option<Rect>,
    header_rows: Vec<(MetricName, Range<u16>)>,
}

#[derive(Default)]
pub(crate) struct TextMetricsState {
    data: BTreeMap<String, MetricGroup>,
    names: Vec<MetricName>,
    pane: TextHitState,
}

/// What a mouse event meant for the left pane. Drives both selection routing
/// (the `Clicked` arm carries the metric name to switch to) and redraw gating
/// (anything other than `Ignored` should cause a redraw).
pub(crate) enum TextEventOutcome {
    Clicked(MetricName),
    HoverChanged,
    Ignored,
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
        TextMetricView::new(&self.names, &self.data, &mut self.pane)
    }

    /// Updates hover state and reports what the event meant for the left pane.
    pub(crate) fn on_event(&mut self, event: &Event) -> TextEventOutcome {
        let Event::Mouse(mouse) = event else {
            return TextEventOutcome::Ignored;
        };
        let pos = Position::new(mouse.column, mouse.row);
        let hit = if self.pane.rect.is_some_and(|pane| pane.contains(pos)) {
            self.pane
                .header_rows
                .iter()
                .find(|(_, rows)| rows.contains(&pos.y))
                .map(|(name, _)| name.clone())
        } else {
            None
        };

        match mouse.kind {
            MouseEventKind::Moved => {
                if self.pane.hovered == hit {
                    TextEventOutcome::Ignored
                } else {
                    self.pane.hovered = hit;
                    TextEventOutcome::HoverChanged
                }
            }
            MouseEventKind::Down(MouseButton::Left) => match hit {
                Some(name) => TextEventOutcome::Clicked(name),
                None => TextEventOutcome::Ignored,
            },
            _ => TextEventOutcome::Ignored,
        }
    }
}

pub(crate) struct TextMetricView<'a> {
    lines: Vec<Vec<Span<'static>>>,
    /// Index into `lines` of each metric's header row, in display order.
    header_line_indices: Vec<(MetricName, usize)>,
    pane: &'a mut TextHitState,
}

impl<'a> TextMetricView<'a> {
    fn new(
        names: &[MetricName],
        data: &BTreeMap<String, MetricGroup>,
        pane: &'a mut TextHitState,
    ) -> Self {
        let mut lines = Vec::with_capacity(names.len() * 4);
        let mut header_line_indices = Vec::with_capacity(names.len());

        let hovered = pane.hovered.as_ref();
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
            pane,
        }
    }

    pub(crate) fn render(self, frame: &mut TerminalFrame<'_>, size: Rect) {
        let Self {
            lines,
            header_line_indices,
            pane,
        } = self;

        // Skip the 1-cell top border. Header lines longer than the pane will
        // wrap and misplace the hit zone, accepted as a known limitation.
        let text_origin_y = size.y.saturating_add(1);
        pane.rect = Some(size);
        pane.header_rows = header_line_indices
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

    fn state_with(headers: Vec<(MetricName, Range<u16>)>) -> TextMetricsState {
        TextMetricsState {
            pane: TextHitState {
                rect: Some(Rect::new(0, 0, 20, 10)),
                header_rows: headers,
                ..TextHitState::default()
            },
            ..TextMetricsState::default()
        }
    }

    #[test]
    fn click_on_header_row_returns_clicked() {
        let mut state = state_with(vec![(name("Loss"), 1..2), (name("Accuracy"), 5..6)]);

        let outcome = state.on_event(&mouse(MouseEventKind::Down(MouseButton::Left), 3, 5));

        match outcome {
            TextEventOutcome::Clicked(n) => assert_eq!(n.as_str(), "Accuracy"),
            _ => panic!("expected Clicked"),
        }
    }

    #[test]
    fn click_off_any_header_returns_ignored() {
        let mut state = state_with(vec![(name("Loss"), 1..2)]);

        let on_data_row = state.on_event(&mouse(MouseEventKind::Down(MouseButton::Left), 3, 3));
        let outside_pane = state.on_event(&mouse(MouseEventKind::Down(MouseButton::Left), 50, 50));

        assert!(matches!(on_data_row, TextEventOutcome::Ignored));
        assert!(matches!(outside_pane, TextEventOutcome::Ignored));
    }

    #[test]
    fn moved_event_signals_hover_change_only_when_target_changes() {
        let mut state = state_with(vec![(name("Loss"), 1..2)]);

        // First move onto Loss: hover changes from None to Some(Loss).
        let r1 = state.on_event(&mouse(MouseEventKind::Moved, 3, 1));
        assert!(matches!(r1, TextEventOutcome::HoverChanged));
        assert_eq!(
            state.pane.hovered.as_deref().map(|s| s.as_str()),
            Some("Loss")
        );

        // Second move still on Loss: no change, ignored (no redraw needed).
        let r2 = state.on_event(&mouse(MouseEventKind::Moved, 4, 1));
        assert!(matches!(r2, TextEventOutcome::Ignored));

        // Move off the header: hover changes to None.
        let r3 = state.on_event(&mouse(MouseEventKind::Moved, 3, 8));
        assert!(matches!(r3, TextEventOutcome::HoverChanged));
        assert!(state.pane.hovered.is_none());
    }
}
