use super::TerminalFrame;
use crate::{
    metric::{MetricEntry, MetricName},
    renderer::tui::{TuiGroup, TuiSplit},
};
use ratatui::{
    prelude::{Alignment, Rect},
    style::{Color, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};
use std::collections::BTreeMap;

#[derive(Default)]
pub(crate) struct TextMetricsState {
    data: BTreeMap<String, MetricGroup>,
    names: Vec<MetricName>,
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
    pub(crate) fn update(&mut self, split: TuiSplit, group: TuiGroup, metric: MetricEntry) {
        if let Some(existing) = self.data.get_mut(metric.name.as_ref()) {
            existing.update(split, group, metric);
        } else {
            let key = metric.name.clone();
            let value = MetricSplits::new(split, metric);

            self.names.push(key.clone());
            self.data
                .insert(key.to_string(), MetricGroup::new(group, value));
        }
    }
    pub(crate) fn view(&self) -> TextMetricView {
        TextMetricView::new(&self.names, &self.data)
    }
}

pub(crate) struct TextMetricView {
    lines: Vec<Vec<Span<'static>>>,
}

impl TextMetricView {
    fn new(names: &[MetricName], data: &BTreeMap<String, MetricGroup>) -> Self {
        let mut lines = Vec::with_capacity(names.len() * 4);

        let start_line = |title: &str| vec![Span::from(format!(" {title} ")).bold().yellow()];
        let format_line = |group: &TuiGroup, split: &TuiSplit, formatted: &str| {
            vec![
                Span::from(format!(" {group}{split} ")).bold(),
                Span::from(formatted.to_string()).italic(),
            ]
        };

        for name in names {
            lines.push(start_line(name));

            let entry = data.get(name.as_ref()).unwrap();

            for (name, group) in entry.groups.iter() {
                for (split, entry) in group.splits.iter() {
                    lines.push(format_line(name, split, &entry.formatted));
                }
            }

            lines.push(vec![Span::from("")]);
        }

        Self { lines }
    }

    pub(crate) fn render(self, frame: &mut TerminalFrame<'_>, size: Rect) {
        let paragraph = Paragraph::new(self.lines.into_iter().map(Line::from).collect::<Vec<_>>())
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: false })
            .block(Block::default().borders(Borders::ALL).title("Metrics"))
            .style(Style::default().fg(Color::Gray));

        frame.render_widget(paragraph, size);
    }
}
