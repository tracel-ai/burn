use super::TerminalFrame;
use crate::metric::MetricEntry;
use ratatui::{
    prelude::{Alignment, Rect},
    style::{Color, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};
use std::{collections::HashMap, sync::Arc};

#[derive(Default)]
pub(crate) struct TextMetricsState {
    data: HashMap<String, MetricGroup>,
    names: Vec<String>,
}

struct MetricGroup {
    groups: HashMap<TuiGroup, MetricSplits>,
}

impl MetricGroup {
    fn new(group: TuiGroup, metric: MetricSplits) -> Self {
        Self {
            groups: HashMap::from_iter(Some((group, metric)).into_iter()),
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

#[derive(Hash, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TuiSplit {
    Train,
    Valid,
    Test,
}

#[derive(Hash, Clone, PartialEq, Eq)]
pub(crate) enum TuiGroup {
    Default,
    Named(Arc<String>),
}

impl core::fmt::Display for TuiGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TuiGroup::Default => f.write_str(""),
            TuiGroup::Named(group) => f.write_fmt(format_args!("{group} ")),
        }
    }
}

impl core::fmt::Display for TuiSplit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TuiSplit::Train => f.write_str("Train"),
            TuiSplit::Valid => f.write_str("Valid"),
            TuiSplit::Test => f.write_str("Test"),
        }
    }
}

impl TuiSplit {
    pub(crate) fn color(&self) -> Color {
        match self {
            TuiSplit::Train => Color::LightRed,
            TuiSplit::Valid => Color::LightBlue,
            TuiSplit::Test => Color::LightGreen,
        }
    }
}

struct MetricSplits {
    splits: HashMap<TuiSplit, MetricEntry>,
}

impl MetricSplits {
    fn new(split: TuiSplit, metric: MetricEntry) -> Self {
        Self {
            splits: HashMap::from_iter(Some((split, metric)).into_iter()),
        }
    }

    fn update(&mut self, split: TuiSplit, metric: MetricEntry) {
        self.splits.insert(split, metric);
    }
}

impl TextMetricsState {
    pub(crate) fn update(&mut self, split: TuiSplit, group: TuiGroup, metric: MetricEntry) {
        if let Some(existing) = self.data.get_mut(&metric.name) {
            existing.update(split, group, metric);
        } else {
            let key = metric.name.clone();
            let value = MetricSplits::new(split, metric);

            self.names.push(key.clone());
            self.data.insert(key, MetricGroup::new(group.into(), value));
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
    fn new(names: &[String], data: &HashMap<String, MetricGroup>) -> Self {
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

            let entry = data.get(name).unwrap();

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
