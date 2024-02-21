use super::TerminalFrame;
use crate::metric::MetricEntry;
use ratatui::{
    prelude::{Alignment, Rect},
    style::{Color, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};
use std::collections::HashMap;

#[derive(Default)]
pub(crate) struct TextMetricsState {
    data: HashMap<String, MetricData>,
    names: Vec<String>,
}

#[derive(new)]
pub(crate) struct MetricData {
    train: Option<MetricEntry>,
    valid: Option<MetricEntry>,
}

impl TextMetricsState {
    pub(crate) fn update_train(&mut self, metric: MetricEntry) {
        if let Some(existing) = self.data.get_mut(&metric.name) {
            existing.train = Some(metric);
        } else {
            let key = metric.name.clone();
            let value = MetricData::new(Some(metric), None);

            self.names.push(key.clone());
            self.data.insert(key, value);
        }
    }
    pub(crate) fn update_valid(&mut self, metric: MetricEntry) {
        if let Some(existing) = self.data.get_mut(&metric.name) {
            existing.valid = Some(metric);
        } else {
            let key = metric.name.clone();
            let value = MetricData::new(None, Some(metric));

            self.names.push(key.clone());
            self.data.insert(key, value);
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
    fn new(names: &[String], data: &HashMap<String, MetricData>) -> Self {
        let mut lines = Vec::with_capacity(names.len() * 4);

        let start_line = |title: &str| vec![Span::from(format!(" {title} ")).bold().yellow()];
        let train_line = |formatted: &str| {
            vec![
                Span::from("   Train ").bold(),
                Span::from(formatted.to_string()).italic(),
            ]
        };
        let valid_line = |formatted: &str| {
            vec![
                Span::from("   Valid ").bold(),
                Span::from(formatted.to_string()).italic(),
            ]
        };

        for name in names {
            lines.push(start_line(name));

            let entry = data.get(name).unwrap();

            if let Some(entry) = &entry.train {
                lines.push(train_line(&entry.formatted));
            }

            if let Some(entry) = &entry.valid {
                lines.push(valid_line(&entry.formatted));
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
