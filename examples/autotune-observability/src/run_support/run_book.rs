use std::path::PathBuf;

use super::ProblemKind;
use crate::example_dir;

/// A saved run configuration: one workload on one backend with a set of shapes and input/output
/// dtypes. A list of these forms a [`RunBook`] — a reusable batch the UI can launch in one click.
#[derive(Clone)]
pub(crate) struct RunSpec {
    /// Friendly label shown in the run book and used as the console source label when launched.
    pub name: String,
    pub backend: String,
    pub problem: ProblemKind,
    pub input: String,
    pub output: String,
    pub shapes: Vec<Vec<usize>>,
}

impl RunSpec {
    fn empty() -> Self {
        Self {
            name: String::new(),
            backend: String::new(),
            problem: ProblemKind::Matmul,
            input: String::from("f32"),
            output: String::from("f32"),
            shapes: Vec::new(),
        }
    }

    /// The shapes rendered as `1x2x3,4x5x6`, the format used both on disk and in the UI.
    pub fn shapes_string(&self) -> String {
        self.shapes
            .iter()
            .map(|shape| {
                shape
                    .iter()
                    .map(|dim| dim.to_string())
                    .collect::<Vec<_>>()
                    .join("x")
            })
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// A named, ordered list of run specs — e.g. one book of matmuls, one of attentions.
#[derive(Clone)]
pub(crate) struct RunBook {
    pub name: String,
    pub specs: Vec<RunSpec>,
}

impl RunBook {
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            specs: Vec::new(),
        }
    }
}

/// Every run book, persisted to `run-book.txt` next to the other UI config files. Always holds at
/// least one book so the UI has something to show.
#[derive(Clone)]
pub(crate) struct RunBooks {
    pub books: Vec<RunBook>,
}

impl Default for RunBooks {
    fn default() -> Self {
        Self {
            books: vec![RunBook::named("Default")],
        }
    }
}

fn book_path() -> PathBuf {
    example_dir().join("run-book.txt")
}

impl RunBooks {
    pub fn load() -> Self {
        match std::fs::read_to_string(book_path()) {
            Ok(text) => Self::parse(&text),
            Err(_) => Self::default(),
        }
    }

    /// Parse the block format written by [`RunBooks::to_text`]. `[book]` starts a new book and a
    /// following `name =` (before any `[run]`) names it; `[run]` starts a spec and its `key = value`
    /// lines fill it in. A file with `[run]` blocks but no `[book]` header (the old single-book
    /// format) lands in one "Default" book. Unknown keys and malformed lines are ignored so a
    /// hand-edited file degrades gracefully rather than dropping everything.
    fn parse(text: &str) -> Self {
        let mut books: Vec<RunBook> = Vec::new();
        let mut current: Option<RunSpec> = None;

        let flush = |books: &mut Vec<RunBook>, spec: Option<RunSpec>| {
            if let (Some(spec), Some(book)) = (spec, books.last_mut()) {
                book.specs.push(spec);
            }
        };

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if line == "[book]" {
                flush(&mut books, current.take());
                books.push(RunBook::named(String::new()));
                continue;
            }
            if line == "[run]" {
                flush(&mut books, current.take());
                if books.is_empty() {
                    books.push(RunBook::named("Default"));
                }
                current = Some(RunSpec::empty());
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            let value = value.trim();
            match (current.as_mut(), key.trim()) {
                (Some(spec), "name") => spec.name = value.to_string(),
                (Some(spec), "backend") => spec.backend = value.to_string(),
                (Some(spec), "problem") => {
                    if let Ok(problem) = ProblemKind::from_str(value) {
                        spec.problem = problem;
                    }
                }
                (Some(spec), "input") => spec.input = value.to_string(),
                (Some(spec), "output") => spec.output = value.to_string(),
                (Some(spec), "shapes") => spec.shapes = parse_shapes(value),
                (None, "name") => {
                    if let Some(book) = books.last_mut() {
                        book.name = value.to_string();
                    }
                }
                _ => {}
            }
        }
        flush(&mut books, current.take());

        if books.is_empty() {
            books.push(RunBook::named("Default"));
        }
        Self { books }
    }

    pub fn save(&self) {
        let _ = std::fs::write(book_path(), self.to_text());
    }

    fn to_text(&self) -> String {
        let mut text = String::new();
        for book in &self.books {
            text.push_str("[book]\n");
            text.push_str(&format!("name = {}\n\n", book.name));
            for spec in &book.specs {
                text.push_str("[run]\n");
                text.push_str(&format!("name = {}\n", spec.name));
                text.push_str(&format!("backend = {}\n", spec.backend));
                text.push_str(&format!("problem = {}\n", spec.problem.name()));
                text.push_str(&format!("input = {}\n", spec.input));
                text.push_str(&format!("output = {}\n", spec.output));
                text.push_str(&format!("shapes = {}\n\n", spec.shapes_string()));
            }
        }
        text
    }
}

fn parse_shapes(value: &str) -> Vec<Vec<usize>> {
    value
        .split(',')
        .map(str::trim)
        .filter(|shape| !shape.is_empty())
        .filter_map(|shape| {
            shape
                .split('x')
                .map(|dim| dim.parse::<usize>().ok())
                .collect::<Option<Vec<_>>>()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> RunBooks {
        RunBooks {
            books: vec![
                RunBook {
                    name: String::from("Matmuls"),
                    specs: vec![RunSpec {
                        name: String::from("square sweep"),
                        backend: String::from("cuda"),
                        problem: ProblemKind::Matmul,
                        input: String::from("f16"),
                        output: String::from("f16"),
                        shapes: vec![vec![512, 512, 512], vec![1024, 1024, 1024]],
                    }],
                },
                RunBook {
                    name: String::from("Attentions"),
                    specs: vec![RunSpec {
                        name: String::new(),
                        backend: String::from("wgpu"),
                        problem: ProblemKind::FlashAttention,
                        input: String::from("f32"),
                        output: String::from("f32"),
                        shapes: vec![vec![1, 256, 256]],
                    }],
                },
            ],
        }
    }

    #[test]
    fn round_trips_multiple_books() {
        let parsed = RunBooks::parse(&sample().to_text());
        assert_eq!(parsed.books.len(), 2);
        assert_eq!(parsed.books[0].name, "Matmuls");
        assert_eq!(parsed.books[0].specs[0].name, "square sweep");
        assert_eq!(parsed.books[0].specs[0].shapes, vec![vec![512, 512, 512], vec![1024, 1024, 1024]]);
        assert_eq!(parsed.books[1].name, "Attentions");
        assert_eq!(parsed.books[1].specs[0].problem.name(), "flash_attention");
        assert_eq!(parsed.books[1].specs[0].shapes, vec![vec![1, 256, 256]]);
    }

    #[test]
    fn reads_legacy_single_book_format() {
        let legacy = "[run]\nname = old\nbackend = cuda\nproblem = matmul\ninput = f32\noutput = f32\nshapes = 512x512x512\n";
        let parsed = RunBooks::parse(legacy);
        assert_eq!(parsed.books.len(), 1);
        assert_eq!(parsed.books[0].name, "Default");
        assert_eq!(parsed.books[0].specs[0].name, "old");
    }
}
