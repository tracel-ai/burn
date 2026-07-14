use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::{BACKENDS, ProblemKind};
use crate::example_dir;

fn default_backend() -> String {
    BACKENDS[0].1.to_string()
}

fn default_dtype() -> String {
    String::from("f32")
}

/// A saved run configuration: one workload with a set of shapes and input/output dtypes. The
/// backend is inherited from the owning [`RunBook`] unless `backend` overrides it. A list of these
/// forms a book — a reusable batch the UI can launch.
#[derive(Clone)]
pub(crate) struct RunSpec {
    /// Friendly label shown in the book and used as the console source label when launched.
    pub name: String,
    /// Backend override; `None` means "use the book's backend".
    pub backend: Option<String>,
    pub problem: ProblemKind,
    pub input: String,
    pub output: String,
    pub shapes: Vec<Vec<usize>>,
}

impl RunSpec {
    /// The shapes rendered as `1x2x3,4x5x6`, the format used in the UI.
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

/// A named, ordered list of run specs with a default backend — e.g. one book of matmuls on cuda,
/// one of attentions on wgpu.
#[derive(Clone)]
pub(crate) struct RunBook {
    pub name: String,
    pub backend: String,
    pub specs: Vec<RunSpec>,
}

impl RunBook {
    pub fn new(name: impl Into<String>, backend: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            backend: backend.into(),
            specs: Vec::new(),
        }
    }

    /// The backend a given entry actually runs on: its override, or the book's default.
    pub fn effective_backend(&self, spec: &RunSpec) -> String {
        spec.backend.clone().unwrap_or_else(|| self.backend.clone())
    }
}

/// Every run book, persisted to `run-book.toml` next to the other UI config files. Always holds at
/// least one book so the UI has something to show.
#[derive(Clone)]
pub(crate) struct RunBooks {
    pub books: Vec<RunBook>,
}

impl Default for RunBooks {
    fn default() -> Self {
        Self {
            books: vec![RunBook::new("Default", default_backend())],
        }
    }
}

fn book_path() -> PathBuf {
    example_dir().join("run-book.toml")
}

impl RunBooks {
    pub fn load() -> Self {
        std::fs::read_to_string(book_path())
            .ok()
            .and_then(|text| Self::from_toml(&text))
            .filter(|books| !books.books.is_empty())
            .unwrap_or_default()
    }

    pub fn save(&self) {
        if let Ok(text) = toml::to_string_pretty(&self.to_file()) {
            let _ = std::fs::write(book_path(), text);
        }
    }

    fn to_file(&self) -> BooksFile {
        BooksFile {
            books: self.books.iter().map(BookDto::from_book).collect(),
        }
    }

    fn from_toml(text: &str) -> Option<Self> {
        let file: BooksFile = toml::from_str(text).ok()?;
        Some(Self {
            books: file.books.into_iter().map(BookDto::into_book).collect(),
        })
    }
}

/// The on-disk TOML shape. Each `[[book]]` holds scalar metadata plus a list of `[[book.run]]`
/// entries. Kept separate from the runtime types so `ProblemKind` and the backend-override
/// semantics can be mapped explicitly, and so missing fields degrade gracefully via `serde`
/// defaults rather than failing the whole parse.
#[derive(Serialize, Deserialize, Default)]
struct BooksFile {
    #[serde(default, rename = "book")]
    books: Vec<BookDto>,
}

#[derive(Serialize, Deserialize)]
struct BookDto {
    #[serde(default)]
    name: String,
    #[serde(default = "default_backend")]
    backend: String,
    #[serde(default, rename = "run")]
    runs: Vec<RunDto>,
}

#[derive(Serialize, Deserialize)]
struct RunDto {
    #[serde(default)]
    name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    backend: Option<String>,
    problem: String,
    #[serde(default = "default_dtype")]
    input: String,
    #[serde(default = "default_dtype")]
    output: String,
    /// Shapes as `"512x512x512"` strings — more readable in the file than nested integer arrays.
    #[serde(default)]
    shapes: Vec<String>,
}

fn parse_shape(text: &str) -> Vec<usize> {
    text.split('x').filter_map(|dim| dim.parse().ok()).collect()
}

fn shape_to_string(shape: &[usize]) -> String {
    shape.iter().map(|dim| dim.to_string()).collect::<Vec<_>>().join("x")
}

impl BookDto {
    fn from_book(book: &RunBook) -> Self {
        Self {
            name: book.name.clone(),
            backend: book.backend.clone(),
            runs: book.specs.iter().map(RunDto::from_spec).collect(),
        }
    }

    fn into_book(self) -> RunBook {
        RunBook {
            name: self.name,
            backend: self.backend,
            specs: self.runs.into_iter().map(RunDto::into_spec).collect(),
        }
    }
}

impl RunDto {
    fn from_spec(spec: &RunSpec) -> Self {
        Self {
            name: spec.name.clone(),
            backend: spec.backend.clone(),
            problem: spec.problem.name().to_string(),
            input: spec.input.clone(),
            output: spec.output.clone(),
            shapes: spec.shapes.iter().map(|shape| shape_to_string(shape)).collect(),
        }
    }

    fn into_spec(self) -> RunSpec {
        RunSpec {
            name: self.name,
            backend: self.backend,
            problem: ProblemKind::from_str(&self.problem).unwrap_or(ProblemKind::Matmul),
            input: self.input,
            output: self.output,
            shapes: self.shapes.iter().map(|shape| parse_shape(shape)).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> RunBooks {
        RunBooks {
            books: vec![
                RunBook {
                    name: String::from("Matmuls"),
                    backend: String::from("cuda"),
                    specs: vec![
                        RunSpec {
                            name: String::from("square sweep"),
                            backend: None,
                            problem: ProblemKind::Matmul,
                            input: String::from("f16"),
                            output: String::from("f16"),
                            shapes: vec![vec![512, 512, 512], vec![1024, 1024, 1024]],
                        },
                        RunSpec {
                            name: String::from("wgpu override"),
                            backend: Some(String::from("wgpu")),
                            problem: ProblemKind::Matmul,
                            input: String::from("f32"),
                            output: String::from("f32"),
                            shapes: vec![vec![256, 256, 256]],
                        },
                    ],
                },
                RunBook {
                    name: String::from("Attentions"),
                    backend: String::from("wgpu"),
                    specs: vec![RunSpec {
                        name: String::new(),
                        backend: None,
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
    fn round_trips_through_toml() {
        let toml = toml::to_string_pretty(&sample().to_file()).unwrap();
        let parsed = RunBooks::from_toml(&toml).unwrap();

        assert_eq!(parsed.books.len(), 2);
        let matmuls = &parsed.books[0];
        assert_eq!(matmuls.name, "Matmuls");
        assert_eq!(matmuls.backend, "cuda");
        assert_eq!(matmuls.specs[0].shapes, vec![vec![512, 512, 512], vec![1024, 1024, 1024]]);
        assert_eq!(matmuls.specs[0].backend, None);
        assert_eq!(matmuls.effective_backend(&matmuls.specs[0]), "cuda");
        assert_eq!(matmuls.specs[1].backend.as_deref(), Some("wgpu"));
        assert_eq!(matmuls.effective_backend(&matmuls.specs[1]), "wgpu");

        assert_eq!(parsed.books[1].backend, "wgpu");
        assert_eq!(parsed.books[1].specs[0].problem.name(), "flash_attention");
    }

    #[test]
    fn missing_optional_fields_use_defaults() {
        let toml = r#"
[[book]]
name = "Reduce"

[[book.run]]
problem = "reduce"
shapes = ["32x512x1024"]
"#;
        let parsed = RunBooks::from_toml(toml).unwrap();
        assert_eq!(parsed.books.len(), 1);
        let book = &parsed.books[0];
        assert_eq!(book.backend, default_backend());
        let spec = &book.specs[0];
        assert_eq!(spec.name, "");
        assert_eq!(spec.backend, None);
        assert_eq!(spec.input, "f32");
        assert_eq!(spec.output, "f32");
        assert_eq!(spec.problem.name(), "reduce");
    }
}


