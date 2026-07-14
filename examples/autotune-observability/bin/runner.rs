//! Headless one-shot workload runner.
//!
//! The UI shells out to this binary once per "Run" so each workload happens in a fresh process:
//! the in-memory autotune cache starts empty every time, so the same config re-tunes (and
//! re-logs) on every run. It also keeps burn out of the UI binary, so the UI rebuilds fast.
//!
//! Usage:
//!   runner --list-backends
//!   runner --backend <name> --problem <matmul|attention> \
//!          --shape <DxExF> [--shape <DxExF> ...] \
//!          --input <dtype> --output <dtype> [--run-dir <path>]
//!
//! With `--run-dir`, a per-run `cubecl.toml` is written there so the autotune log and cache
//! land inside that directory (the UI passes one directory per run). Without it, the crate's
//! fallback `cubecl.toml` is used.

use std::path::PathBuf;

use autotune_observability::{ProblemKind, example_dir, write_run_config};
use burn::tensor::{
    Device, DeviceConfig, DeviceKind, Element, FloatDType, Tensor, TensorData, module,
};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.iter().any(|a| a == "--list-backends") {
        for name in backend_names() {
            println!("{name}");
        }
        return;
    }

    let cfg = match RunConfig::parse(&args) {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("bad arguments: {err}");
            std::process::exit(2);
        }
    };

    // Run from the per-run directory (its `cubecl.toml` directs the log + cache there), or from
    // the crate dir as a fallback. Set before the backend first touches cubecl.
    let workdir = match &cfg.run_dir {
        Some(dir) => {
            if let Err(err) = write_run_config(dir, cfg.disable_throughput_cache) {
                eprintln!("could not set up run dir {}: {err}", dir.display());
                std::process::exit(2);
            }
            let _ = std::fs::write(dir.join("meta.txt"), cfg.meta());
            dir.clone()
        }
        None => example_dir(),
    };
    let _ = std::env::set_current_dir(&workdir);

    let Some(mut device) = device_for(&cfg.backend) else {
        eprintln!(
            "backend '{}' is not compiled into the runner (available: {})",
            cfg.backend,
            backend_names().join(", ")
        );
        std::process::exit(2);
    };

    // Pin f32/i32 so the created tensors and autotune keys are stable before we cast.
    let _ = device.configure(
        DeviceConfig::default()
            .float_dtype(<f32 as Element>::dtype())
            .int_dtype(<i32 as Element>::dtype()),
    );

    for (index, shape) in cfg.shapes.iter().enumerate() {
        println!(
            "running {} {} / {}: {}  {}x{}x{}  in={} out={}",
            cfg.problem.name(),
            index + 1,
            cfg.shapes.len(),
            shape.name(),
            shape.m,
            shape.k,
            shape.n,
            cfg.input_name,
            cfg.output_name
        );

        let result = match cfg.problem {
            ProblemKind::Matmul => {
                run_matmul(&device, shape.m, shape.k, shape.n, cfg.input, cfg.output)
            }
            ProblemKind::Attention => {
                run_attention(&device, shape.m, shape.k, shape.n, cfg.input, cfg.output)
            }
        };

        match result {
            Ok(()) => {}
            Err(err) => {
                eprintln!("{} {} failed: {err}", cfg.problem.name(), index + 1);
                std::process::exit(1);
            }
        }
    }
    println!("done");
}

struct RunConfig {
    backend: String,
    problem: ProblemKind,
    shapes: Vec<MatmulShape>,
    input: FloatDType,
    input_name: String,
    output: FloatDType,
    output_name: String,
    run_dir: Option<PathBuf>,
    disable_throughput_cache: bool,
}

struct MatmulShape {
    m: usize,
    k: usize,
    n: usize,
}

impl MatmulShape {
    fn name(&self) -> String {
        format!("{}x{}x{}", self.m, self.k, self.n)
    }
}

impl RunConfig {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut backend = "wgpu".to_string();
        let mut problem = ProblemKind::Matmul;
        let mut shapes = Vec::new();
        let mut legacy_shape = [None; 3];
        let mut input_name = "f32".to_string();
        let mut output_name = "f32".to_string();
        let mut run_dir = None;
        let mut disable_throughput_cache = false;

        let mut it = args.iter();
        while let Some(flag) = it.next() {
            match flag.as_str() {
                "--backend" => backend = next(&mut it, flag)?,
                "--problem" => problem = ProblemKind::from_str(&next(&mut it, flag)?)?,
                "--shape" | "--matmul" => shapes.push(parse_matmul(&next(&mut it, flag)?, flag)?),
                "--m" => legacy_shape[0] = Some(parse_usize(&next(&mut it, flag)?, flag)?),
                "--k" => legacy_shape[1] = Some(parse_usize(&next(&mut it, flag)?, flag)?),
                "--n" => legacy_shape[2] = Some(parse_usize(&next(&mut it, flag)?, flag)?),
                "--input" => input_name = next(&mut it, flag)?,
                "--output" => output_name = next(&mut it, flag)?,
                "--run-dir" => run_dir = Some(PathBuf::from(next(&mut it, flag)?)),
                "--no-throughput-cache" => disable_throughput_cache = true,
                other => return Err(format!("unknown argument '{other}'")),
            }
        }

        let input = float_dtype(&input_name)?;
        let output = float_dtype(&output_name)?;
        if shapes.is_empty() {
            shapes.push(MatmulShape {
                m: legacy_shape[0].unwrap_or(512),
                k: legacy_shape[1].unwrap_or(512),
                n: legacy_shape[2].unwrap_or(512),
            });
        } else if legacy_shape.iter().any(Option::is_some) {
            return Err("use --shape or legacy --m/--k/--n, not both".into());
        }
        Ok(Self {
            backend,
            problem,
            shapes,
            input,
            input_name,
            output,
            output_name,
            run_dir,
            disable_throughput_cache,
        })
    }

    fn meta(&self) -> String {
        format!(
            "backend={}\nproblem={}\nshapes={}\ninput={} output={}\n",
            self.backend,
            self.problem.name(),
            self.shapes
                .iter()
                .map(MatmulShape::name)
                .collect::<Vec<_>>()
                .join(","),
            self.input_name,
            self.output_name
        )
    }
}

fn next(it: &mut std::slice::Iter<'_, String>, flag: &str) -> Result<String, String> {
    it.next()
        .cloned()
        .ok_or(format!("missing value for {flag}"))
}

fn parse_usize(value: &str, flag: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("{flag} expects an integer, got '{value}'"))
}

fn parse_matmul(value: &str, flag: &str) -> Result<MatmulShape, String> {
    let dimensions: Vec<_> = value.split('x').collect();
    if dimensions.len() != 3 {
        return Err(format!("{flag} expects MxKxN, got '{value}'"));
    }
    Ok(MatmulShape {
        m: parse_usize(dimensions[0], flag)?,
        k: parse_usize(dimensions[1], flag)?,
        n: parse_usize(dimensions[2], flag)?,
    })
}

fn float_dtype(name: &str) -> Result<FloatDType, String> {
    match name {
        "f32" => Ok(FloatDType::F32),
        "f16" => Ok(FloatDType::F16),
        "bf16" => Ok(FloatDType::BF16),
        "flex32" => Ok(FloatDType::Flex32),
        other => Err(format!("unknown dtype '{other}'")),
    }
}

/// Names of the backends compiled into this build. wgpu is always present; others are opt-in
/// cargo features (e.g. `--features cuda`), because wgpu can't target tensor cores directly.
fn backend_names() -> Vec<&'static str> {
    let mut names = Vec::new();
    #[cfg(feature = "cuda")]
    names.push("cuda");
    #[cfg(feature = "vulkan")]
    names.push("vulkan");
    #[cfg(feature = "metal")]
    names.push("metal");
    names.push("wgpu");
    #[cfg(feature = "cpu")]
    names.push("cpu");
    names
}

fn device_for(name: &str) -> Option<Device> {
    match name {
        #[cfg(feature = "cuda")]
        "cuda" => Some(Device::cuda(0)),
        #[cfg(feature = "vulkan")]
        "vulkan" => Some(Device::vulkan(DeviceKind::DefaultDevice)),
        #[cfg(feature = "metal")]
        "metal" => Some(Device::metal(DeviceKind::DefaultDevice)),
        "wgpu" => Some(Device::wgpu(DeviceKind::DefaultDevice)),
        #[cfg(feature = "cpu")]
        "cpu" => Some(Device::cpu()),
        _ => None,
    }
}

/// Run a single `m x k` by `k x n` matmul and block until it completes, which triggers
/// autotuning (and thus a log entry). `input` casts both operands (setting the matmul element
/// types); `output` casts the result afterward (a separate op, not the matmul's own output).
fn run_matmul(
    device: &Device,
    m: usize,
    k: usize,
    n: usize,
    input: FloatDType,
    output: FloatDType,
) -> Result<(), String> {
    let a = Tensor::<2>::from_data(TensorData::new(fill(&[m, k]), [m, k]), device).cast(input);
    let b = Tensor::<2>::from_data(TensorData::new(fill(&[k, n]), [k, n]), device).cast(input);

    let _c = a.matmul(b).cast(output);
    device.sync().map_err(|err| format!("{err:?}"))
}

fn fill(dims: &[usize]) -> Vec<f32> {
    let len = dims.iter().product();
    (0..len)
        .map(|i| ((i % 1000) as f32 / 1000.0) - 0.5)
        .collect()
}

/// Run a single attention workload using `module::attention` to exercise attention autotuning.
/// Shape mapping for `m x k x n` is:
/// - `m`: batch size
/// - `k`: sequence length
/// - `n`: head dimension
fn run_attention(
    device: &Device,
    m: usize,
    k: usize,
    n: usize,
    input: FloatDType,
    output: FloatDType,
) -> Result<(), String> {
    let query = Tensor::<4>::from_data(TensorData::new(fill(&[m, 1, k, n]), [m, 1, k, n]), device)
        .cast(input);
    let key = Tensor::<4>::from_data(TensorData::new(fill(&[m, 1, k, n]), [m, 1, k, n]), device)
        .cast(input);
    let value = Tensor::<4>::from_data(TensorData::new(fill(&[m, 1, k, n]), [m, 1, k, n]), device)
        .cast(input);

    let _out = module::attention(query, key, value, None, None, Default::default()).cast(output);
    device.sync().map_err(|err| format!("{err:?}"))
}
