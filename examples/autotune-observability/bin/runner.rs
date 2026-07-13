//! Headless one-shot matmul runner.
//!
//! The UI shells out to this binary once per "Run" so each matmul happens in a fresh process:
//! the in-memory autotune cache starts empty every time, so the same config re-tunes (and
//! re-logs) on every run. It also keeps burn out of the UI binary, so the UI rebuilds fast.
//!
//! Usage:
//!   runner --list-backends
//!   runner --backend <name> --m <M> --k <K> --n <N> --input <dtype> --output <dtype>

use autotune_observability::{example_dir, start_fresh_session};
use burn::tensor::{Device, DeviceConfig, DeviceKind, Element, FloatDType, Tensor, TensorData};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.iter().any(|a| a == "--list-backends") {
        for name in backend_names() {
            println!("{name}");
        }
        return;
    }

    // The `cubecl.toml` next to this crate configures the autotune logger; run from here so it's
    // discovered and the log lands at the path the UI reads.
    let _ = std::env::set_current_dir(example_dir());

    let cfg = match RunConfig::parse(&args) {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("bad arguments: {err}");
            std::process::exit(2);
        }
    };

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

    start_fresh_session();

    println!(
        "running {}  {}x{} * {}x{}  in={} out={}",
        cfg.backend, cfg.m, cfg.k, cfg.k, cfg.n, cfg.input_name, cfg.output_name
    );

    match run_matmul(&device, cfg.m, cfg.k, cfg.n, cfg.input, cfg.output) {
        Ok(()) => println!("done"),
        Err(err) => {
            eprintln!("matmul failed: {err}");
            std::process::exit(1);
        }
    }
}

struct RunConfig {
    backend: String,
    m: usize,
    k: usize,
    n: usize,
    input: FloatDType,
    input_name: String,
    output: FloatDType,
    output_name: String,
}

impl RunConfig {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut backend = "wgpu".to_string();
        let (mut m, mut k, mut n) = (512usize, 512usize, 512usize);
        let mut input_name = "f32".to_string();
        let mut output_name = "f32".to_string();

        let mut it = args.iter();
        while let Some(flag) = it.next() {
            match flag.as_str() {
                "--backend" => backend = next(&mut it, flag)?,
                "--m" => m = parse_usize(&next(&mut it, flag)?, flag)?,
                "--k" => k = parse_usize(&next(&mut it, flag)?, flag)?,
                "--n" => n = parse_usize(&next(&mut it, flag)?, flag)?,
                "--input" => input_name = next(&mut it, flag)?,
                "--output" => output_name = next(&mut it, flag)?,
                other => return Err(format!("unknown argument '{other}'")),
            }
        }

        let input = float_dtype(&input_name)?;
        let output = float_dtype(&output_name)?;
        Ok(Self {
            backend,
            m,
            k,
            n,
            input,
            input_name,
            output,
            output_name,
        })
    }
}

fn next(it: &mut std::slice::Iter<'_, String>, flag: &str) -> Result<String, String> {
    it.next().cloned().ok_or(format!("missing value for {flag}"))
}

fn parse_usize(value: &str, flag: &str) -> Result<usize, String> {
    value.parse().map_err(|_| format!("{flag} expects an integer, got '{value}'"))
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
    let a = Tensor::<2>::from_data(TensorData::new(fill(m, k), [m, k]), device).cast(input);
    let b = Tensor::<2>::from_data(TensorData::new(fill(k, n), [k, n]), device).cast(input);

    let _c = a.matmul(b).cast(output);
    device.sync().map_err(|err| format!("{err:?}"))
}

fn fill(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| ((i % 1000) as f32 / 1000.0) - 0.5)
        .collect()
}
