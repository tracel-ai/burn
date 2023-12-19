// ../examples/torch-import/pytorch/mnist.pt

#[cfg(feature = "pytorch")]
use anyhow::Result;
#[cfg(feature = "pytorch")]
use burn_import::pytorch::RecordType;
#[cfg(feature = "pytorch")]
use clap::Parser;

#[cfg(feature = "pytorch")]
#[derive(Parser)]
#[clap(version, author, about)]
struct Opts {
    /// Sets the input file to use
    #[clap(short, long)]
    filename: String,

    /// Sets the output directory
    #[clap(short, long)]
    outdir: String,

    /// Record type
    #[clap(short, long, default_value = "named-mpk")]
    record_type: RecordType,
}

#[cfg(feature = "pytorch")]
fn main() -> Result<()> {
    use burn_import::pytorch::Converter;

    let opts: Opts = Opts::parse();

    Converter::new()
        .input(&opts.filename)
        .out_dir(&opts.outdir)
        .development(true)
        .run_from_cli();

    Ok(())
}

#[cfg(not(feature = "pytorch"))]
fn main() {
    println!("Compiled without Pytorch feature.");
}
