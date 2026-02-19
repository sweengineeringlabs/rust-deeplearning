mod cmd;

use anyhow::Result;
use clap::Parser;

/// SweAI — unified CLI for RustML.
#[derive(Parser)]
#[command(name = "sweai", version, about)]
struct Cli {
    #[command(subcommand)]
    command: cmd::Command,
}

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();
    cmd::run(cli.command)
}
