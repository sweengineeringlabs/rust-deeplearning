mod args;
mod commands;
mod gguf_tokenizer;
mod model_source;

use clap::Parser;

fn main() -> anyhow::Result<()> {
    let cli = args::Cli::parse();

    match &cli.command {
        args::Commands::Run(run_args) => commands::run::execute(run_args),
        args::Commands::Info(info_args) => commands::info::execute(info_args),
        args::Commands::Download(dl_args) => commands::download::execute(dl_args),
    }
}
