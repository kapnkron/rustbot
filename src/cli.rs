use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Path to the configuration file
    #[arg(short, long)] // Removed default_value here, handled in main.rs
    pub config: Option<PathBuf>,

    /// Enable debug logging
    #[arg(short, long)]
    pub debug: bool,
} 