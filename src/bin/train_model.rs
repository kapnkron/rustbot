use clap::Parser;
use std::path::PathBuf;
use trading_bot::ml::run_training_session; // Import the new function
use anyhow::Result;
use log::error;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Path to the historical market data CSV file.
    #[clap(short = 'd', long, value_parser, default_value = "data/raw/poloniex_solusdc_1h.csv")]
    csv_path: PathBuf,

    /// Path where the trained model should be saved.
    #[clap(short, long, value_parser, default_value = "models/trained_model.ot")]
    model_path: PathBuf,

    /// Path to the main configuration TOML file.
    #[clap(short, long, value_parser, default_value = "config/config.toml")]
    config_path: PathBuf,

    /// Ratio for splitting data into training set (e.g., 0.8 for 80%).
    #[clap(short, long, value_parser, default_value_t = 0.8)]
    split_ratio: f64,
    
    // TODO: Add arguments for other training parameters if needed (epochs, lr, etc.)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logger (use RUST_LOG env var, e.g., RUST_LOG=info)
    env_logger::init();

    let args = Args::parse();

    // Call the library function to handle the actual training
    if let Err(e) = run_training_session(
        args.config_path,
        args.csv_path,
        args.model_path,
        args.split_ratio,
    ).await {
        error!("Model training failed: {:?}", e);
        // Use std::process::exit to return a non-zero exit code on failure
        std::process::exit(1);
    }

    println!("Training process initiated successfully. See logs for details.");
    Ok(())
} 