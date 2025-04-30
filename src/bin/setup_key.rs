#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! # trading_bot = { path = ".." } # Remove direct lib dependency for config
//! # keyring = "3.0" # Removed
//! # rpassword = "7.3" # Removed
//! solana_sdk = "2.0.3"
//! anyhow = "1.0"
//! clap = { version = "4.0", features = ["derive"] } # Added clap
//! reqwest = { version = "0.12", features = ["json"] } # Added reqwest with json feature
//! serde_json = "1.0" # Added serde_json
//! tokio = { version = "1", features = ["full"] } # Added tokio
//! # trading_bot = { path = ".." } # Removed library path dependency
//! ```
// use trading_bot::config::SecurityConfig; // Remove this import
// use trading_bot::error::Error as BotError; // Keep error type if needed for mapping - Removed as unused
// use keyring::Entry; // Removed as unused
// use rpassword::read_password; // Removed as unused
// use std::io::{stdout, Write}; // Removed as unused
use solana_sdk::signer::keypair::Keypair;
use anyhow::Result;
use clap::Parser;
// use trading_bot::decode_key_material; // Removed import, function defined below
use reqwest::Client;
use serde_json::json;
use solana_sdk::signature::read_keypair_file;
use solana_sdk::signer::Signer;
use std::fs;
use std::path::Path;

/// Reads the content of a file and returns it as a string.
fn decode_key_material<P: AsRef<Path>>(path: P) -> Result<String> {
    fs::read_to_string(path.as_ref())
        .map_err(|e| anyhow::anyhow!("Failed to read key material file {:?}: {}", path.as_ref(), e))
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The path to the keypair file
    #[clap(short, long, value_parser)]
    keypair_path: String,

    /// The path to the key material file
    #[clap(short, long, value_parser)]
    key_material_path: String,

    /// The API endpoint URL
    #[clap(short, long, value_parser)]
    url: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Read the keypair from the file
    let keypair = read_keypair_file(&args.keypair_path)
        .map_err(|e| anyhow::anyhow!("Failed to read keypair file: {}", e))?;

    // Read and decode the key material
    let key_material = decode_key_material(&args.key_material_path)?;

    // This now directly returns a Keypair or panics.
    let service_keypair = Keypair::from_base58_string(&key_material);

    // Prepare the request payload
    let payload = json!({
        "publicKey": keypair.pubkey().to_string(),
        "servicePublicKey": service_keypair.pubkey().to_string(),
    });

    // Send the request
    let client = Client::new();
    let res = client
        .post(&args.url)
        .json(&payload)
        .send()
        .await?;

    if res.status().is_success() {
        println!("Request successful: {}", res.status());
        // Optionally print the response body
        // let body = res.text().await?;
        // println!("Response body: {}", body);
    } else {
        println!("Request failed: {}", res.status());
        let body = res.text().await?;
        println!("Response body: {}", body);
    }

    Ok(())
} 