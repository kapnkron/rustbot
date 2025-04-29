#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! keyring = "3.0"
//! solana_sdk = "2.0.3" // Or your project's version
//! clap = { version = "4.0", features = ["derive"] }
//! anyhow = "1.0"
//! serde_json = "1.0" // For parsing the keypair file
//! bs58 = "0.5"      // For base58 encoding
//! ```
use anyhow::{Context, Result};
use clap::Parser;
use keyring::Entry;
use solana_sdk::signer::keypair::Keypair;
use std::{fs, path::PathBuf};

#[derive(Parser, Debug)]
#[clap(author, version, about = "Reads a Solana keypair file and stores the private key in the OS keychain.", long_about = None)]
struct Args {
    /// Path to the Solana keypair file (JSON format).
    #[clap(short = 'k', long, value_parser)]
    keypair_path: PathBuf,

    /// The service name for the keychain entry (e.g., 'my-trading-bot').
    #[clap(short, long, value_parser)]
    service_name: String,

    /// The username/account name for the keychain entry (e.g., 'solana-key').
    #[clap(short, long, value_parser)]
    username: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Read the keypair file content
    let keypair_json = fs::read_to_string(&args.keypair_path)
        .with_context(|| format!("Failed to read keypair file: {:?}", args.keypair_path))?;

    // Parse the JSON file to get the byte array for the private key
    let keypair_bytes: Vec<u8> = serde_json::from_str(&keypair_json)
        .with_context(|| format!("Failed to parse JSON keypair file: {:?}", args.keypair_path))?;
        
    // Reconstruct the keypair object to easily access the secret key bytes
    // Note: Keypair::from_bytes expects 64 bytes (secret + public), but the JSON file usually contains only the 64 bytes.
    let keypair = Keypair::from_bytes(&keypair_bytes)
         .map_err(|e| anyhow::anyhow!("Failed to create keypair from bytes: {}", e))?;

    // Get the secret key bytes (first 32 bytes)
    let secret_key_bytes = &keypair.secret().to_bytes()[..32];

    // Encode the secret key bytes as a base58 string (common format for private keys)
    let private_key_base58 = bs58::encode(secret_key_bytes).into_string();

    // Create the keychain entry
    let entry = Entry::new(&args.service_name, &args.username)
        .with_context(|| format!("Failed to create keychain entry for service '{}', user '{}'", args.service_name, args.username))?;

    // Store the private key string in the keychain
    entry.set_password(&private_key_base58)
        .with_context(|| format!("Failed to store secret in keychain for service '{}', user '{}'", args.service_name, args.username))?;

    println!(
        "Successfully stored Solana keypair from {:?} in keychain.",
        args.keypair_path
    );
    println!("Service: {}", args.service_name);
    println!("Username: {}", args.username);

    Ok(())
} 