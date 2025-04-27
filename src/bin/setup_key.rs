#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! // trading_bot = { path = ".." } # Remove direct lib dependency for config
//! keyring = "3.0"
//! rpassword = "7.3"
//! solana-sdk = "2.0.3"
//! anyhow = "1.0"
//! ```
// use trading_bot::config::SecurityConfig; // Remove this import
use trading_bot::error::Error as BotError; // Keep error type if needed for mapping
use keyring::Entry;
use rpassword::read_password;
use std::io::{stdout, Write};
use solana_sdk::signer::{keypair::Keypair, Signer};

fn main() -> anyhow::Result<()> {
    println!("Solana Keypair Setup Utility for Trading Bot");
    println!("---------------------------------------------");

    // Define service/username directly (matching SecurityConfig defaults)
    let service_name = "test-bot";
    let username = "test-sol-key"; 

    println!("This will store your Solana private key (base58 format) in the OS keychain.");
    println!("Service:  {}", service_name);
    println!("Username: {}", username);

    print!("Paste your Solana private key (base58 encoded string): ");
    stdout().flush()?;
    let key_material = read_password()?;

    if key_material.trim().is_empty() {
        anyhow::bail!("Private key cannot be empty.");
    }

    // Explicitly match the Result from Keypair::from_base58_string
    match Keypair::from_base58_string(&key_material) {
        Ok(_) => { /* Keypair parsed okay, continue */ },
        Err(_) => {
             anyhow::bail!("Invalid private key format. Please ensure it's a base58 encoded string.");
        }
    }

    println!("\nStoring key in keychain...");

    let entry = Entry::new(service_name, username)
        .map_err(|e| BotError::KeychainError(format!("Failed to create keychain entry: {}", e)))?; // Keep BotError mapping for now

    match entry.set_password(&key_material) {
        Ok(_) => {
            println!("Successfully stored Solana key for service '{}', username '{}' in keychain.", service_name, username);
            Ok(())
        }
        Err(e) => {
            eprintln!("Failed to store key in keychain: {}", e);
            Err(anyhow::anyhow!("Keychain store error: {}", e))
        }
    }
} 