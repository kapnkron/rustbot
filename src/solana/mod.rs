// Solana blockchain interaction module
use solana_client::rpc_client::RpcClient;
use solana_sdk::signer::{keypair::Keypair, Signer};
use crate::config::{SolanaConfig, SecurityConfig}; // Import SecurityConfig
use crate::error::{Result, Error}; // Assuming crate::error::Error exists
use keyring::Entry; // Ensure Entry is imported
use std::sync::Arc;
use log; // Import log crate
use solana_sdk::pubkey::Pubkey;
use spl_associated_token_account::get_associated_token_address;
use std::str::FromStr; // For Pubkey::from_str
use std::panic; // Ensure panic is imported

#[derive(Clone)] // Keep Clone if needed
pub struct SolanaManager {
    rpc_client: Arc<RpcClient>, // Use Arc for potential sharing across threads
    config: SolanaConfig, // Keep SolanaConfig for rpc_url
    keychain_service_name: String, // Added field
    solana_key_username: String,   // Added field
    // keypair: Option<Box<dyn Signer>>, // Load keypair later when needed
}

// Manually implement Debug
impl std::fmt::Debug for SolanaManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SolanaManager")
         .field("rpc_client", &format_args!("<RpcClient>")) // Placeholder for non-Debug field
         .field("config", &self.config)
         .field("keychain_service_name", &self.keychain_service_name) // Added field to Debug output
         .field("solana_key_username", &self.solana_key_username)     // Added field to Debug output
         .finish()
    }
}

impl SolanaManager {
    pub fn new(solana_config: SolanaConfig, security_config: &SecurityConfig) -> Result<Self> {
        // Initialize RPC client
        let rpc_client = Arc::new(RpcClient::new(solana_config.rpc_url.clone()));
        
        // Check connection (optional, but good practice)
        // let version = rpc_client.get_version()?;
        // log::info!("Connected to Solana RPC: {} version: {}", solana_config.rpc_url, version);

        Ok(Self {
            rpc_client,
            config: solana_config, // Store the SolanaConfig
            keychain_service_name: security_config.keychain_service_name.clone(), // Store from SecurityConfig
            solana_key_username: security_config.solana_key_username.clone(),     // Store from SecurityConfig
            // keypair: None,
        })
    }

    // Helper to get keypair
    fn get_keypair(&self) -> Result<Keypair> {
        #[cfg(test)]
        {
            // Return a fixed, known keypair for testing purposes
            // IMPORTANT: This keypair should ONLY be used for tests and hold no real value.
            log::warn!("Using hardcoded test keypair for SolanaManager!");
            // Generate a new throwaway keypair each time for tests
            Ok(Keypair::new())
            // Or use a fixed one if necessary for specific address checks:
            // Ok(Keypair::from_base58_string("YOUR_TEST_PRIVATE_KEY_BASE58_HERE").expect("Failed to parse hardcoded test keypair"))
        }
        
        #[cfg(not(test))]
        {
            // Original keychain logic for non-test builds
            let service_name = &self.keychain_service_name;
            let username = &self.solana_key_username;
            
            log::debug!(
                "Attempting to retrieve key from keychain for service: '{}', username: '{}'",
                service_name,
                username
            );

            let entry = Entry::new(service_name, username)
                .map_err(|e| Error::KeychainError(format!("Failed to create keychain entry: {}", e)))?;
                
            match entry.get_password() {
                Ok(key_material) => {
                    let result = panic::catch_unwind(|| {
                        Keypair::from_base58_string(&key_material)
                    });

                    match result {
                        Ok(keypair) => Ok(keypair),
                        Err(_) => Err(Error::KeychainError(
                            "Failed to parse key material from keychain (invalid format)".to_string()
                        )),
                    }
                }
                Err(keyring::Error::NoEntry) => Err(Error::KeychainError(format!(
                    "No key found in keychain for service '{}', username '{}'",
                    service_name,
                    username
                ))),
                Err(e) => Err(Error::KeychainError(format!(
                    "Keychain access error for service '{}', username '{}': {}",
                    service_name,
                    username,
                    e
                ))),
            }
        }
    }

    // Updated get_balance function
    pub async fn get_balance(&self, token_mint_address_str: Option<&str>) -> Result<u64> {
        let keypair = self.get_keypair()?;
        let owner_pubkey = keypair.pubkey();
        log::info!("Checking balance for wallet: {}", owner_pubkey);

        match token_mint_address_str {
            // Get native SOL balance
            None => {
                log::info!("Getting native SOL balance...");
                let balance = self.rpc_client.get_balance(&owner_pubkey)
                    .map_err(|e| Error::SolanaRpcError(format!("Failed to get SOL balance: {}", e)))?;
                log::info!("Native SOL balance (lamports): {}", balance);
                Ok(balance)
            }
            // Get SPL token balance
            Some(mint_str) => {
                log::info!("Getting balance for SPL token mint: {}", mint_str);
                let mint_pubkey = Pubkey::from_str(mint_str)
                    .map_err(|e| Error::InvalidInput(format!("Invalid token mint address '{}': {}", mint_str, e)))?;
                
                // Calculate the associated token account (ATA) address
                let ata_address = get_associated_token_address(&owner_pubkey, &mint_pubkey);
                log::info!("Calculated ATA address: {}", ata_address);

                match self.rpc_client.get_token_account_balance(&ata_address) {
                    Ok(ui_token_amount) => {
                        // Amount is returned as a string, parse it to u64
                        let balance = ui_token_amount.amount.parse::<u64>()
                            .map_err(|_| Error::SolanaRpcError("Failed to parse token balance amount".to_string()))?;
                        log::info!(
                            "Token balance for mint {} (base units): {}", 
                            mint_str, balance
                        );
                        Ok(balance)
                    }
                    Err(e) => {
                        // Handle case where ATA might not exist yet (balance is 0)
                        if e.to_string().contains("AccountNotFound") || e.to_string().contains("could not find account") {
                             log::info!("Token account {} not found, assuming balance is 0", ata_address);
                             Ok(0) 
                        } else {
                            Err(Error::SolanaRpcError(format!(
                                "Failed to get token balance for ATA {}: {}", 
                                ata_address, e
                            )))
                        }
                    }
                }
            }
        }
    }

    // Placeholder functions for actions
    pub async fn get_quote(&self, input_mint: &str, output_mint: &str, amount: u64) -> Result<()> {
        log::info!("Placeholder: Getting quote {} -> {} for amount {}", input_mint, output_mint, amount);
        // TODO: Implement DEX interaction (e.g., Jupiter API call)
        Ok(())
    }

    pub async fn execute_swap(&self, input_mint: &str, output_mint: &str, amount: u64) -> Result<()> {
        log::info!("Placeholder: Executing swap {} -> {} for amount {}", input_mint, output_mint, amount);
        // TODO: 
        // 1. Get quote from DEX
        // 2. Load keypair from keychain
        // 3. Construct transaction
        // 4. Sign transaction
        // 5. Send and confirm transaction
        Ok(())
    }
} 