// Solana blockchain interaction module
use solana_client::nonblocking::rpc_client::RpcClient;
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
use serde::{Deserialize, Serialize};
use solana_sdk::transaction::Transaction; // Added for deserializing swap transactions
use std::time::{Duration, Instant}; // Added for potential timing/timeouts
use base64; // Added for base64 decoding
use bincode; // Added for bincode deserialization
use base64::{engine::general_purpose::STANDARD, Engine as _};

// --- Jupiter API Structures --- //

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct QuoteRequest {
    input_mint: String,
    output_mint: String,
    amount: u64,
    slippage_bps: u16, // Basis points, e.g., 50 for 0.5%
    // Add other params like 'onlyDirectRoutes', 'platformFeeBps' if needed
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct QuoteResponse {
    input_mint: String,
    in_amount: String, // String representation of u64
    output_mint: String,
    out_amount: String, // String representation of u64
    other_amount_threshold: String, // String representation of u64
    swap_mode: String,
    slippage_bps: u16,
    platform_fee: Option<PlatformFee>,
    price_impact_pct: String, // String representation of f64
    route_plan: Vec<RoutePlanStep>,
    context_slot: Option<u64>,
    time_taken: Option<f64>,
    // This quote response is needed for the /swap request
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PlatformFee {
    amount: String, // String representation of u64
    fee_bps: u16,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct RoutePlanStep {
    swap_info: SwapInfo,
    percent: u8,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SwapInfo {
    amm_key: String,
    label: String,
    input_mint: String,
    output_mint: String,
    in_amount: String, // String representation of u64
    out_amount: String, // String representation of u64
    fee_amount: String, // String representation of u64
    fee_mint: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SwapRequest {
    user_public_key: String,
    quote_response: QuoteResponse, // The full quote response object
    wrap_and_unwrap_sol: Option<bool>,
    // Add 'feeAccount', 'prioritizationFeeLamports' etc. if needed
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SwapResponse {
    swap_transaction: String, // base64 encoded transaction
    last_valid_block_height: u64,
    // May also include 'prioritizationFeeLamports'
}

// Define Jupiter API base URL
const JUPITER_API_BASE_URL: &str = "https://quote-api.jup.ag/v6";

// --- SolanaManager --- //

#[derive(Clone)] // Keep Clone if needed
pub struct SolanaManager {
    rpc_client: Arc<RpcClient>,
    http_client: Arc<reqwest::Client>,
    config: SolanaConfig,
    keychain_service_name: String,
    solana_key_username: String,
}

// Manually implement Debug
impl std::fmt::Debug for SolanaManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SolanaManager")
         .field("rpc_client", &format_args!("<RpcClient>"))
         .field("http_client", &format_args!("<HttpClient>"))
         .field("config", &self.config)
         .field("keychain_service_name", &self.keychain_service_name)
         .field("solana_key_username", &self.solana_key_username)
         .finish()
    }
}

impl SolanaManager {
    pub fn new(solana_config: SolanaConfig, security_config: &SecurityConfig) -> Result<Self> {
        let rpc_client = Arc::new(RpcClient::new(solana_config.rpc_url.clone()));
        let http_client = Arc::new(reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| Error::InternalError(format!("Failed to create HTTP client: {}", e)))?);

        Ok(Self {
            rpc_client,
            http_client,
            config: solana_config,
            keychain_service_name: security_config.keychain_service_name.clone(),
            solana_key_username: security_config.solana_key_username.clone(),
        })
    }

    fn get_keypair(&self) -> Result<Keypair> {
        #[cfg(test)]
        {
            log::warn!("Using hardcoded test keypair for SolanaManager!");
            Ok(Keypair::new())
        }
        
        #[cfg(not(test))]
        {
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
                    let result = panic::catch_unwind(|| Keypair::from_base58_string(&key_material));
                    match result {
                        Ok(keypair) => Ok(keypair),
                        Err(_) => Err(Error::KeychainError("Failed to parse key material from keychain (invalid format)".to_string())),
                    }
                }
                Err(keyring::Error::NoEntry) => Err(Error::KeychainError(format!("No key found in keychain for service '{}', username '{}'", service_name, username))),
                Err(e) => Err(Error::KeychainError(format!("Keychain access error for service '{}', username '{}': {}", service_name, username, e))),
            }
        }
    }

    pub async fn get_balance(&self, token_mint_address_str: Option<&str>) -> Result<u64> {
        let keypair = self.get_keypair()?;
        let owner_pubkey = keypair.pubkey();
        log::info!("Checking balance for wallet: {}", owner_pubkey);

        match token_mint_address_str {
            None => {
                log::info!("Getting native SOL balance...");
                let balance = self.rpc_client.get_balance(&owner_pubkey).await
                    .map_err(|e| Error::SolanaRpcError(format!("Failed to get SOL balance: {}", e)))?;
                log::info!("Native SOL balance (lamports): {}", balance);
                Ok(balance)
            }
            Some(mint_str) => {
                log::info!("Getting balance for SPL token mint: {}", mint_str);
                let mint_pubkey = Pubkey::from_str(mint_str)
                    .map_err(|e| Error::InvalidInput(format!("Invalid token mint address '{}': {}", mint_str, e)))?;
                
                let ata_address = get_associated_token_address(&owner_pubkey, &mint_pubkey);
                log::info!("Calculated ATA address: {}", ata_address);

                match self.rpc_client.get_token_account_balance(&ata_address).await {
                    Ok(ui_token_amount) => {
                        let balance = ui_token_amount.amount.parse::<u64>()
                            .map_err(|_| Error::SolanaRpcError("Failed to parse token balance amount".to_string()))?;
                        log::info!(
                            "Token balance for mint {} (base units): {}", 
                            mint_str, balance
                        );
                        Ok(balance)
                    }
                    Err(e) => {
                        let e_str = e.to_string();
                        if e_str.contains("AccountNotFound") || e_str.contains("could not find account") {
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

    pub async fn get_quote(&self, input_mint: &str, output_mint: &str, amount: u64, slippage_bps: u16) -> Result<QuoteResponse> {
        log::info!(
            "Getting Jupiter quote: {} -> {} for amount {} with slippage {} bps",
            input_mint,
            output_mint,
            amount,
            slippage_bps
        );

        let quote_url = format!("{}/quote", JUPITER_API_BASE_URL);

        let params = [
            ("inputMint", input_mint),
            ("outputMint", output_mint),
            ("amount", &amount.to_string()),
            ("slippageBps", &slippage_bps.to_string()),
        ];

        let request_start = Instant::now();
        match self.http_client.get(&quote_url).query(&params).send().await {
            Ok(response) => {
                let response_time = request_start.elapsed();
                log::debug!("Jupiter /quote request took: {:?}", response_time);

                if response.status().is_success() {
                    match response.json::<QuoteResponse>().await {
                        Ok(quote_response) => {
                            log::info!(
                                "Received quote: {} {} -> {} {} (Impact: {}%)",
                                quote_response.in_amount,
                                quote_response.input_mint,
                                quote_response.out_amount,
                                quote_response.output_mint,
                                quote_response.price_impact_pct
                            );
                            Ok(quote_response)
                        }
                        Err(e) => {
                            log::error!("Failed to parse Jupiter quote response JSON: {}", e);
                            Err(Error::ApiError(format!(
                                "Failed to parse Jupiter quote JSON: {}",
                                e
                            )))
                        }
                    }
                } else {
                    let status = response.status();
                    let error_text = response.text().await.unwrap_or_else(|_| "Failed to read error body".to_string());
                    log::error!(
                        "Jupiter quote request failed with status {}: {}",
                        status,
                        error_text
                    );
                    Err(Error::ApiError(format!(
                        "Jupiter quote request failed (Status: {}): {}",
                        status,
                        error_text
                    )))
                }
            }
            Err(e) => {
                 log::error!("HTTP request failed for Jupiter quote: {}", e);
                 Err(Error::ApiError(format!("HTTP request failed for Jupiter quote: {}", e)))
            }
        }
    }

    pub async fn execute_swap(&self, input_mint: &str, output_mint: &str, amount: u64, slippage_bps: u16) -> Result<String> {
        log::info!(
            "Attempting Jupiter swap: {} -> {} for amount {} with slippage {} bps",
            input_mint,
            output_mint,
            amount,
            slippage_bps
        );

        let quote_response = self.get_quote(input_mint, output_mint, amount, slippage_bps).await
            .map_err(|e| {
                log::error!("Swap failed: Could not get quote: {}", e);
                e
            })?;
        log::debug!("Obtained quote for swap: {:?}", quote_response);

        let keypair = self.get_keypair()
             .map_err(|e| {
                 log::error!("Swap failed: Could not get keypair: {}", e);
                 e
             })?;
        let user_public_key = keypair.pubkey().to_string();
        log::debug!("Using wallet address for swap: {}", user_public_key);

        let swap_request = SwapRequest {
            user_public_key,
            quote_response: quote_response.clone(),
            wrap_and_unwrap_sol: Some(true),
        };

        let swap_url = format!("{}/swap", JUPITER_API_BASE_URL);
        log::debug!("Sending swap request to Jupiter...");
        let swap_request_start = Instant::now();
        
        let swap_api_response = match self.http_client.post(&swap_url).json(&swap_request).send().await {
             Ok(response) => response,
             Err(e) => {
                 log::error!("HTTP request failed for Jupiter swap: {}", e);
                 return Err(Error::ApiError(format!("Jupiter /swap HTTP request failed: {}", e)));
             }
        };
        let swap_response_time = swap_request_start.elapsed();
        log::debug!("Jupiter /swap request took: {:?}", swap_response_time);

        let swap_response = if swap_api_response.status().is_success() {
            match swap_api_response.json::<SwapResponse>().await {
                Ok(resp) => resp,
                Err(e) => {
                    log::error!("Failed to parse Jupiter swap response JSON: {}", e);
                    return Err(Error::ApiError(format!("Failed to parse Jupiter swap JSON: {}", e)));
                }
            }
        } else {
            let status = swap_api_response.status();
            let error_text = swap_api_response.text().await.unwrap_or_else(|_| "Failed to read error body".to_string());
            log::error!(
                "Jupiter swap request failed with status {}: {}",
                status,
                error_text
            );
            return Err(Error::ApiError(format!(
                "Jupiter swap request failed (Status: {}): {}",
                status,
                error_text
            )));
        };
        log::debug!("Received swap transaction details from Jupiter.");

        let transaction_bytes = match STANDARD.decode(&swap_response.swap_transaction) {
            Ok(bytes) => bytes,
            Err(e) => {
                log::error!("Failed to decode base64 swap transaction: {}", e);
                return Err(Error::InternalError(format!("Base64 decode failed: {}", e)));
            }
        };

        let mut transaction: Transaction = match bincode::deserialize(&transaction_bytes) {
            Ok(tx) => tx,
            Err(e) => {
                log::error!("Failed to deserialize swap transaction: {}", e);
                return Err(Error::InternalError(format!("Transaction deserialize failed: {}", e)));
            }
        };
        log::debug!("Successfully deserialized swap transaction.");

        let recent_blockhash = match self.rpc_client.get_latest_blockhash().await {
            Ok(blockhash) => blockhash,
            Err(e) => {
                log::error!("Failed to get recent blockhash: {}", e);
                return Err(Error::SolanaRpcError(format!("Failed to get blockhash: {}", e)));
            }
        };
        transaction.message.recent_blockhash = recent_blockhash;

        match transaction.try_sign(&[&keypair], recent_blockhash) {
            Ok(_) => log::debug!("Successfully signed swap transaction."),
            Err(e) => {
                log::error!("Failed to sign swap transaction: {}", e);
                return Err(Error::SigningError(format!("Failed to sign swap tx: {}", e)));
            }
        }

        log::info!("Sending signed transaction to Solana network...");
        let send_start = Instant::now();
        match self.rpc_client.send_and_confirm_transaction_with_spinner(&transaction).await {
            Ok(signature) => {
                let send_time = send_start.elapsed();
                log::info!(
                    "Swap successful! Signature: {}. Confirmation took: {:?}",
                    signature,
                    send_time
                );
                Ok(signature.to_string())
            }
            Err(e) => {
                log::error!("Failed to send/confirm swap transaction: {}", e);
                Err(Error::SolanaRpcError(format!(
                    "Failed to send/confirm swap transaction: {}",
                    e
                )))
            }
        }
    }
} 