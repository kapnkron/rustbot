use crate::error::Result;
use crate::api::types::{MarketData, Quote, USDData};
use csv::ReaderBuilder;
use std::path::Path;
use std::fs::File;
use chrono::{DateTime, Utc, NaiveDateTime};
use log::{info, warn, error};

/// Loads historical market data from a CryptoDataDownload CSV file.
/// 
/// Assumes the specific Poloniex SOL/USDC hourly format:
/// - Skips the first row (URL).
/// - Headers on the second row: unix, date, symbol, open, high, low, close, Volume SOL, Volume USDC, ...
/// - Timestamp in the 'unix' column is in **milliseconds**.
/// 
/// Args:
/// * `file_path`: Path to the CSV file.
/// 
/// Returns:
/// * A `Result` containing a `Vec<MarketData>` or an error.
pub fn load_market_data_from_csv(file_path: &Path) -> Result<Vec<MarketData>> {
    info!("Attempting to load market data from CSV: {:?}", file_path);

    let file = File::open(file_path).map_err(|e| {
        error!("Failed to open CSV file {:?}: {}", file_path, e);
        crate::error::Error::IoError(e)
    })?;

    // Use ReaderBuilder to configure parsing options
    let mut rdr = ReaderBuilder::new()
        .has_headers(true) // Headers are present (on row 2, but we handle offset)
        .flexible(true)    // Allow variable number of fields per record temporarily
        .from_reader(file);

    let mut records = Vec::new();
    let mut skipped_header = false;

    // Get the headers separately first to handle the offset
    let headers = rdr.headers()?.clone();

    for (line_num, result) in rdr.records().enumerate() {
        // Skip the first actual data row (which corresponds to header row 2 in the file)
        if !skipped_header {
             skipped_header = true;
             // We already got headers using rdr.headers(), so just skip this record iteration
             continue; 
         }
        
        let record = match result {
            Ok(rec) => rec,
            Err(e) => {
                warn!("Skipping malformed CSV record on line {}: {}", line_num + 3, e); // +3 because 1-based line, header row, URL row
                continue;
            }
        };

        // Basic check for minimum expected fields based on headers we need
        if record.len() < 9 { // Need at least up to Volume USDC
             warn!("Skipping record on line {} due to insufficient fields (got {}, need at least 9)", line_num + 3, record.len());
             continue;
        }

        // Parse required fields, handling potential errors
        let timestamp_ms: i64 = match record.get(0).unwrap_or("").parse() {
            Ok(ts) => ts,
            Err(_) => {
                warn!("Skipping record on line {}: Invalid timestamp format '{}'", line_num + 3, record.get(0).unwrap_or("[missing]"));
                continue;
            }
        };
        // Convert milliseconds to DateTime<Utc>
        let naive_dt = NaiveDateTime::from_timestamp_opt(timestamp_ms / 1000, (timestamp_ms % 1000) as u32 * 1_000_000);
        let timestamp = match naive_dt {
            Some(ndt) => DateTime::<Utc>::from_naive_utc_and_offset(ndt, Utc),
            None => {
                 warn!("Skipping record on line {}: Invalid timestamp value '{}'", line_num + 3, timestamp_ms);
                 continue;
            }
        };

        let symbol = record.get(2).unwrap_or("").trim().to_string();
        // Use 'close' price for the main price field
        let price: f64 = match record.get(6).unwrap_or("").parse() {
            Ok(p) => p,
            Err(_) => {
                warn!("Skipping record on line {}: Invalid close price format '{}'", line_num + 3, record.get(6).unwrap_or("[missing]"));
                continue;
            }
        };
        // Use 'Volume USDC' for volume fields
        let volume: f64 = match record.get(8).unwrap_or("").parse() {
            Ok(v) => v,
            Err(_) => {
                 warn!("Skipping record on line {}: Invalid Volume USDC format '{}'", line_num + 3, record.get(8).unwrap_or("[missing]"));
                 continue;
            }
        };

        // Defaults for missing fields in MarketData struct
        let market_cap = 0.0;
        let price_change_24h = 0.0;
        let volume_change_24h = 0.0;

        records.push(MarketData {
            symbol,
            price,
            volume, // Map Volume USDC
            market_cap,
            price_change_24h,
            volume_change_24h,
            timestamp,
            volume_24h: volume, // Redundant, but map Volume USDC here too
            change_24h: price_change_24h, // Redundant, use default
            quote: Quote {
                usd: USDData {
                    price,
                    volume_24h: volume,
                    market_cap,
                    percent_change_24h: price_change_24h,
                    volume_change_24h,
                }
            }
        });
    }

    info!("Successfully loaded {} records from CSV: {:?}", records.len(), file_path);
    Ok(records)
} 