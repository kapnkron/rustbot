use log::{Level, LevelFilter, Metadata, Record};
use std::sync::Mutex;
use std::fs::OpenOptions;
use std::io::Write;
use chrono::Local;

pub struct Logger {
    file: Mutex<std::fs::File>,
}

impl Logger {
    pub fn new(log_file: &str) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_file)?;
        
        Ok(Self {
            file: Mutex::new(file),
        })
    }
}

impl log::Log for Logger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Info
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            let mut file = self.file.lock().unwrap();
            let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S");
            writeln!(
                file,
                "{} [{}] {}",
                timestamp,
                record.level(),
                record.args()
            ).unwrap();
        }
    }

    fn flush(&self) {
        let mut file = self.file.lock().unwrap();
        file.flush().unwrap();
    }
}

pub fn init(log_file: &str) -> Result<(), log::SetLoggerError> {
    let logger = Logger::new(log_file).expect("Failed to create logger");
    log::set_boxed_logger(Box::new(logger))?;
    log::set_max_level(LevelFilter::Info);
    Ok(())
} 