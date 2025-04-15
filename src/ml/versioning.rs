use crate::utils::error::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::fs;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    pub version: String,
    pub timestamp: DateTime<Utc>,
    pub metrics: HashMap<String, f64>,
    pub model_path: PathBuf,
    pub training_data_size: usize,
    pub features_used: Vec<String>,
    pub hyperparameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersionManager {
    versions: Vec<ModelVersion>,
    current_version: String,
    base_path: PathBuf,
}

impl ModelVersionManager {
    pub fn new(base_path: &Path) -> Result<Self> {
        let versions_path = base_path.join("versions.json");
        let versions = if versions_path.exists() {
            let content = fs::read_to_string(&versions_path)?;
            serde_json::from_str(&content)?
        } else {
            Vec::new()
        };

        Ok(Self {
            versions,
            current_version: "1.0.0".to_string(),
            base_path: base_path.to_path_buf(),
        })
    }

    pub fn create_version(
        &mut self,
        metrics: HashMap<String, f64>,
        model_path: &Path,
        training_data_size: usize,
        features_used: Vec<String>,
        hyperparameters: HashMap<String, String>,
    ) -> Result<ModelVersion> {
        let version = ModelVersion {
            version: self.current_version.clone(),
            timestamp: Utc::now(),
            metrics,
            model_path: model_path.to_path_buf(),
            training_data_size,
            features_used,
            hyperparameters,
        };

        self.versions.push(version.clone());
        self.save_versions()?;
        Ok(version)
    }

    pub fn get_version(&self, version: &str) -> Option<&ModelVersion> {
        self.versions.iter().find(|v| v.version == version)
    }

    pub fn get_latest_version(&self) -> Option<&ModelVersion> {
        self.versions.last()
    }

    pub fn compare_versions(&self, version1: &str, version2: &str) -> Result<HashMap<String, f64>> {
        let v1 = self.get_version(version1)
            .ok_or_else(|| anyhow::anyhow!("Version {} not found", version1))?;
        let v2 = self.get_version(version2)
            .ok_or_else(|| anyhow::anyhow!("Version {} not found", version2))?;

        let mut comparison = HashMap::new();
        for (metric, value1) in &v1.metrics {
            if let Some(value2) = v2.metrics.get(metric) {
                comparison.insert(
                    format!("{}_improvement", metric),
                    value2 - value1,
                );
            }
        }

        Ok(comparison)
    }

    fn save_versions(&self) -> Result<()> {
        let versions_path = self.base_path.join("versions.json");
        let content = serde_json::to_string_pretty(&self.versions)?;
        fs::write(versions_path, content)?;
        Ok(())
    }

    pub fn increment_version(&mut self) {
        let parts: Vec<&str> = self.current_version.split('.').collect();
        if parts.len() == 3 {
            if let (Ok(major), Ok(minor), Ok(patch)) = (
                parts[0].parse::<u32>(),
                parts[1].parse::<u32>(),
                parts[2].parse::<u32>(),
            ) {
                self.current_version = format!("{}.{}.{}", major, minor, patch + 1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use tempfile::tempdir;

    #[test]
    fn test_model_versioning() -> Result<()> {
        let temp_dir = tempdir()?;
        let mut manager = ModelVersionManager::new(temp_dir.path())?;

        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.85);
        metrics.insert("precision".to_string(), 0.82);

        let mut hyperparameters = HashMap::new();
        hyperparameters.insert("learning_rate".to_string(), "0.001".to_string());
        hyperparameters.insert("batch_size".to_string(), "32".to_string());

        let version = manager.create_version(
            metrics,
            &temp_dir.path().join("model.pt"),
            1000,
            vec!["price".to_string(), "volume".to_string()],
            hyperparameters,
        )?;

        assert_eq!(version.version, "1.0.0");
        assert_eq!(version.training_data_size, 1000);
        assert!(version.metrics.contains_key("accuracy"));

        manager.increment_version();
        assert_eq!(manager.current_version, "1.0.1");

        Ok(())
    }
} 