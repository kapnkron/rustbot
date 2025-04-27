use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::collections::HashMap;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    pub version: String,
    pub timestamp: DateTime<Utc>,
    pub metrics: HashMap<String, f64>,
    pub input_size: i64,
    pub hidden_size: i64,
    pub output_size: i64,
    pub learning_rate: f64,
    pub window_size: usize,
}

#[derive(Debug)]
pub struct ModelVersionManager {
    versions: HashMap<String, ModelVersion>,
    base_path: String,
}

impl ModelVersionManager {
    pub fn new(base_path: &str) -> Result<Self> {
        let path = Path::new(base_path);
        if !path.exists() {
            std::fs::create_dir_all(path)?;
        }

        Ok(Self {
            versions: HashMap::new(),
            base_path: base_path.to_string(),
        })
    }

    pub fn add_version(&mut self, version: ModelVersion) -> Result<()> {
        self.versions.insert(version.version.clone(), version.clone());
        let path = Path::new(&self.base_path).join(format!("version_{}.json", version.version));
        let version_str = serde_json::to_string_pretty(&version)?;
        std::fs::write(path, version_str)?;
        Ok(())
    }

    pub fn get_version(&self, version: &str) -> Option<&ModelVersion> {
        self.versions.get(version)
    }

    pub fn get_latest_version(&self) -> Option<&ModelVersion> {
        self.versions.values().max_by_key(|v| v.timestamp)
    }

    pub fn compare_versions(&self, version1: &str, version2: &str) -> Result<HashMap<String, f64>> {
        let v1 = self.versions.get(version1)
            .ok_or_else(|| anyhow::anyhow!("Version {} not found", version1))?;
        let v2 = self.versions.get(version2)
            .ok_or_else(|| anyhow::anyhow!("Version {} not found", version2))?;

        let mut differences = HashMap::new();
        for (metric, value1) in &v1.metrics {
            if let Some(value2) = v2.metrics.get(metric) {
                differences.insert(metric.clone(), value2 - value1);
            }
        }

        Ok(differences)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    use tempfile::tempdir;

    #[test]
    fn test_model_versioning() -> Result<()> {
        let temp_dir = tempdir()?;
        let mut manager = ModelVersionManager::new(temp_dir.path().to_str().unwrap())?;

        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.85);
        metrics.insert("precision".to_string(), 0.82);

        let version = ModelVersion {
            version: "1.0.0".to_string(),
            timestamp: Utc::now(),
            metrics,
            input_size: 0,
            hidden_size: 0,
            output_size: 0,
            learning_rate: 0.0,
            window_size: 0,
        };

        manager.add_version(version)?;

        assert_eq!(manager.get_version("1.0.0").unwrap().version, "1.0.0");
        assert_eq!(manager.get_latest_version().unwrap().version, "1.0.0");

        let comparison = manager.compare_versions("1.0.0", "1.0.0")?;
        assert_eq!(comparison.len(), 2);
        assert_eq!(comparison["accuracy"], 0.0);
        assert_eq!(comparison["precision"], 0.0);

        Ok(())
    }
} 