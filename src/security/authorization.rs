use crate::error::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Permission {
    ReadMarketData,
    WriteMarketData,
    ExecuteTrades,
    ManageUsers,
    ViewAnalytics,
    ManageModels,
}

#[derive(Debug, Clone)]
pub struct Role {
    name: String,
    permissions: Vec<Permission>,
}

#[derive(Debug, Clone)]
pub struct AuthorizationManager {
    roles: Arc<RwLock<HashMap<String, Role>>>,
}

impl AuthorizationManager {
    pub fn new() -> Self {
        let mut roles = HashMap::new();
        
        // Define default roles
        roles.insert(
            "admin".to_string(),
            Role {
                name: "admin".to_string(),
                permissions: vec![
                    Permission::ReadMarketData,
                    Permission::WriteMarketData,
                    Permission::ExecuteTrades,
                    Permission::ManageUsers,
                    Permission::ViewAnalytics,
                    Permission::ManageModels,
                ],
            },
        );

        roles.insert(
            "trader".to_string(),
            Role {
                name: "trader".to_string(),
                permissions: vec![
                    Permission::ReadMarketData,
                    Permission::ExecuteTrades,
                    Permission::ViewAnalytics,
                ],
            },
        );

        roles.insert(
            "analyst".to_string(),
            Role {
                name: "analyst".to_string(),
                permissions: vec![
                    Permission::ReadMarketData,
                    Permission::ViewAnalytics,
                ],
            },
        );

        Self {
            roles: Arc::new(RwLock::new(roles)),
        }
    }

    pub async fn has_permission(&self, role: &str, permission: &Permission) -> bool {
        let roles = self.roles.read().await;
        if let Some(role) = roles.get(role) {
            role.permissions.contains(permission)
        } else {
            false
        }
    }

    pub async fn add_role(&self, name: String, permissions: Vec<Permission>) -> Result<()> {
        let mut roles = self.roles.write().await;
        if roles.contains_key(&name) {
            return Err(anyhow::anyhow!("Role already exists"));
        }

        roles.insert(
            name.clone(),
            Role {
                name,
                permissions,
            },
        );

        Ok(())
    }

    pub async fn update_role_permissions(
        &self,
        role_name: &str,
        permissions: Vec<Permission>,
    ) -> Result<()> {
        let mut roles = self.roles.write().await;
        if let Some(role) = roles.get_mut(role_name) {
            role.permissions = permissions;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Role not found"))
        }
    }

    pub async fn remove_role(&self, role_name: &str) -> Result<()> {
        let mut roles = self.roles.write().await;
        if roles.remove(role_name).is_some() {
            Ok(())
        } else {
            Err(anyhow::anyhow!("Role not found"))
        }
    }

    pub async fn get_role_permissions(&self, role_name: &str) -> Option<Vec<Permission>> {
        let roles = self.roles.read().await;
        roles.get(role_name).map(|role| role.permissions.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_authorization() -> Result<()> {
        let authz_manager = AuthorizationManager::new();

        // Test default roles
        assert!(authz_manager.has_permission("admin", &Permission::ManageUsers).await);
        assert!(authz_manager.has_permission("trader", &Permission::ExecuteTrades).await);
        assert!(!authz_manager.has_permission("analyst", &Permission::ExecuteTrades).await);

        // Test adding new role
        authz_manager.add_role(
            "custom".to_string(),
            vec![Permission::ReadMarketData, Permission::ViewAnalytics],
        ).await?;

        assert!(authz_manager.has_permission("custom", &Permission::ReadMarketData).await);
        assert!(!authz_manager.has_permission("custom", &Permission::ExecuteTrades).await);

        // Test updating role
        authz_manager.update_role_permissions(
            "custom",
            vec![Permission::ExecuteTrades],
        ).await?;

        assert!(!authz_manager.has_permission("custom", &Permission::ReadMarketData).await);
        assert!(authz_manager.has_permission("custom", &Permission::ExecuteTrades).await);

        // Test removing role
        authz_manager.remove_role("custom").await?;
        assert!(!authz_manager.has_permission("custom", &Permission::ExecuteTrades).await);

        Ok(())
    }
} 