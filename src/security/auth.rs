use crate::utils::error::Result;
use chrono::{Duration, Utc};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,  // User ID
    pub exp: usize,   // Expiration time
    pub role: String, // User role
}

#[derive(Debug, Clone)]
pub struct User {
    pub id: String,
    pub username: String,
    pub password_hash: String,
    pub role: String,
    pub is_active: bool,
}

#[derive(Debug, Clone)]
pub struct AuthManager {
    users: Arc<RwLock<HashMap<String, User>>>,
    jwt_secret: String,
    token_expiration: Duration,
}

impl AuthManager {
    pub fn new(jwt_secret: String, token_expiration: Duration) -> Self {
        Self {
            users: Arc::new(RwLock::new(HashMap::new())),
            jwt_secret,
            token_expiration,
        }
    }

    pub async fn register_user(&self, username: String, password: String, role: String) -> Result<String> {
        let user_id = uuid::Uuid::new_v4().to_string();
        let password_hash = bcrypt::hash(password, bcrypt::DEFAULT_COST)?;
        
        let user = User {
            id: user_id.clone(),
            username: username.clone(),
            password_hash,
            role,
            is_active: true,
        };

        let mut users = self.users.write().await;
        users.insert(user_id.clone(), user);

        Ok(user_id)
    }

    pub async fn authenticate(&self, username: &str, password: &str) -> Result<String> {
        let users = self.users.read().await;
        let user = users.values()
            .find(|u| u.username == username && u.is_active)
            .ok_or_else(|| anyhow::anyhow!("Invalid credentials"))?;

        if !bcrypt::verify(password, &user.password_hash)? {
            return Err(anyhow::anyhow!("Invalid credentials"));
        }

        let claims = Claims {
            sub: user.id.clone(),
            exp: (Utc::now() + self.token_expiration).timestamp() as usize,
            role: user.role.clone(),
        };

        let token = encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(self.jwt_secret.as_bytes()),
        )?;

        Ok(token)
    }

    pub fn validate_token(&self, token: &str) -> Result<Claims> {
        let token_data = decode::<Claims>(
            token,
            &DecodingKey::from_secret(self.jwt_secret.as_bytes()),
            &Validation::default(),
        )?;

        Ok(token_data.claims)
    }

    pub async fn get_user(&self, user_id: &str) -> Option<User> {
        let users = self.users.read().await;
        users.get(user_id).cloned()
    }

    pub async fn deactivate_user(&self, user_id: &str) -> Result<()> {
        let mut users = self.users.write().await;
        if let Some(user) = users.get_mut(user_id) {
            user.is_active = false;
            Ok(())
        } else {
            Err(anyhow::anyhow!("User not found"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[tokio::test]
    async fn test_auth_flow() -> Result<()> {
        let auth_manager = AuthManager::new(
            "test_secret".to_string(),
            Duration::hours(1),
        );

        // Register user
        let user_id = auth_manager.register_user(
            "test_user".to_string(),
            "password123".to_string(),
            "user".to_string(),
        ).await?;

        // Authenticate
        let token = auth_manager.authenticate("test_user", "password123").await?;
        
        // Validate token
        let claims = auth_manager.validate_token(&token)?;
        assert_eq!(claims.sub, user_id);
        assert_eq!(claims.role, "user");

        // Get user
        let user = auth_manager.get_user(&user_id).await;
        assert!(user.is_some());
        assert_eq!(user.unwrap().username, "test_user");

        // Deactivate user
        auth_manager.deactivate_user(&user_id).await?;
        let user = auth_manager.get_user(&user_id).await;
        assert!(user.is_some());
        assert!(!user.unwrap().is_active);

        Ok(())
    }
} 