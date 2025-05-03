
pub struct UserService {
    // Comment out unused field
    // users: Arc<Mutex<Vec<User>>>,
}

impl Default for UserService {
    fn default() -> Self {
        Self::new()
    }
}

impl UserService {
    pub fn new() -> Self {
        Self {
            // users: Arc::new(Mutex::new(Vec::new())) 
        }
    }
}

pub struct TradeService {
    // Comment out unused field
    // trades: Arc<Mutex<Vec<Trade>>>,
}

impl Default for TradeService {
    fn default() -> Self {
        Self::new()
    }
}

impl TradeService {
    pub fn new() -> Self {
        Self {
            // trades: Arc::new(Mutex::new(Vec::new()))
         }
    }
} 