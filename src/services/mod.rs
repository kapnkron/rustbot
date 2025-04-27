
pub struct UserService {
    // Comment out unused field
    // users: Arc<Mutex<Vec<User>>>,
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

impl TradeService {
    pub fn new() -> Self {
        Self {
            // trades: Arc::new(Mutex::new(Vec::new()))
         }
    }
} 