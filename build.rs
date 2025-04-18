use std::process::Command;
use std::env;

fn main() {
    // Check if we're on a Unix-like system
    if cfg!(unix) {
        // Check for libykneomgr
        let pkg_config_output = Command::new("pkg-config")
            .arg("--exists")
            .arg("libykneomgr")
            .output();

        if let Err(_) = pkg_config_output {
            println!("cargo:warning=libykneomgr not found. Attempting to install...");
            
            // Try to install using package manager
            let install_cmd = if cfg!(target_os = "ubuntu") || cfg!(target_os = "debian") {
                "sudo apt-get install -y libykneomgr-dev"
            } else if cfg!(target_os = "fedora") || cfg!(target_os = "centos") {
                "sudo dnf install -y ykneomgr-devel"
            } else if cfg!(target_os = "arch") {
                "sudo pacman -S --noconfirm yubikey-neo-manager"
            } else {
                println!("cargo:warning=Unsupported OS. Please install libykneomgr manually.");
                return;
            };

            let status = Command::new("sh")
                .arg("-c")
                .arg(install_cmd)
                .status();

            if let Err(e) = status {
                println!("cargo:warning=Failed to install libykneomgr: {}", e);
                println!("cargo:warning=Please install it manually using your package manager.");
            }
        }

        // Check for udev rules
        let udev_rules = "/etc/udev/rules.d/70-yubikey.rules";
        if !std::path::Path::new(udev_rules).exists() {
            println!("cargo:warning=YubiKey udev rules not found. Attempting to install...");
            
            let udev_rules_content = r#"# YubiKey
ATTRS{idVendor}=="1050", ATTRS{idProduct}=="0010|0110|0111|0112|0113|0114|0115|0116|0120|0200|0401|0402|0403|0404|0405|0406|0407|0410", MODE="0660", GROUP="plugdev"
"#;

            if let Err(e) = std::fs::write(udev_rules_content, udev_rules) {
                println!("cargo:warning=Failed to write udev rules: {}", e);
                println!("cargo:warning=Please create the file manually at {}", udev_rules);
            } else {
                // Reload udev rules
                let _ = Command::new("sudo")
                    .arg("udevadm")
                    .arg("control")
                    .arg("--reload-rules");
                let _ = Command::new("sudo")
                    .arg("udevadm")
                    .arg("trigger");
            }
        }
    } else {
        println!("cargo:warning=YubiKey support is only available on Unix-like systems");
    }
} 