use std::panic;

fn main() {
    println!("Checking system configuration...");

    // Skip the has_cuda check for now, assume it might work
    // let cuda_available = tch::utils::has_cuda();
    // println!("CUDA available: {}", cuda_available);

    // Try to force CUDA device
    let device = tch::Device::Cuda(0);
    println!("Attempting to use device: {:?}", device);

    // Try a simple operation on the CUDA device.
    // Use catch_unwind to handle potential panics if CUDA fails.
    let result = panic::catch_unwind(|| {
        println!("Attempting to create tensor on {:?}...", device);
        let tensor = tch::Tensor::randn(&[2, 2], (tch::Kind::Float, device));
        println!("Successfully created tensor on CUDA device:");
        tensor.print();
    });

    if result.is_err() {
        eprintln!("PANIC CAUGHT! Failed to use CUDA device.");
        println!("Falling back to CPU.");
        let device = tch::Device::Cpu;
        println!("Using device: {:?}", device);
        // Potentially create tensor on CPU here if needed
        // let tensor = tch::Tensor::randn(&[2, 2], (tch::Kind::Float, device));
    }

    // Your tensor operations can continue here if the main logic
    // doesn't depend on the tensor created in the catch_unwind block,
    // or you handle the fallback appropriately.
} 