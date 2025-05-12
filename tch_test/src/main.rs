use pyo3::prelude::*;
use pyo3::types::PyModule;

fn main() -> PyResult<()> {
    Python::with_gil(|py| {
        let ml_infer = PyModule::import(py, "ml_infer")?;
        let cuda_add = ml_infer.getattr("cuda_add")?;
        let result: Vec<f32> = cuda_add.call1((3.0f32, 4.0f32))?.extract()?;
        println!("Result from Python: {:?}", result);
        Ok(())
    })
} 