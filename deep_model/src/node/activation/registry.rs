// builtin
use std::{
    collections::HashMap,
    sync::{OnceLock, RwLock},
};

// external

// internal
use crate::node::activation::{activation_function::ActivationType, types::relu::ReLUActivation};

pub fn init_registry() {
    if REGISTRY_INSTANCE.get().is_none() {
        let activation_registry: Registry = Registry::init();

        REGISTRY_INSTANCE
            .set(RwLock::new(activation_registry))
            .unwrap();

        Registry::register(ReLUActivation.name(), Box::new(ReLUActivation));
    }
}

#[derive(Debug)]
pub struct Registry {
    registry: HashMap<String, Box<dyn ActivationType>>,
}

static REGISTRY_INSTANCE: OnceLock<RwLock<Registry>> = OnceLock::new();

impl Registry {
    fn init() -> Registry {
        let hmap: HashMap<String, Box<dyn ActivationType>> = HashMap::new();

        Registry { registry: hmap }
    }

    pub fn get(name: &str) -> Box<dyn ActivationType> {
        let guard = REGISTRY_INSTANCE
            .get()
            .expect("Tried to use registry prior to initialization")
            .read()
            .expect("Failed to acquire read lock on registry");

        guard
            .registry
            .get(name)
            .map(|f| f.copy())
            .expect("Failed to fetch activation function from registry")
    }

    pub fn register(name: &str, func: Box<dyn ActivationType>) {
        let mut guard = REGISTRY_INSTANCE
            .get()
            .expect("Tried to use registry prior to initialization")
            .write()
            .expect("Failed to acquire write lock on registry");

        guard.registry.insert(name.to_string(), func);
    }
}
