// builtin
use std::{
    collections::HashMap,
    sync::{OnceLock, RwLock},
};

// external

// internal
use crate::node::activation::{
    activation_function::ActivationType,
    types::{none::LinearActivation, relu::ReLUActivation},
};

pub fn init_activation_registry() {
    if REGISTRY_INSTANCE.get().is_none() {
        let activation_registry: ActivationRegistry = ActivationRegistry::init();

        REGISTRY_INSTANCE
            .set(RwLock::new(activation_registry))
            .unwrap();

        ActivationRegistry::register(ReLUActivation.name(), Box::new(ReLUActivation));
        ActivationRegistry::register(LinearActivation.name(), Box::new(LinearActivation));
    }
}

#[derive(Debug)]
pub struct ActivationRegistry {
    registry: HashMap<String, Box<dyn ActivationType>>,
}

static REGISTRY_INSTANCE: OnceLock<RwLock<ActivationRegistry>> = OnceLock::new();

impl ActivationRegistry {
    fn init() -> ActivationRegistry {
        let hmap: HashMap<String, Box<dyn ActivationType>> = HashMap::new();

        ActivationRegistry { registry: hmap }
    }

    pub fn get(name: &str) -> Box<dyn ActivationType> {
        let guard = REGISTRY_INSTANCE
            .get()
            .expect("[ACTIVATION] Tried to use registry prior to initialization")
            .read()
            .expect("[ACTIVATION] Failed to acquire read lock on registry");

        guard
            .registry
            .get(name)
            .map(|f| f.copy())
            .expect("[ACTIVATION] Failed to fetch activation function from registry")
    }

    pub fn register(name: &str, func: Box<dyn ActivationType>) {
        let mut guard = REGISTRY_INSTANCE
            .get()
            .expect("[ACTIVATION] Tried to use registry prior to initialization")
            .write()
            .expect("[ACTIVATION] Failed to acquire write lock on registry");

        guard.registry.insert(name.to_string(), func);
    }
}
