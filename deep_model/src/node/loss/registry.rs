// builtin
use std::{
    collections::HashMap,
    sync::{OnceLock, RwLock},
};

// external

// internal
use crate::node::loss::loss_function::LossType;

pub fn init_loss_registry() {
    if REGISTRY_INSTANCE.get().is_none() {
        let activation_registry: LossRegistry = LossRegistry::init();

        REGISTRY_INSTANCE
            .set(RwLock::new(activation_registry))
            .unwrap();
    }
}

#[derive(Debug)]
pub struct LossRegistry {
    registry: HashMap<String, Box<dyn LossType>>,
}

static REGISTRY_INSTANCE: OnceLock<RwLock<LossRegistry>> = OnceLock::new();

impl LossRegistry {
    fn init() -> LossRegistry {
        let hmap: HashMap<String, Box<dyn LossType>> = HashMap::new();

        LossRegistry { registry: hmap }
    }

    pub fn get(name: &str) -> Box<dyn LossType> {
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

    pub fn register(name: &str, func: Box<dyn LossType>) {
        let mut guard = REGISTRY_INSTANCE
            .get()
            .expect("Tried to use registry prior to initialization")
            .write()
            .expect("Failed to acquire write lock on registry");

        guard.registry.insert(name.to_string(), func);
    }
}
