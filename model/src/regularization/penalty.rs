// builtin
use std::{
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
};

// external
use serde::{Deserialize, Serialize};

// internal
use crate::{node::NodeRef, regularization::penalty::no_penalty::builder::NullBuilder};
pub mod l1_penalty;
pub mod l2_penalty;
pub mod no_penalty;

#[derive(Serialize, Deserialize, Clone)]
pub enum PenaltyType {
    L2 { alpha: f32 },
    L1 { alpha: f32 },
    None,
}

pub type PenaltyRef<'a> = Rc<RefCell<dyn PenaltyUnit<'a> + 'a>>;

pub struct PenaltyConfig<'a> {
    builder: Box<dyn PenaltyBuilder<'a> + 'a>,
}

impl<'a> PenaltyConfig<'a> {
    pub fn new(builder: impl PenaltyBuilder<'a> + 'a) -> PenaltyConfig<'a> {
        PenaltyConfig {
            builder: Box::new(builder),
        }
    }

    pub fn none() -> PenaltyConfig<'a> {
        PenaltyConfig {
            builder: Box::new(NullBuilder),
        }
    }

    pub fn create_first(&self, parameter_node: &NodeRef<'a>) -> PenaltyContainer<'a> {
        self.builder.create_first(parameter_node)
    }

    pub fn create_new(
        &self,
        prev: &PenaltyRef<'a>,
        parameter_node: &NodeRef<'a>,
    ) -> PenaltyContainer<'a> {
        self.builder.create_new(prev, parameter_node)
    }

    pub fn get_builder(&self) -> &Box<dyn PenaltyBuilder<'a> + 'a> {
        &self.builder
    }

    pub fn get_type(&self) -> PenaltyType {
        self.builder.get_associated_type()
    }
}

pub struct PenaltyContainer<'a> {
    unit: Rc<RefCell<dyn PenaltyUnit<'a> + 'a>>,
}

impl<'a> PenaltyContainer<'a> {
    pub fn new(penalty_unit: impl PenaltyUnit<'a> + 'a) -> PenaltyContainer<'a> {
        PenaltyContainer {
            unit: Rc::new(RefCell::new(penalty_unit)),
        }
    }

    pub fn borrow(&self) -> Ref<dyn PenaltyUnit<'a> + 'a> {
        self.unit.borrow()
    }

    pub fn borrow_mut(&self) -> RefMut<dyn PenaltyUnit<'a> + 'a> {
        self.unit.borrow_mut()
    }

    pub fn get_ref(&self) -> PenaltyRef<'a> {
        Rc::clone(&self.unit)
    }
}

pub trait PenaltyUnit<'a> {
    fn add_penalty_input(&mut self, input: &PenaltyRef<'a>);

    fn add_parameter_input(&mut self, parameter_node: &NodeRef<'a>);

    fn get_output_ref(&self) -> &NodeRef<'a>;

    fn is_null(&self) -> bool;
}

pub trait PenaltyBuilder<'a> {
    fn create_first(&self, parameter_node: &NodeRef<'a>) -> PenaltyContainer<'a>;

    fn create_new(
        &self,
        prev: &PenaltyRef<'a>,
        parameter_node: &NodeRef<'a>,
    ) -> PenaltyContainer<'a>;

    fn get_associated_type(&self) -> PenaltyType;
}
