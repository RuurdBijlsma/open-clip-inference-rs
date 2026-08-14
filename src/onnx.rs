use crate::ClipError;
use ort::ep::ExecutionProviderDispatch;
use ort::session::{Session, builder::GraphOptimizationLevel};
use std::path::Path;
use std::sync::RwLock;

#[derive(Debug)]
pub struct OnnxSession {
    pub session: RwLock<Session>,
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub optimization_level: Option<GraphOptimizationLevel>,
    pub intra_threads: Option<usize>,
    pub inter_threads: Option<usize>,
    pub memory_pattern: Option<bool>,
}

impl OnnxSession {
    pub fn new(
        path: impl AsRef<Path>,
        execution_providers: &[ExecutionProviderDispatch],
        optimization_level: Option<GraphOptimizationLevel>,
        intra_threads: Option<usize>,
        inter_threads: Option<usize>,
        memory_pattern: Option<bool>,
    ) -> Result<Self, ClipError> {
        let mut session_builder =
            Session::builder()?.with_execution_providers(execution_providers)?;
        if let Some(optimization_level) = optimization_level {
            session_builder = session_builder.with_optimization_level(optimization_level)?;
        }
        if let Some(intra_threads) = intra_threads {
            session_builder = session_builder.with_intra_threads(intra_threads)?;
        }
        if let Some(inter_threads) = inter_threads {
            session_builder = session_builder.with_inter_threads(inter_threads)?;
        }
        if let Some(memory_pattern) = memory_pattern {
            session_builder = session_builder.with_memory_pattern(memory_pattern)?;
        }
        let session = session_builder.commit_from_file(path)?;

        Ok(Self {
            session: RwLock::new(session),
            execution_providers: execution_providers.to_vec(),
            intra_threads,
            inter_threads,
            memory_pattern,
            optimization_level,
        })
    }

    /// Helper to check if the model expects a specific input name
    pub fn has_input(&self, name: &str) -> Result<bool, ClipError> {
        let session = self.session.read()?;
        Ok(session.inputs().iter().any(|i| i.name() == name))
    }

    /// Helper to find the first likely input name for a specific role
    pub fn find_input(&self, possibilities: &[&str]) -> Result<Option<String>, ClipError> {
        let session = self.session.read()?;
        for &p in possibilities {
            if session.inputs().iter().any(|i| i.name() == p) {
                return Ok(Some(p.to_string()));
            }
        }
        Ok(None)
    }
}
