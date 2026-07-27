use crate::error::ClipError;
#[cfg(feature = "hf-hub")]
use hf_hub::{split_id, HFClient};
use std::env;
use std::path::{Path, PathBuf};

/// Files to download from the Hugging Face repository.
pub const MODEL_FILES: &[&str] = &[
    "model_config.json",
    "open_clip_config.json",
    "special_tokens_map.json",
    "text.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "visual.onnx",
    "text.onnx.data",
    "visual.onnx.data",
];

/// Ensures that the model files are present locally.
#[cfg(feature = "hf-hub")]
pub async fn get_hf_model(model_id: &str, cache_dir: Option<&Path>) -> Result<PathBuf, ClipError> {
    let client = match cache_dir {
        Some(dir) => HFClient::builder().cache_dir(dir).build()?,
        None => HFClient::new()?,
    };

    let (owner, name) = split_id(model_id);
    let repo = client.model(owner, name);

    let mut model_dir = None;
    for &file in MODEL_FILES {
        tracing::info!("Downloading {file}...");
        let downloaded_file = repo.download_file().filename(file).send().await?;

        if model_dir.is_none() {
            model_dir = downloaded_file.parent().map(ToOwned::to_owned);
        }
    }

    model_dir.ok_or_else(|| {
        ClipError::HfHub(format!(
            "Could not determine model directory for '{model_id}'"
        ))
    })
}

/// Get default model base folder (where `pull_onnx.py` also exports to by default).
#[must_use]
pub fn get_default_base_folder() -> PathBuf {
    env::home_dir().map_or_else(
        || Path::new(".open_clip_cache").to_owned(),
        |p| p.join(".cache/open_clip_rs"),
    )
}

/// Verify that a model directory is valid, and contains the right files.
pub fn verify_model_dir(model_dir: &Path) -> Result<(), ClipError> {
    if !model_dir.exists() {
        return Err(ClipError::ModelFolderNotFound(model_dir.to_owned()));
    }

    for file in MODEL_FILES {
        let path = model_dir.join(file);
        if !path.is_file() {
            return Err(ClipError::MissingModelFile {
                model_dir: model_dir.to_owned(),
                file: file.to_string(),
            });
        }
    }

    Ok(())
}