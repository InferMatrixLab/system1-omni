//! Cua checkpoint/provenance validation with the shared Qwen execution.
use super::VisionCheckpoint;
use crate::{cuda::DeviceBf16Matrix, image_preprocess::ProcessedImage};
use anyhow::Result;
use half::bf16;
use std::path::Path;
pub struct VisionModel(omni_qwen3_5_native::vision::VisionModel);
impl VisionModel {
    pub fn load(base: impl AsRef<Path>, adapter: impl AsRef<Path>, library: &Path) -> Result<Self> {
        let checkpoint = VisionCheckpoint::load(base, adapter)?;
        let c = checkpoint.config();
        // Serialize the parsed struct instead of re-listing its fields: a member
        // added to VisionConfig can no longer be silently dropped from the
        // shared model configuration.
        let config =
            omni_qwen3_5_native::vision::VisionConfig::from_value(serde_json::to_value(c)?)?;
        let base = checkpoint
            .base_names()
            .map(|n| Ok((n.to_owned(), checkpoint.base_tensor(n)?)))
            .collect::<Result<Vec<_>>>()?;
        let lora = checkpoint
            .adapter_names()
            .map(|n| Ok((n.to_owned(), checkpoint.adapter_tensor(n)?)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self(
            omni_qwen3_5_native::vision::VisionModel::from_tensors(config, base, lora, library)?,
        ))
    }
    pub fn synchronize(&self) -> Result<()> {
        self.0.synchronize()
    }
    pub fn forward(&mut self, image: &ProcessedImage) -> Result<Vec<bf16>> {
        self.0.forward(image)
    }
    /// Borrow synchronized shared-executor output until the language consumer drains.
    pub fn forward_device(&mut self, image: &ProcessedImage) -> Result<DeviceBf16Matrix<'_>> {
        self.0.forward_device(image)
    }
    pub fn forward_with_trace(
        &mut self,
        image: &ProcessedImage,
        callback: impl FnMut(&str, &[bf16]) -> Result<()>,
    ) -> Result<Vec<bf16>> {
        self.0.forward_with_trace(image, callback)
    }
}
