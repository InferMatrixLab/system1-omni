//! Model ownership and request-sized serial FIFO execution.
use crate::{
    checkpoint,
    executor::Executor,
    processing::{Prepared, Processor},
};
use anyhow::Result;
use omni_runtime::{SerialScheduler, control::ExecutionContext};
use serde_json::Value;
use std::{
    path::Path,
    sync::{Arc, Mutex},
};

pub struct Engine {
    pub processor: Processor,
    scheduler: SerialScheduler,
    executor: Arc<Mutex<Executor>>,
}
impl Engine {
    pub async fn load(dir: &Path, library: &Path) -> Result<Self> {
        let (dir, library) = (dir.to_owned(), library.to_owned());
        let (processor, executor) = tokio::task::spawn_blocking(move || {
            checkpoint::verify(&dir)?;
            let processor = Processor::load(&dir)?;
            let executor = Executor::load(&dir, &library)?;
            Ok::<_, anyhow::Error>((processor, executor))
        })
        .await??;
        Ok(Self {
            processor,
            scheduler: SerialScheduler::default(),
            executor: Arc::new(Mutex::new(executor)),
        })
    }
    pub async fn infer(&self, prepared: Prepared, context: ExecutionContext) -> Result<Value> {
        let executor = self.executor.clone();
        self.scheduler
            .run_with_context(context, move || {
                let logits = executor
                    .lock()
                    .map_err(|_| anyhow::anyhow!("poisoned Kev executor"))?
                    .execute(&prepared.rows)?;
                prepared.context.finish(logits)
            })
            .await
    }
    pub async fn warmup(&self) -> Result<()> {
        let request = serde_json::json!({
            "state": "The light is green.",
            "questions": {
                "c": {"type":"choice", "criteria":{"red":null,"green":null}},
                "n": {"type":"noul", "instructions":"The light is green."},
                "s": {"type":"score", "criteria":["stop","wait","go"]}
            }
        });
        self.infer(
            self.processor.prepare(request.as_object().unwrap())?,
            ExecutionContext::default(),
        )
        .await?;
        Ok(())
    }
}
