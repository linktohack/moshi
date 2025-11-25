use crate::nn::{
    linear, MaybeQuantizedEmbedding as Embedding, MaybeQuantizedLinear as Linear,
    MaybeQuantizedVarBuilder as VarBuilder,
};
use candle::{DType, Module, Result, Tensor};
use std::collections::HashMap;

/// Configuration for LUT-based conditioner (e.g., cfg, control tokens)
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LutConfig {
    pub n_bins: usize,
    pub dim: usize,
    #[serde(default)]
    pub possible_values: Vec<String>,
}

/// Configuration for continuous attribute conditioner
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ContinuousAttributeConfig {
    pub dim: usize,
    pub scale_factor: f32,
    pub max_period: f32,
}

/// Configuration for tensor conditioner (e.g., speaker embeddings)
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TensorConfig {
    pub dim: usize,
}

/// Configuration for different types of conditioners.
/// Uses serde's tag-based deserialization to handle the Python config format:
/// `{"type": "lut", "lut": {...}}` or `{"type": "tensor", "tensor": {...}}`
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ConditionerConfig {
    Lut {
        lut: LutConfig,
    },
    Tensor {
        tensor: TensorConfig,
    },
    ContinuousAttribute(ContinuousAttributeConfig),
}

pub type Config = HashMap<String, ConditionerConfig>;

#[derive(Debug, Clone)]
pub struct LutConditioner {
    embed: Embedding,
    output_proj: Linear,
    #[allow(unused)]
    learnt_padding: Tensor,
    possible_values: HashMap<String, usize>,
}

impl LutConditioner {
    pub fn new(output_dim: usize, cfg: &LutConfig, vb: VarBuilder) -> Result<Self> {
        let embed = Embedding::new(cfg.n_bins + 1, cfg.dim, vb.pp("embed"))?;
        let output_proj = linear(cfg.dim, output_dim, false, vb.pp("output_proj"))?;
        let learnt_padding = vb.get_as_tensor((1, 1, output_dim), "learnt_padding")?;
        let possible_values: HashMap<String, usize> =
            cfg.possible_values.iter().enumerate().map(|(i, v)| (v.to_string(), i)).collect();
        Ok(Self { embed, output_proj, learnt_padding, possible_values })
    }

    pub fn condition(&self, value: &str) -> Result<Condition> {
        let idx = match self.possible_values.get(value) {
            None => candle::bail!("unknown value for lut conditioner '{value}'"),
            Some(idx) => *idx,
        };
        let cond = Tensor::from_vec(vec![idx as u32], (1, 1), self.embed.embeddings().device())?
            .apply(&self.embed)?
            .apply(&self.output_proj)?;
        Ok(Condition::AddToInput(cond))
    }
}

#[derive(Debug, Clone)]
pub struct ContinuousAttributeConditioner {
    scale_factor: f32,
    max_period: f32,
    dim: usize,
    output_proj: Linear,
    #[allow(unused)]
    learnt_padding: Tensor,
    device: candle::Device,
}

impl ContinuousAttributeConditioner {
    pub fn new(output_dim: usize, cfg: &ContinuousAttributeConfig, vb: VarBuilder) -> Result<Self> {
        let output_proj = linear(cfg.dim, output_dim, false, vb.pp("output_proj"))?;
        let learnt_padding = vb.get_as_tensor((1, 1, output_dim), "learnt_padding")?;
        Ok(Self {
            scale_factor: cfg.scale_factor,
            max_period: cfg.max_period,
            dim: cfg.dim,
            output_proj,
            learnt_padding,
            device: vb.device().clone(),
        })
    }

    // `positions` should have shape (b, t, 1), the output will be (b, t, dim)
    pub fn create_sin_embeddings(&self, positions: &Tensor, dtype: DType) -> Result<Tensor> {
        let dev = positions.device();
        let half_dim = self.dim / 2;
        let positions = positions.to_dtype(dtype)?;
        let adim: Vec<_> = (0..half_dim)
            .map(|i| 1f32 / self.max_period.powf(i as f32 / (half_dim - 1) as f32))
            .collect();
        let adim = Tensor::from_vec(adim, (1, 1, ()), dev)?;
        let freqs = positions.broadcast_mul(&adim)?;
        let pos_emb = Tensor::cat(&[freqs.cos()?, freqs.sin()?], candle::D::Minus1)?;
        Ok(pos_emb)
    }

    // TODO(laurent): should we support different values per batch element?
    pub fn condition(&self, value: f32) -> Result<Condition> {
        let value = value * self.scale_factor;
        let positions = Tensor::full(value, (1, 1, 1), &self.device)?;
        let cond = self
            .create_sin_embeddings(&positions, DType::F32)?
            .to_dtype(self.output_proj.dtype())?
            .apply(&self.output_proj)?;
        Ok(Condition::AddToInput(cond))
    }
}

/// Tensor conditioner for speaker embeddings (cross-attention source).
/// Takes pre-computed embeddings and projects them to model dimension.
#[derive(Debug, Clone)]
pub struct TensorConditioner {
    output_proj: Linear,
    learnt_padding: Tensor,
    #[allow(unused)]
    dim: usize,
    output_dim: usize,
}

impl TensorConditioner {
    pub fn new(output_dim: usize, cfg: &TensorConfig, vb: VarBuilder) -> Result<Self> {
        let output_proj = linear(cfg.dim, output_dim, false, vb.pp("output_proj"))?;
        let learnt_padding = vb.get_as_tensor((1, 1, output_dim), "learnt_padding")?;
        Ok(Self {
            output_proj,
            learnt_padding,
            dim: cfg.dim,
            output_dim,
        })
    }

    /// Condition on a pre-computed tensor embedding.
    /// Input tensor should have shape [B, T, dim] where dim matches config.dim.
    /// Mask should have shape [B, T] indicating valid positions.
    pub fn condition(&self, tensor: &Tensor, mask: &Tensor) -> Result<Condition> {
        // Project to output dimension
        let cond = self.output_proj.forward(tensor)?;
        // Apply mask and learnt padding for invalid positions
        let mask_f = mask.unsqueeze(2)?.to_dtype(cond.dtype())?;
        let inv_mask = (1.0 - mask_f.clone())?;
        let masked_cond = cond.broadcast_mul(&mask_f)?;
        let padding_contrib = self.learnt_padding.broadcast_mul(&inv_mask)?;
        let cond = masked_cond.broadcast_add(&padding_contrib)?;
        Ok(Condition::CrossAttention(cond))
    }

    /// Get learnt padding for null conditioning
    pub fn null_condition(&self, batch_size: usize, seq_len: usize) -> Result<Condition> {
        let dev = self.learnt_padding.device();
        let dtype = self.learnt_padding.dtype();
        let zeros = Tensor::zeros((batch_size, seq_len, self.output_dim), dtype, dev)?;
        Ok(Condition::CrossAttention(zeros))
    }
}

#[derive(Debug, Clone)]
pub enum Conditioner {
    Lut(LutConditioner),
    ContinuousAttribute(ContinuousAttributeConditioner),
    Tensor(TensorConditioner),
}

#[derive(Debug, Clone)]
pub struct ConditionProvider {
    conditioners: HashMap<String, Conditioner>,
}

/// Condition output type - either additive or cross-attention source.
#[derive(Debug, Clone)]
pub enum Condition {
    /// Added to the input embeddings (sum fusion)
    AddToInput(Tensor),
    /// Used as cross-attention source
    CrossAttention(Tensor),
}

/// Combined condition tensors ready for the model.
#[derive(Debug, Clone, Default)]
pub struct ConditionTensors {
    /// Sum of all "sum" conditions to add to input
    pub sum: Option<Tensor>,
    /// Cross-attention source tensor
    pub cross: Option<Tensor>,
}

impl ConditionTensors {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_sum(&mut self, tensor: Tensor) -> Result<()> {
        self.sum = Some(match self.sum.take() {
            None => tensor,
            Some(existing) => (existing + tensor)?,
        });
        Ok(())
    }

    pub fn set_cross(&mut self, tensor: Tensor) {
        self.cross = Some(tensor);
    }
}

impl ConditionProvider {
    pub fn new(output_dim: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("conditioners");
        let mut conditioners = HashMap::new();
        for (conditioner_name, conditioner_cfg) in cfg.iter() {
            let vb = vb.pp(conditioner_name);
            let conditioner = match conditioner_cfg {
                ConditionerConfig::Lut { lut } => {
                    Conditioner::Lut(LutConditioner::new(output_dim, lut, vb)?)
                }
                ConditionerConfig::Tensor { tensor } => {
                    Conditioner::Tensor(TensorConditioner::new(output_dim, tensor, vb)?)
                }
                ConditionerConfig::ContinuousAttribute(cfg) => Conditioner::ContinuousAttribute(
                    ContinuousAttributeConditioner::new(output_dim, cfg, vb)?,
                ),
            };
            conditioners.insert(conditioner_name.to_string(), conditioner);
        }
        Ok(Self { conditioners })
    }

    /// Get a reference to a conditioner by name
    pub fn get(&self, name: &str) -> Option<&Conditioner> {
        self.conditioners.get(name)
    }

    /// Check if a tensor conditioner exists
    pub fn has_tensor_conditioner(&self, name: &str) -> bool {
        matches!(self.conditioners.get(name), Some(Conditioner::Tensor(_)))
    }

    pub fn condition_lut(&self, name: &str, value: &str) -> Result<Condition> {
        let lut = match self.conditioners.get(name) {
            None => candle::bail!("unknown conditioner {name}"),
            Some(Conditioner::Lut(l)) => l,
            Some(_) => candle::bail!("cannot use LUT conditioner with wrong type for {name}"),
        };
        let cond = lut.condition(value)?;
        Ok(cond)
    }

    pub fn condition_lut_or_null(&self, name: &str, value: Option<&str>) -> Result<Condition> {
        match value {
            Some(v) => self.condition_lut(name, v),
            None => self.learnt_padding(name),
        }
    }

    pub fn condition_tensor(&self, name: &str, tensor: &Tensor, mask: &Tensor) -> Result<Condition> {
        let tc = match self.conditioners.get(name) {
            None => candle::bail!("unknown conditioner {name}"),
            Some(Conditioner::Tensor(t)) => t,
            Some(_) => candle::bail!("cannot use tensor conditioner with wrong type for {name}"),
        };
        tc.condition(tensor, mask)
    }

    pub fn condition_cont(&self, name: &str, value: f32) -> Result<Condition> {
        let c = match self.conditioners.get(name) {
            None => candle::bail!("unknown conditioner {name}"),
            Some(Conditioner::ContinuousAttribute(c)) => c,
            Some(_) => candle::bail!("cannot use continuous attribute conditioner for {name}"),
        };
        let cond = c.condition(value)?;
        Ok(cond)
    }

    pub fn learnt_padding(&self, name: &str) -> Result<Condition> {
        let c = match self.conditioners.get(name) {
            None => candle::bail!("unknown conditioner {name}"),
            Some(Conditioner::ContinuousAttribute(c)) => Condition::AddToInput(c.learnt_padding.clone()),
            Some(Conditioner::Lut(c)) => Condition::AddToInput(c.learnt_padding.clone()),
            Some(Conditioner::Tensor(t)) => Condition::CrossAttention(t.learnt_padding.clone()),
        };
        Ok(c)
    }
}
