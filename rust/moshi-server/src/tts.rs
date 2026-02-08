// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

//! Native Rust TTS implementation using Candle.
//!
//! This module provides a native Rust TTS model using the DSM (Delayed Streams Modeling)
//! architecture, matching the Python implementation in tts.py.

use anyhow::{Context, Result};
use axum::extract::ws;
use candle::{DType, Device, IndexOp, Tensor};
use candle_transformers::generation::LogitsProcessor;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use moshi::tts_state_machine::{Entry, State, StateMachine, TokenIds};
use candle::Module;
use candle_nn::VarBuilder;

// ===== Sinusoidal Positional Embeddings for Cross-Attention =====
// Python fuser adds positional embeddings to cross-attention source.
// This matches Python's create_sin_embedding + ConditionFuser.get_cross()

/// Create sinusoidal positional embedding with shape [B, T, dim].
/// Matches Python's create_sin_embedding from transformer.py
fn create_sin_embedding(
    seq_len: usize,
    dim: usize,
    device: &Device,
    dtype: DType,
    max_period: f32,
) -> Result<Tensor> {
    assert!(dim % 2 == 0, "dim must be even for sinusoidal embeddings");
    let half_dim = dim / 2;

    // positions: [0, 1, 2, ..., seq_len-1] with shape [1, seq_len, 1]
    let positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
    let positions = Tensor::from_vec(positions, (1, seq_len, 1), device)?;

    // adim: frequency scales for each dimension
    // Python: phase = positions / (max_period ** (adim / (half_dim - 1)))
    let adim: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / max_period.powf(i as f32 / (half_dim - 1) as f32))
        .collect();
    let adim = Tensor::from_vec(adim, (1, 1, half_dim), device)?;

    // Compute phase = positions * (1 / max_period^(adim / (half_dim-1)))
    let phase = positions.broadcast_mul(&adim)?;

    // Concatenate [cos(phase), sin(phase)] along last dimension
    let pos_emb = Tensor::cat(&[phase.cos()?, phase.sin()?], candle::D::Minus1)?;
    Ok(pos_emb.to_dtype(dtype)?)
}

/// Add sinusoidal positional embeddings to cross-attention source.
/// Matches Python ConditionFuser.get_cross() when cross_attention_pos_emb=True.
fn add_cross_attention_pos_emb(
    cross_src: &Tensor,
    scale: f32,
    max_period: f32,
) -> Result<Tensor> {
    let (_b, seq_len, dim) = cross_src.dims3()?;
    let pos_emb = create_sin_embedding(seq_len, dim, cross_src.device(), cross_src.dtype(), max_period)?;
    // cross = cross + scale * pos_emb
    let scaled_pos_emb = (pos_emb * scale as f64)?;
    Ok(cross_src.broadcast_add(&scaled_pos_emb)?)
}

// ===== TTS-Specific DepFormer Implementation =====
// The TTS model uses a shared transformer with per-slice gating,
// which differs from the standard moshi-core DepFormer that has
// separate transformers per slice.

/// Per-slice gating module (SiLU gated linear)
/// Matches Python's ActivationGating in gating.py
#[derive(Debug, Clone)]
struct TtsGating {
    linear_in: candle_nn::Linear,
    linear_out: candle_nn::Linear,
}

impl TtsGating {
    fn new(d_model: usize, dim_feedforward: usize, vb: VarBuilder) -> Result<Self> {
        // Calculate hidden size matching Python's gating.py
        // Python: hidden = (21 * dim) // 8 when dim_feedforward == 4 * dim
        //         hidden = (2 * dim_feedforward) // 3 otherwise
        let hidden = if dim_feedforward == 4 * d_model {
            (21 * d_model) / 8
        } else {
            (2 * dim_feedforward) / 3
        };
        let linear_in = candle_nn::linear_no_bias(d_model, 2 * hidden, vb.pp("linear_in"))?;
        let linear_out = candle_nn::linear_no_bias(hidden, d_model, vb.pp("linear_out"))?;
        Ok(Self { linear_in, linear_out })
    }

    /// Load gating with dimensions inferred from weights in the safetensors file
    #[allow(dead_code)]
    fn load_from_weights(vb: VarBuilder) -> Result<Self> {
        // Load weights directly - dimensions are inferred from the file
        let linear_in_weight = vb.pp("linear_in").get_with_hints((), "weight", candle_nn::Init::Const(0.0))?;
        let linear_out_weight = vb.pp("linear_out").get_with_hints((), "weight", candle_nn::Init::Const(0.0))?;
        let linear_in = candle_nn::Linear::new(linear_in_weight, None);
        let linear_out = candle_nn::Linear::new(linear_out_weight, None);
        Ok(Self { linear_in, linear_out })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.linear_in.forward(xs)?;
        let (b, t, _) = xs.dims3()?;
        let xs = xs.reshape((b, t, 2, ()))?;
        // SiLU gating: silu(x[..., 0]) * x[..., 1]
        let gate = candle_nn::ops::silu(&xs.i((.., .., 0))?)?;
        let value = xs.i((.., .., 1))?;
        let xs = (gate * value)?;
        Ok(self.linear_out.forward(&xs)?)
    }
}

/// KV cache for depformer attention (like Python's RingKVCache)
/// Stores K and V tensors from previous slices so each slice can attend to all previous slices
#[derive(Debug, Clone)]
pub struct DepFormerKVCache {
    /// Cached K tensors: shape [B, num_heads, slice_idx, d_head]
    k_cache: Option<Tensor>,
    /// Cached V tensors: shape [B, num_heads, slice_idx, d_head]
    v_cache: Option<Tensor>,
    /// Current position in the cache
    position: usize,
}

impl DepFormerKVCache {
    pub fn new() -> Self {
        Self {
            k_cache: None,
            v_cache: None,
            position: 0,
        }
    }

    /// Reset the cache for a new TTS step (like Python's `with depformer.streaming(...)`)
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.k_cache = None;
        self.v_cache = None;
        self.position = 0;
    }

    /// Add new K, V to cache and return complete K, V for attention
    /// Returns (full_k, full_v) where full tensors include all cached + current values
    pub fn complete(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (full_k, full_v) = if let (Some(cached_k), Some(cached_v)) = (&self.k_cache, &self.v_cache) {
            // Concatenate cached with current along the time dimension (dim 2)
            let full_k = Tensor::cat(&[cached_k, k], 2)?;
            let full_v = Tensor::cat(&[cached_v, v], 2)?;
            (full_k, full_v)
        } else {
            // First slice, no cache yet
            (k.clone(), v.clone())
        };

        // Update cache with the new full tensors
        self.k_cache = Some(full_k.clone());
        self.v_cache = Some(full_v.clone());
        self.position += 1;

        Ok((full_k, full_v))
    }
}

/// Shared transformer layer with per-slice gating and weight-per-step schedule
/// Matches Python's StreamingMultiheadAttention with weights_per_step
#[derive(Debug, Clone)]
struct TtsDepFormerLayer {
    /// Per-step QKV projection layers (split from fused weights like Python's _load_hook)
    in_projs: Vec<candle_nn::Linear>,
    /// Per-step output projection layers
    out_projs: Vec<candle_nn::Linear>,
    norm1: moshi::transformer::RmsNorm,
    norm2: moshi::transformer::RmsNorm,
    /// Per-unique-step gating modules
    gatings: Vec<TtsGating>,
    num_heads: usize,
    d_model: usize,
}

impl TtsDepFormerLayer {
    fn new(
        d_model: usize,
        num_heads: usize,
        dim_feedforward: usize,
        num_unique_steps: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Load fused self-attention weights (all steps concatenated)
        // Then split them like Python's _load_hook does
        let in_proj_size = num_unique_steps * 3 * d_model;
        let out_proj_size = num_unique_steps * d_model;
        let fused_in_proj = vb.pp("self_attn").get((in_proj_size, d_model), "in_proj_weight")?;
        let fused_out_proj = vb.pp("self_attn").pp("out_proj").get((out_proj_size, d_model), "weight")?;

        // Split into per-step Linear layers (matching Python's _load_hook behavior)
        let mut in_projs = Vec::with_capacity(num_unique_steps);
        let mut out_projs = Vec::with_capacity(num_unique_steps);
        for step_idx in 0..num_unique_steps {
            // Extract in_proj weights for this step: [3*d_model, d_model]
            let in_offset = step_idx * 3 * d_model;
            let in_weight = fused_in_proj.narrow(0, in_offset, 3 * d_model)?;
            let in_proj = candle_nn::Linear::new(in_weight, None);
            in_projs.push(in_proj);

            // Extract out_proj weights for this step: [d_model, d_model]
            let out_offset = step_idx * d_model;
            let out_weight = fused_out_proj.narrow(0, out_offset, d_model)?;
            let out_proj = candle_nn::Linear::new(out_weight, None);
            out_projs.push(out_proj);
        }

        // Load norms using MaybeQuantizedVarBuilder wrapper
        use moshi::nn::MaybeQuantizedVarBuilder as MQVarBuilder;
        let norm1 = moshi::transformer::RmsNorm::new(d_model, 1e-8, MQVarBuilder::Real(vb.pp("norm1")))?;
        let norm2 = moshi::transformer::RmsNorm::new(d_model, 1e-8, MQVarBuilder::Real(vb.pp("norm2")))?;

        // Load per-unique-step gating
        let mut gatings = Vec::with_capacity(num_unique_steps);
        for step_idx in 0..num_unique_steps {
            let gating = TtsGating::new(d_model, dim_feedforward, vb.pp("gating").pp(step_idx))?;
            gatings.push(gating);
        }

        Ok(Self {
            in_projs,
            out_projs,
            norm1,
            norm2,
            gatings,
            num_heads,
            d_model,
        })
    }

    /// Forward pass for a specific weight step index (from schedule)
    /// Uses KV cache to allow each slice to attend to all previous slices (like Python)
    fn forward(&self, xs: &Tensor, step_idx: usize, kv_cache: &mut DepFormerKVCache) -> Result<Tensor> {
        let residual = xs.clone();

        // Self-attention with pre-norm
        let xs = self.norm1.forward(xs)?;

        // Project Q, K, V using per-step linear (like Python's apply_weights_per_step)
        let qkv = self.in_projs[step_idx].forward(&xs)?;
        let (b, t, _) = qkv.dims3()?;
        let d_head = self.d_model / self.num_heads;

        // Split into Q, K, V
        let q = qkv.narrow(2, 0, self.d_model)?;
        let k = qkv.narrow(2, self.d_model, self.d_model)?;
        let v = qkv.narrow(2, 2 * self.d_model, self.d_model)?;

        // Reshape for multi-head attention: [B, T, H, D] -> [B, H, T, D]
        let q = q.reshape((b, t, self.num_heads, d_head))?.transpose(1, 2)?;
        let k = k.reshape((b, t, self.num_heads, d_head))?.transpose(1, 2)?;
        let v = v.reshape((b, t, self.num_heads, d_head))?.transpose(1, 2)?;

        // Use KV cache to get full K, V (current + all previous slices)
        // This is the key difference - Python's depformer uses streaming KV cache
        let (full_k, full_v) = kv_cache.complete(&k, &v)?;

        // Get dimensions for attention mask
        let t_q = q.dim(2)?; // Query length (1 for streaming)
        let t_kv = full_k.dim(2)?; // KV length (number of slices so far)

        // Scaled dot-product attention (causal)
        let scale = (d_head as f64).sqrt();
        let attn_weights = (q.matmul(&full_k.transpose(2, 3)?)? / scale)?;

        // Apply causal mask - query can only attend to current and previous positions
        // For streaming mode with t_q=1, the current query at position (t_kv-1) can attend to all positions 0 to t_kv-1
        // So we just need to ensure causality: position i can attend to positions 0..=i
        let mut mask_data = vec![0f32; t_q * t_kv];
        for i in 0..t_q {
            let query_pos = t_kv - t_q + i; // Position of this query in the full sequence
            for j in 0..t_kv {
                if j > query_pos {
                    mask_data[i * t_kv + j] = f32::NEG_INFINITY; // mask future positions
                }
            }
        }
        let mask = Tensor::from_vec(mask_data, (1, 1, t_q, t_kv), xs.device())?;
        let dtype = attn_weights.dtype();
        let attn_weights = attn_weights.broadcast_add(&mask.to_dtype(dtype)?)?;

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_out = attn_weights.matmul(&full_v)?;

        // Reshape back: [B, H, T, D] -> [B, T, H*D]
        let attn_out = attn_out.transpose(1, 2)?.reshape((b, t, self.d_model))?;

        // Output projection using per-step linear
        let attn_out = self.out_projs[step_idx].forward(&attn_out)?;

        // Residual connection
        let xs = (residual + attn_out)?;

        // FFN with pre-norm and per-step gating
        let residual = xs.clone();
        let xs = self.norm2.forward(&xs)?;
        let xs = self.gatings[step_idx].forward(&xs)?;

        Ok((residual + xs)?)
    }
}

/// TTS-specific DepFormer with shared transformer and per-slice gating
#[derive(Debug, Clone)]
pub struct TtsDepFormer {
    layers: Vec<TtsDepFormerLayer>,
    /// Text embedding for slice 0 (demux_second_stream=True in Python)
    text_emb: MultiplexedTextEmbedding,
    audio_embs: Vec<TtsLowRankEmbedding>,
    linear_ins: Vec<candle_nn::Linear>,
    linear_outs: Vec<candle_nn::Linear>,
    num_slices: usize,
    /// Maps slice index to unique weight step index
    weight_schedule: Vec<usize>,
}

/// Low-rank embedding for TTS (matches Python ScaledEmbedding with low_rank)
#[derive(Debug, Clone)]
struct TtsLowRankEmbedding {
    embeddings: candle_nn::Embedding,
    low_rank: Option<candle_nn::Linear>,
}

impl TtsLowRankEmbedding {
    fn new(vocab_size: usize, dim: usize, low_rank_dim: Option<usize>, vb: VarBuilder) -> Result<Self> {
        let (embeddings, low_rank) = match low_rank_dim {
            None => {
                let embeddings = candle_nn::embedding(vocab_size, dim, vb)?;
                (embeddings, None)
            }
            Some(lr_dim) => {
                let embeddings = candle_nn::embedding(vocab_size, lr_dim, vb.clone())?;
                let low_rank = candle_nn::linear_no_bias(lr_dim, dim, vb.pp("low_rank"))?;
                (embeddings, Some(low_rank))
            }
        };
        Ok(Self { embeddings, low_rank })
    }

    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        // Candle embedding expects U32 indices
        let ids = ids.to_dtype(DType::U32)?;
        let embs = self.embeddings.forward(&ids)?;
        match &self.low_rank {
            None => Ok(embs),
            Some(lr) => Ok(lr.forward(&embs)?),
        }
    }
}

/// Multiplexed text embedding for TTS (demux_second_stream=True in Python)
///
/// This handles multiplexed tokens from the state machine when second_stream_ahead > 0.
/// Input tokens are: (second + 1) * card + first
/// - first: main stream token (0..card)
/// - second: lookahead stream token (-1..card-1), where -1 means "zero embedding"
///
/// When low_rank is provided:
/// - embeddings: [vocab_size, low_rank_dim]
/// - out1, out2: [low_rank_dim, dim]
#[derive(Debug, Clone)]
struct MultiplexedTextEmbedding {
    embeddings: candle_nn::Embedding,
    out1: candle_nn::Linear,
    out2: candle_nn::Linear,
    vocab_size: usize,
}

impl MultiplexedTextEmbedding {
    fn new(vocab_size: usize, dim: usize, low_rank_dim: Option<usize>, vb: VarBuilder) -> Result<Self> {
        // With low_rank: embedding is [vocab_size, low_rank], out1/out2 are [low_rank, dim]
        // Without low_rank: embedding is [vocab_size, dim], out1/out2 are [dim, dim]
        let emb_dim = low_rank_dim.unwrap_or(dim);

        // Load base embedding
        let embeddings = candle_nn::embedding(vocab_size, emb_dim, vb.clone())?;
        // Load output projections (from emb_dim to dim)
        let out1 = candle_nn::linear_no_bias(emb_dim, dim, vb.pp("out1"))?;
        let out2 = candle_nn::linear_no_bias(emb_dim, dim, vb.pp("out2"))?;
        Ok(Self { embeddings, out1, out2, vocab_size })
    }

    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        // ids can be multiplexed: (second + 1) * card + first
        // De-multiplex: left = ids % card, right = ids / card - 1
        //
        // IMPORTANT: Like Python, we must clamp negative ids (zero_idx = -1) to 0
        // before doing modulo/division, then zero out those outputs at the end.
        let card = self.vocab_size as u32;

        // Convert to Vec for arithmetic (simpler than tensor ops for small tensors)
        let ids_vec: Vec<i64> = ids.to_dtype(DType::I64)?.flatten_all()?.to_vec1()?;
        let original_shape = ids.shape().clone();

        // Track which ids are zero_idx (typically -1) - these get zeroed output
        let mut is_zero_vec = Vec::with_capacity(ids_vec.len());

        // Compute left (first token) and right (second token - 1) values
        let mut left_vec = Vec::with_capacity(ids_vec.len());
        let mut right_vec = Vec::with_capacity(ids_vec.len());

        for &id in &ids_vec {
            // Check for zero_idx (-1) before processing
            // NOTE: Python checks `input == self.zero_idx` where zero_idx=-1
            // Only -1 should be zeroed, NOT other negative values like -2 (ungenerated)
            let is_zero = id == -1;
            is_zero_vec.push(is_zero as u8);

            // Clamp to 0 before modulo/division (like Python's input.clamp(min=0))
            let id_clamped = if id < 0 { 0u32 } else { id as u32 };
            let left_val = id_clamped % card;
            let right_val = (id_clamped / card) as i64 - 1;  // Can be -1 for "zero embedding"
            left_vec.push(left_val);
            right_vec.push(right_val);
        }

        // Create tensors for left and right
        let left = Tensor::from_slice(&left_vec, original_shape.dims(), ids.device())?;
        let right = Tensor::from_slice(&right_vec, original_shape.dims(), ids.device())?;
        let is_zero = Tensor::from_slice(&is_zero_vec, original_shape.dims(), ids.device())?;
        let is_zero = is_zero.gt(0u8)?;

        // Lookup left tokens (always valid, 0..card)
        let left_emb = self.embeddings.forward(&left)?;

        // For right, we need to check if it's < 0 (meaning zero embedding)
        let right_zero = right.lt(0i64)?;

        // Clamp right to min 0 for lookup
        let right_clamped = right.maximum(0i64)?;
        let right_clamped = right_clamped.to_dtype(DType::U32)?;
        let right_emb = self.embeddings.forward(&right_clamped)?;

        // Apply out1 to left, out2 to right
        let left_proj = self.out1.forward(&left_emb)?;
        let right_proj = self.out2.forward(&right_emb)?;

        // Where right < 0, use 0 instead of right_proj
        // Expand right_zero to match embedding dims
        let right_zero = right_zero.unsqueeze(candle::D::Minus1)?;
        let right_zero = right_zero.broadcast_as(right_proj.shape())?;
        let zero = Tensor::zeros_like(&right_proj)?;
        let right_proj = right_zero.where_cond(&zero, &right_proj)?;

        // Sum the projections
        let output = (left_proj + right_proj)?;

        // Zero out outputs where original input was zero_idx (like Python's output[is_zero] = 0)
        let is_zero = is_zero.unsqueeze(candle::D::Minus1)?;
        let is_zero = is_zero.broadcast_as(output.shape())?;
        let output = is_zero.where_cond(&zero, &output)?;

        Ok(output)
    }
}

impl TtsDepFormer {
    /// Load TTS depformer from safetensors with correct weight naming
    pub fn load(
        num_layers: usize,
        num_heads: usize,
        d_model: usize,
        dim_feedforward: usize,
        num_slices: usize,
        main_dim: usize,
        text_vocab_size: usize,
        audio_vocab_size: usize,
        low_rank_dim: Option<usize>,
        weight_schedule: Vec<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Calculate number of unique weight steps
        let num_unique_steps = weight_schedule.iter().max().map(|m| m + 1).unwrap_or(num_slices);
        tracing::info!(num_unique_steps, num_slices, "TtsDepFormer weight schedule");

        // Load transformer layers
        let mut layers = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let layer = TtsDepFormerLayer::new(
                d_model,
                num_heads,
                dim_feedforward,
                num_unique_steps,
                vb.pp("depformer").pp("layers").pp(layer_idx),
            )?;
            layers.push(layer);
        }

        // Load text embedding with demux_second_stream=True
        // This handles multiplexed tokens from the state machine (second_stream_ahead > 0)
        let text_emb = MultiplexedTextEmbedding::new(
            text_vocab_size,
            d_model,
            low_rank_dim,
            vb.pp("depformer_text_emb"),
        )?;

        // Audio embeddings are per-slice (31 total for slices 1-31)
        // Slice 0 uses text_emb, slices 1-31 use depformer_emb.0 through depformer_emb.30
        let mut audio_embs = Vec::with_capacity(num_slices - 1);
        for emb_idx in 0..(num_slices - 1) {
            let emb = TtsLowRankEmbedding::new(
                audio_vocab_size,
                d_model,
                low_rank_dim,
                vb.pp("depformer_emb").pp(emb_idx),
            )?;
            audio_embs.push(emb);
        }

        // Load input projections - follows schedule (num_unique_steps)
        let mut linear_ins = Vec::with_capacity(num_unique_steps);
        for step_idx in 0..num_unique_steps {
            let linear = candle_nn::linear_no_bias(main_dim, d_model, vb.pp("depformer_in").pp(step_idx))?;
            linear_ins.push(linear);
        }

        // Load output projections
        let mut linear_outs = Vec::with_capacity(num_slices);
        for slice_idx in 0..num_slices {
            let linear = candle_nn::linear_no_bias(d_model, audio_vocab_size - 1, vb.pp("linears").pp(slice_idx))?;
            linear_outs.push(linear);
        }

        Ok(Self {
            layers,
            text_emb,
            audio_embs,
            linear_ins,
            linear_outs,
            num_slices,
            weight_schedule,
        })
    }

    /// Forward pass for one slice at a time (streaming mode)
    /// Uses KV caches to allow each slice to attend to all previous slices (like Python's streaming depformer)
    pub fn forward_slice(
        &self,
        slice_idx: usize,
        prev_token: &Tensor,
        main_hidden: &Tensor,
        kv_caches: &mut Vec<DepFormerKVCache>,
    ) -> Result<Tensor> {
        // Map slice index to weight step using schedule (for transformer weights)
        let step_idx = self.weight_schedule.get(slice_idx).copied().unwrap_or(slice_idx);

        // Project main transformer hidden state (uses schedule)
        let mut xs = self.linear_ins[step_idx].forward(main_hidden)?;

        // Add token embedding:
        // - Slice 0: text_emb (receives text token from LM)
        // - Slices 1-31: audio_embs[slice_idx - 1] (receives previous audio token)
        //   slice 1 uses depformer_emb.0, slice 2 uses depformer_emb.1, etc.
        let token_emb = if slice_idx == 0 {
            self.text_emb.forward(prev_token)?
        } else {
            self.audio_embs[slice_idx - 1].forward(prev_token)?
        };
        xs = (xs + token_emb)?;

        // Run through transformer layers with weight step from schedule
        // Each layer uses its own KV cache for streaming attention
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(&xs, step_idx, &mut kv_caches[layer_idx])?;
        }

        // Output projection (per-slice, not scheduled)
        Ok(self.linear_outs[slice_idx].forward(&xs)?)
    }

    /// Create new KV caches for a fresh TTS step (like Python's `with depformer.streaming(...)`)
    pub fn new_kv_caches(&self) -> Vec<DepFormerKVCache> {
        (0..self.layers.len()).map(|_| DepFormerKVCache::new()).collect()
    }

    /// Sample tokens for all slices
    pub fn sample(
        &self,
        main_hidden: &Tensor,
        text_token: u32,
        lp: &mut LogitsProcessor,
        device: &Device,
    ) -> Result<Vec<u32>> {
        let mut tokens = Vec::with_capacity(self.num_slices);
        let mut prev_token = text_token as i64;

        // Create fresh KV caches for this TTS step (like Python's `with depformer.streaming(...)`)
        // This ensures each TTS step starts with empty caches
        let mut kv_caches = self.new_kv_caches();

        for slice_idx in 0..self.num_slices {
            let _step_idx = self.weight_schedule.get(slice_idx).copied().unwrap_or(slice_idx);

            // Create token tensor with I64 dtype (required for embeddings)
            let token_tensor = Tensor::from_slice(&[prev_token], (1, 1), device)?;
            let logits = self.forward_slice(slice_idx, &token_tensor, main_hidden, &mut kv_caches)?;
            // Convert to f32 for sampling
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

            let next_token = lp.sample(&logits)?;
            tokens.push(next_token);
            prev_token = next_token as i64;
        }

        Ok(tokens)
    }
}

/// Word timing information for transcripts
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WordWithTimestamps {
    pub text: String,
    pub start_s: f64,
    pub stop_s: f64,
}

/// Output message types for WebSocket communication
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum OutMsg {
    Text { text: String, start_s: f64, stop_s: f64 },
    Audio { pcm: Vec<f32> },
    OggOpus { data: Vec<u8> },
    Error { message: String },
    Ready,
}

/// Audio encoder supporting multiple output formats
pub enum Encoder {
    OggOpus(kaudio::ogg_opus::Encoder),
    OggOpusMessagePack(kaudio::ogg_opus::Encoder),
    Pcm,
    PcmMessagePack,
}

impl Encoder {
    pub fn new(format: crate::StreamingOutput) -> Result<Self> {
        // Sample rate 24000 from Python moshi/models/loaders.py:85 (sample_rate: 24000)
        match format {
            crate::StreamingOutput::OggOpus => Self::ogg_opus(24000),
            crate::StreamingOutput::OggOpusMessagePack => Self::ogg_opus_message_pack(24000),
            crate::StreamingOutput::Pcm => Ok(Self::pcm()),
            crate::StreamingOutput::PcmMessagePack => Ok(Self::pcm_message_pack()),
        }
    }

    fn ogg_opus(sample_rate: usize) -> Result<Self> {
        Ok(Self::OggOpus(kaudio::ogg_opus::Encoder::new(sample_rate)?))
    }

    fn ogg_opus_message_pack(sample_rate: usize) -> Result<Self> {
        Ok(Self::OggOpusMessagePack(kaudio::ogg_opus::Encoder::new(sample_rate)?))
    }

    fn pcm_message_pack() -> Self {
        Self::PcmMessagePack
    }

    fn pcm() -> Self {
        Self::Pcm
    }

    pub fn header(&self) -> Result<Option<Vec<u8>>> {
        let header = match self {
            Self::OggOpus(oo) => Some(oo.header_data().to_vec()),
            Self::OggOpusMessagePack(oo) => {
                use serde::Serialize;
                let msg = OutMsg::OggOpus { data: oo.header_data().to_vec() };
                let mut buf = vec![];
                msg.serialize(
                    &mut rmp_serde::Serializer::new(&mut buf)
                        .with_human_readable()
                        .with_struct_map(),
                )?;
                Some(buf)
            }
            Self::Pcm => None,
            Self::PcmMessagePack => None,
        };
        Ok(header)
    }

    pub fn encode_word(&self, wwts: WordWithTimestamps) -> Result<Option<Vec<u8>>> {
        if wwts.text.is_empty() {
            return Ok(None);
        }
        let buf = match self {
            Self::Pcm | Self::OggOpus(_) => None,
            Self::OggOpusMessagePack(_) | Self::PcmMessagePack => {
                use serde::Serialize;
                let mut buf = vec![];
                OutMsg::Text { text: wwts.text, start_s: wwts.start_s, stop_s: wwts.stop_s }
                    .serialize(
                        &mut rmp_serde::Serializer::new(&mut buf)
                            .with_human_readable()
                            .with_struct_map(),
                    )?;
                Some(buf)
            }
        };
        Ok(buf)
    }

    pub fn encode(&mut self, pcm: Vec<f32>) -> Result<Vec<u8>> {
        use serde::Serialize;
        let buf = match self {
            Self::OggOpus(oo) => oo.encode_page(&pcm)?,
            Self::OggOpusMessagePack(oo) => {
                let data = oo.encode_page(&pcm)?;
                let mut buf = vec![];
                OutMsg::OggOpus { data }.serialize(
                    &mut rmp_serde::Serializer::new(&mut buf)
                        .with_human_readable()
                        .with_struct_map(),
                )?;
                buf
            }
            Self::PcmMessagePack => {
                let mut buf = vec![];
                OutMsg::Audio { pcm }.serialize(
                    &mut rmp_serde::Serializer::new(&mut buf)
                        .with_human_readable()
                        .with_struct_map(),
                )?;
                buf
            }
            Self::Pcm => {
                use byteorder::ByteOrder;
                let mut buf = vec![0u8; std::mem::size_of_val(pcm.as_slice())];
                byteorder::LittleEndian::write_f32_into(&pcm, &mut buf);
                buf
            }
        };
        Ok(buf)
    }

    pub fn encode_msg(&mut self, msg: OutMsg) -> Result<Option<Vec<u8>>> {
        use serde::Serialize;
        let buf = match self {
            Self::OggOpus(_) | Self::Pcm => None,
            Self::OggOpusMessagePack(_) | Self::PcmMessagePack => {
                let mut buf = vec![];
                msg.serialize(
                    &mut rmp_serde::Serializer::new(&mut buf)
                        .with_human_readable()
                        .with_struct_map(),
                )?;
                Some(buf)
            }
        };
        Ok(buf)
    }
}

/// Model config loaded from config.json
#[allow(dead_code)]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct DsmModelConfig {
    pub dim: usize,
    pub text_card: usize,
    pub n_q: usize,
    pub dep_q: usize,
    pub card: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub hidden_scale: f64,
    pub delays: Vec<usize>,
    pub depformer_dim: usize,
    pub depformer_num_heads: usize,
    pub depformer_num_layers: usize,
    #[serde(default)]
    pub depformer_dim_feedforward: Option<usize>,
    #[serde(default)]
    pub depformer_weights_per_step: bool,
    #[serde(default)]
    pub depformer_weights_per_step_schedule: Option<Vec<usize>>,
    #[serde(default)]
    pub depformer_low_rank_embeddings: Option<usize>,
    #[serde(default)]
    pub conditioners: Option<moshi::conditioner::Config>,
    #[serde(default)]
    pub cross_attention: bool,
    #[serde(default)]
    pub tts_config: Option<TtsSpecificConfig>,
    #[serde(default)]
    pub tokenizer_name: Option<String>,
    #[serde(default)]
    pub mimi_name: Option<String>,
    #[serde(default)]
    pub moshi_name: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct TtsSpecificConfig {
    #[serde(default)]
    pub audio_delay: f64,
    #[serde(default)]
    pub second_stream_ahead: usize,
}

/// Voice embedding cache
#[derive(Debug, Clone)]
pub struct VoiceEmbedding {
    /// Speaker wavs tensor [1, num_speakers * T, dim]
    pub tensor: Tensor,
    /// Mask tensor [1, num_speakers * T]
    pub mask: Tensor,
}

impl VoiceEmbedding {
    /// Load voice embedding from a .safetensors file
    /// Format: speaker_wavs tensor of shape [1, dim, T]
    pub fn load(path: &Path, device: &Device, dtype: DType, max_speakers: usize) -> Result<Self> {
        let tensors = candle::safetensors::load(path, device)?;
        let emb = tensors.get("speaker_wavs")
            .context("missing speaker_wavs tensor")?;

        // emb shape: [1, dim, T] -> need to rearrange for multi-speaker support
        // Following Python: voice_tensor[batch, speaker_idx, :, :] = emb.transpose(1, 2)
        // Then view as [1, -1, dim]
        let emb = emb.to_dtype(dtype)?;
        let (_, dim, t) = emb.dims3()?;

        // Create multi-speaker tensor [1, max_speakers, T, dim]
        let voice_tensor = Tensor::zeros((1, max_speakers, t, dim), dtype, device)?;
        // Transpose emb: [1, dim, T] -> [1, T, dim]
        let emb_t = emb.transpose(1, 2)?;
        // Set first speaker
        let voice_tensor = voice_tensor.slice_assign(&[0..1, 0..1, 0..t, 0..dim], &emb_t.unsqueeze(1)?)?;
        // Also set second speaker slot (Python does [file, file] for voices)
        let voice_tensor = voice_tensor.slice_assign(&[0..1, 1..2, 0..t, 0..dim], &emb_t.unsqueeze(1)?)?;

        // Reshape to [1, max_speakers * T, dim]
        let tensor = voice_tensor.reshape((1, max_speakers * t, dim))?;

        // Create mask [1, max_speakers, T] -> [1, max_speakers * T]
        let mut mask_data = vec![0u8; max_speakers * t];
        // Mark first two speakers as valid
        for i in 0..(2 * t) {
            mask_data[i] = 1;
        }
        let mask = Tensor::from_slice(&mask_data, (1, max_speakers * t), device)?;

        Ok(Self { tensor, mask })
    }
}

/// Configuration for the TTS model
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TtsModelConfig {
    pub batch_size: usize,
    pub n_q: usize,
    pub dep_q: usize,
    pub cfg_coef: f64,
    pub cfg_is_no_text: bool,
    pub temp: f64,
    pub temp_text: f64,
    pub top_k: usize,
    pub top_k_text: usize,
    pub padding_between: usize,
    pub max_padding: i32,
    pub initial_padding: i32,
    pub final_padding: usize,
    pub padding_bonus: f64,
    pub delay_steps: usize,
    pub second_stream_ahead: usize,
    pub delays: Vec<usize>,
    pub frame_rate: f64,
}

impl Default for TtsModelConfig {
    fn default() -> Self {
        Self {
            batch_size: 2,                // Python moshi-server/tts.py (batch_size usage)
            n_q: 24,                      // Python moshi-server/tts.py:63 (n_q: int = 24)
            dep_q: 24,                    // Python moshi/models/loaders.py:95 (dep_q from config.json)
            cfg_coef: 2.0,                // Python moshi-server/tts.py:73 (cfg_coef: float = 2.)
            cfg_is_no_text: true,         // Python moshi-server/tts.py:114 (cfg_is_no_text = True)
            temp: 0.6,                    // Python moshi/models/tts.py:381 (temp: float = 0.6)
            temp_text: 0.6,               // Python moshi/models/tts.py:478 (temp_text=self.temp)
            top_k: 250,                   // Python moshi/models/lm.py:557 (top_k: int = 250)
            top_k_text: 25,               // Python moshi/models/lm.py:558 (top_k_text: int = 25)
            padding_between: 1,           // Python moshi-server/tts.py:78 (padding_between: int = 1)
            max_padding: 8,               // Python moshi/models/tts.py:391 (max_padding: int = 8)
            initial_padding: 2,           // Python moshi/models/tts.py:390 (initial_padding: int = 2)
            final_padding: 4,             // Python moshi/models/tts.py:383 (final_padding: int = 4)
            padding_bonus: 0.0,           // Python moshi/models/tts.py:386 (padding_bonus: float = 0.)
            delay_steps: 16,              // Python moshi/models/tts.py:404 (audio_delay * frame_rate)
            second_stream_ahead: 2,       // Python config.json tts_config.second_stream_ahead
            delays: vec![0; 33],          // Python moshi/models/loaders.py (delays from config.json)
            frame_rate: 12.5,             // Python moshi/models/loaders.py:84 (frame_rate: 12.5)
        }
    }
}

/// Streaming generation state
#[allow(dead_code)]
pub struct GenState {
    /// Cache for tokens with delay handling [batch, num_codebooks, cache_size]
    pub cache: Tensor,
    /// Initial token tensor
    pub initial: Tensor,
    /// Current offset
    pub offset: usize,
    /// Delays on device
    pub delays_cuda: Tensor,
    /// Max delay
    pub max_delay: usize,
    /// State machine state
    pub machine_state: State,
    /// Text token logits processor
    pub text_lp: LogitsProcessor,
    /// Audio token logits processor
    pub audio_lp: LogitsProcessor,
}

impl GenState {
    pub fn new(
        num_codebooks: usize,
        delays: &[usize],
        entries: Vec<Entry>,
        machine: &StateMachine,
        device: &Device,
        temp: f64,
        temp_text: f64,
        top_k: usize,
        top_k_text: usize,
        seed: u64,
    ) -> Result<Self> {
        let max_delay = delays.iter().copied().max().unwrap_or(0);
        let cache_size = max_delay + 2;

        // Initialize cache with ungenerated token ID (-2)
        let ungenerated: i64 = -2;
        let cache = Tensor::full(
            ungenerated,
            (1, num_codebooks, cache_size),
            device,
        )?.to_dtype(DType::I64)?;

        // Initial token tensor (zeros for padding)
        let initial = Tensor::zeros((1, num_codebooks, 1), DType::I64, device)?;

        let delays_cuda = Tensor::from_slice(
            &delays.iter().map(|&d| d as i64).collect::<Vec<_>>(),
            delays.len(),
            device,
        )?;

        let machine_state = machine.new_state(entries);

        // Create logits processors with proper TopK sampling
        // Python lm.py:557-558: top_k=250 (audio), top_k_text=25 (text)
        // Use ArgMax (greedy) when temperature is 0 to avoid division by zero
        use candle_transformers::generation::Sampling;
        let text_sampling = if temp_text == 0.0 {
            Sampling::ArgMax
        } else {
            Sampling::TopK { k: top_k_text, temperature: temp_text }
        };
        let audio_sampling = if temp == 0.0 {
            Sampling::ArgMax
        } else {
            Sampling::TopK { k: top_k, temperature: temp }
        };
        let text_lp = LogitsProcessor::from_sampling(seed, text_sampling);
        let audio_lp = LogitsProcessor::from_sampling(seed, audio_sampling);

        Ok(Self {
            cache,
            initial,
            offset: 0,
            delays_cuda,
            max_delay,
            machine_state,
            text_lp,
            audio_lp,
        })
    }
}

/// Native Rust TTS Model using DSM architecture
#[allow(dead_code)]
pub struct Model {
    /// LM model for text-to-audio generation (needs interior mutability for forward/reset_state)
    lm_model: std::sync::Mutex<moshi::lm::LmModel>,
    /// Multiplexed text embedding for handling second_stream_ahead demuxing
    multiplexed_text_emb: MultiplexedTextEmbedding,
    /// TTS-specific DepFormer with shared layers and per-slice gating
    tts_depformer: TtsDepFormer,
    /// Mimi audio codec for decoding (needs interior mutability for decode)
    mimi: std::sync::Mutex<moshi::mimi::Mimi>,
    /// State machine for TTS
    machine: StateMachine,
    /// Token IDs
    token_ids: TokenIds,
    /// Voice embeddings cache
    voices: HashMap<String, VoiceEmbedding>,
    /// Pre-computed cross-attention sources for each voice (for speaker conditioning)
    cross_attention_cache: HashMap<String, Tensor>,
    /// Default voice name
    default_voice: String,
    /// Voice folder path
    voice_folder: PathBuf,
    /// Voice suffix for finding embeddings
    voice_suffix: String,
    /// Device for computation
    device: Device,
    /// Data type
    dtype: DType,
    /// Config
    config: TtsModelConfig,
    /// Text tokenizer
    text_tokenizer: sentencepiece::SentencePieceProcessor,
    /// Max speakers for voice conditioning
    max_speakers: usize,
    /// DSM config for TTS depformer loading
    dsm_config: DsmModelConfig,
    /// Whether CFG distillation is used (model has 'cfg' conditioner)
    uses_cfg_distillation: bool,
    /// CFG condition value for distillation (e.g., "2.0")
    cfg_condition_value: Option<String>,
    /// Pre-computed condition tensor for sum fusion (cfg + control LUT embeddings)
    sum_condition: Option<Tensor>,
    /// Mutex for exclusive access during generation (async)
    pub(crate) mutex: tokio::sync::Mutex<()>,
}

impl Model {
    /// Create a new TTS model from configuration
    pub fn new(
        tts_cfg: &crate::TtsConfig,
        _config: &crate::Config,
        dev: &Device,
    ) -> Result<Self> {
        tracing::info!("Initializing native Rust TTS model");

        // Determine dtype
        let dtype = crate::utils::model_dtype(tts_cfg.dtype_override.as_deref(), dev)?;
        tracing::info!(?dtype, "Using dtype");

        // Load model config from HuggingFace
        let api = hf_hub::api::sync::ApiBuilder::from_env().build()?.model(tts_cfg.hf_repo.clone());

        // Get config.json
        let config_path = match &tts_cfg.config_path {
            Some(p) => PathBuf::from(crate::utils::resolve_or_download(p)?),
            None => PathBuf::from(api.get("config.json")?),
        };
        tracing::info!(?config_path, "Loading model config");
        let config_content = std::fs::read_to_string(&config_path)?;
        let dsm_config: DsmModelConfig = serde_json::from_str(&config_content)?;

        // Get model file paths
        let moshi_name = dsm_config.moshi_name.as_deref().unwrap_or("dsm_tts_1e68beda@240.safetensors");
        let mimi_name = dsm_config.mimi_name.as_deref().unwrap_or("tokenizer-e351c8d8-checkpoint125.safetensors");

        let moshi_path = match &tts_cfg.moshi_weight {
            Some(p) => PathBuf::from(crate::utils::resolve_or_download(p)?),
            None => PathBuf::from(api.get(moshi_name)?),
        };
        let mimi_path = match &tts_cfg.mimi_weight {
            Some(p) => PathBuf::from(crate::utils::resolve_or_download(p)?),
            None => PathBuf::from(api.get(mimi_name)?),
        };

        tracing::info!(?moshi_path, ?mimi_path, "Loading model weights");

        // Calculate delay_steps from audio_delay
        // Python moshi/models/tts.py:404: delay_steps = int(checkpoint_info.tts_config['audio_delay'] * mimi.frame_rate)
        let tts_specific = dsm_config.tts_config.as_ref().cloned().unwrap_or_default();
        let frame_rate = 12.5; // Python moshi/models/loaders.py:84 (frame_rate: 12.5)
        let delay_steps = (tts_specific.audio_delay * frame_rate) as usize;
        let second_stream_ahead = tts_specific.second_stream_ahead;

        tracing::info!(delay_steps, second_stream_ahead, "TTS config from model");

        // Build LM config matching the DSM model
        let lm_cfg = Self::build_lm_config(&dsm_config)?;
        tracing::info!(?lm_cfg, "Built LM config");

        // Load LM model
        let lm_model = moshi::lm::load_lm_model(lm_cfg, &moshi_path, dtype, dev)?;
        tracing::info!("LM model loaded");

        // Create VarBuilder for loading additional weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&moshi_path], dtype, dev)?
        };

        // Load multiplexed text embedding for handling second_stream_ahead demuxing
        // This replaces the LM's standard text_emb when second_stream_ahead > 0
        // Note: main text_emb does NOT use low_rank (only depformer_text_emb uses it)
        let multiplexed_text_emb = MultiplexedTextEmbedding::new(
            dsm_config.text_card + 1,
            dsm_config.dim,
            None,  // Main text_emb doesn't use low_rank
            vb.pp("text_emb"),
        )?;
        tracing::info!("Multiplexed text embedding loaded");

        // Load TTS-specific depformer with correct weight naming
        let depformer_dim_feedforward = dsm_config.depformer_dim_feedforward
            .unwrap_or((dsm_config.depformer_dim as f64 * dsm_config.hidden_scale) as usize);
        // Get weight schedule (default to identity: 0, 1, 2, ... dep_q-1)
        let weight_schedule = dsm_config.depformer_weights_per_step_schedule
            .clone()
            .unwrap_or_else(|| (0..dsm_config.dep_q).collect());

        let tts_depformer = TtsDepFormer::load(
            dsm_config.depformer_num_layers,
            dsm_config.depformer_num_heads,
            dsm_config.depformer_dim,
            depformer_dim_feedforward,
            dsm_config.dep_q,
            dsm_config.dim,
            dsm_config.text_card + 1,
            dsm_config.card + 1,
            dsm_config.depformer_low_rank_embeddings,
            weight_schedule,
            vb,
        )?;
        tracing::info!("TTS depformer loaded with {} layers, {} slices",
            dsm_config.depformer_num_layers, dsm_config.dep_q);

        // Load Mimi codec
        let mimi = moshi::mimi::load(mimi_path.to_str().unwrap(), Some(tts_cfg.n_q), dev)?;
        tracing::info!(n_q = tts_cfg.n_q, "Mimi codec loaded");

        // Load text tokenizer
        let text_tokenizer = sentencepiece::SentencePieceProcessor::open(&tts_cfg.text_tokenizer_file)?;
        tracing::info!("Text tokenizer loaded");

        // Build token IDs
        let token_ids = TokenIds::new(dsm_config.text_card + 1);

        // Build state machine
        let machine = StateMachine::new(
            token_ids,
            second_stream_ahead,
            tts_cfg.max_padding as i32,
            tts_cfg.initial_padding as i32,
        );

        // Detect CFG distillation: model has 'cfg' in conditioners with possible_values
        // Python: if tts_model.valid_cfg_conditionings: cfg_coef = 1.0, cfg_is_no_text = False
        let uses_cfg_distillation = dsm_config.conditioners.as_ref().map_or(false, |c| {
            c.get("cfg").is_some()
        });

        // If CFG distillation is used, override cfg_coef to 1.0 and store the original as condition
        let (effective_cfg_coef, effective_cfg_is_no_text, cfg_condition_value) = if uses_cfg_distillation {
            tracing::info!(
                original_cfg_coef = tts_cfg.cfg_coef,
                "Model uses CFG distillation, setting cfg_coef=1.0 and passing cfg as condition"
            );
            (1.0, false, Some(format!("{:.1}", tts_cfg.cfg_coef)))
        } else {
            (tts_cfg.cfg_coef, tts_cfg.cfg_is_no_text, None)
        };

        // Prepare sum conditions (cfg + control LUT embeddings) if the model has conditioners
        let sum_condition = if let Some(condition_provider) = lm_model.condition_provider() {
            let mut cond_tensors = moshi::conditioner::ConditionTensors::new();

            // Add cfg condition if using CFG distillation
            if let Some(ref cfg_val) = cfg_condition_value {
                match condition_provider.condition_lut("cfg", cfg_val) {
                    Ok(moshi::conditioner::Condition::AddToInput(t)) => {
                        cond_tensors.add_sum(t)?;
                        tracing::info!(cfg_value = %cfg_val, "Added cfg condition");
                    }
                    Ok(_) => tracing::warn!("cfg condition is not AddToInput type"),
                    Err(e) => tracing::warn!("Failed to compute cfg condition: {}", e),
                }
            }

            // Add control condition (always "ok")
            match condition_provider.condition_lut("control", "ok") {
                Ok(moshi::conditioner::Condition::AddToInput(t)) => {
                    cond_tensors.add_sum(t)?;
                    tracing::info!("Added control condition");
                }
                Ok(_) => tracing::warn!("control condition is not AddToInput type"),
                Err(e) => tracing::warn!("Failed to compute control condition: {}", e),
            }

            cond_tensors.sum
        } else {
            None
        };

        // Build model config
        // dep_q = number of depformer output streams (32)
        // n_q = number of Mimi codebooks used for decoding (24)
        let model_config = TtsModelConfig {
            batch_size: tts_cfg.batch_size,
            n_q: tts_cfg.n_q,
            dep_q: dsm_config.dep_q,  // Don't cap - depformer outputs 32 tokens
            cfg_coef: effective_cfg_coef,
            cfg_is_no_text: effective_cfg_is_no_text,
            temp: tts_cfg.temp,
            temp_text: tts_cfg.temp,  // Python uses same temp for both audio and text
            top_k: tts_cfg.top_k,
            top_k_text: tts_cfg.top_k_text,
            padding_between: tts_cfg.padding_between,
            max_padding: tts_cfg.max_padding as i32,
            initial_padding: tts_cfg.initial_padding as i32,
            final_padding: tts_cfg.final_padding,
            padding_bonus: tts_cfg.padding_bonus,
            delay_steps,
            second_stream_ahead,
            delays: dsm_config.delays.clone(),
            frame_rate,
        };

        // Resolve voice folder
        let voice_folder = PathBuf::from(&tts_cfg.voice_folder);

        // Build voice suffix from model_id
        // Python moshi/models/tts.py:397: voice_suffix = f".{model_id['sig']}@{model_id['epoch']}.safetensors"
        let voice_suffix = ".1e68beda@240.safetensors".to_string();

        // Load voices
        let mut voices = HashMap::new();
        let max_speakers = 5; // Python moshi/models/tts.py:377 (max_speakers: int = 5)

        if voice_folder.exists() {
            Self::load_voices_from_folder(&voice_folder, &voice_suffix, &mut voices, dev, dtype, max_speakers)?;
        }

        tracing::info!(num_voices = voices.len(), "Loaded voice embeddings");

        // Pre-compute cross-attention sources for each voice
        // Python: self._get_cross_attention_source([attributes])
        // This projects voice embeddings to model dimension for cross-attention
        let mut cross_attention_cache = HashMap::new();
        if let Some(condition_provider) = lm_model.condition_provider() {
            tracing::info!("Computing cross-attention cache for voices");
            for (voice_name, voice_emb) in &voices {
                match condition_provider.condition_tensor("speaker_wavs", &voice_emb.tensor, &voice_emb.mask) {
                    Ok(moshi::conditioner::Condition::CrossAttention(cross_src)) => {
                        // Add sinusoidal positional embeddings to cross-attention source
                        // Python ConditionFuser.get_cross() does this when cross_attention_pos_emb=True
                        // Config: cross_attention_pos_emb=true, cross_attention_pos_emb_scale=1, max_period=10000
                        let cross_src_with_pos = add_cross_attention_pos_emb(&cross_src, 1.0, 10000.0)
                            .context("Failed to add positional embeddings to cross-attention source")?;
                        cross_attention_cache.insert(voice_name.clone(), cross_src_with_pos);
                        tracing::debug!(voice = %voice_name, "Computed cross-attention source with positional embeddings");
                    }
                    Ok(_) => tracing::warn!(voice = %voice_name, "speaker_wavs condition is not CrossAttention type"),
                    Err(e) => tracing::warn!(voice = %voice_name, error = ?e, "Failed to compute cross-attention source"),
                }
            }
            tracing::info!(num_cached = cross_attention_cache.len(), "Cross-attention cache ready");
        }

        Ok(Self {
            lm_model: std::sync::Mutex::new(lm_model),
            multiplexed_text_emb,
            tts_depformer,
            mimi: std::sync::Mutex::new(mimi),
            machine,
            token_ids,
            voices,
            cross_attention_cache,
            default_voice: tts_cfg.default_voice.clone(),
            voice_folder,
            voice_suffix,
            device: dev.clone(),
            dtype,
            config: model_config,
            text_tokenizer,
            max_speakers,
            dsm_config,
            uses_cfg_distillation,
            cfg_condition_value,
            sum_condition,
            mutex: tokio::sync::Mutex::new(()),
        })
    }

    fn build_lm_config(dsm: &DsmModelConfig) -> Result<moshi::lm::Config> {
        use moshi::transformer;
        use moshi::NormType;

        let dim_feedforward = (dsm.dim as f64 * dsm.hidden_scale) as usize;
        let depformer_dim_feedforward = dsm.depformer_dim_feedforward
            .unwrap_or((dsm.depformer_dim as f64 * dsm.hidden_scale) as usize);

        // Main transformer config
        let transformer_cfg = transformer::Config {
            d_model: dsm.dim,
            num_heads: dsm.num_heads,
            num_layers: dsm.num_layers,
            dim_feedforward,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 500,
            max_period: 10000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: if dsm.cross_attention {
                Some((
                    transformer::CrossAttentionGating::Normal,
                    NormType::LayerNorm,
                    None,
                ))
            } else {
                None
            },
            gating: Some(candle_nn::Activation::Silu),
            norm: NormType::RmsNorm,
            positional_embedding: transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
            shared_cross_attn: false,
        };

        // Depformer config
        let depformer_cfg = transformer::Config {
            d_model: dsm.depformer_dim,
            num_heads: dsm.depformer_num_heads,
            num_layers: dsm.depformer_num_layers,
            dim_feedforward: depformer_dim_feedforward,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: dsm.dep_q,
            max_period: 10000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Silu),
            norm: NormType::RmsNorm,
            positional_embedding: transformer::PositionalEmbedding::None,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
            shared_cross_attn: false,
        };

        // NOTE: We set depformer: None here because the TTS model uses a different
        // architecture with shared transformer layers and per-slice gating.
        // The TtsDepFormer will be loaded separately with proper weight naming.
        let _depformer_cfg = moshi::lm::DepFormerConfig {
            transformer: depformer_cfg,
            num_slices: dsm.dep_q,
            low_rank_embeddings: dsm.depformer_low_rank_embeddings,
        };

        Ok(moshi::lm::Config {
            transformer: transformer_cfg,
            depformer: None, // Skip built-in depformer, use TtsDepFormer instead
            audio_vocab_size: dsm.card + 1, // +1 for padding token
            text_in_vocab_size: dsm.text_card + 1,
            text_out_vocab_size: dsm.text_card,
            audio_codebooks: dsm.n_q,
            conditioners: dsm.conditioners.clone(),
            extra_heads: None,
        })
    }

    fn load_voices_from_folder(
        folder: &Path,
        suffix: &str,
        voices: &mut HashMap<String, VoiceEmbedding>,
        device: &Device,
        dtype: DType,
        max_speakers: usize,
    ) -> Result<()> {
        for entry in walkdir::WalkDir::new(folder)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.ends_with(suffix) {
                    let relative = path.strip_prefix(folder).unwrap_or(path);
                    let voice_name = relative
                        .with_file_name(name.strip_suffix(suffix).unwrap_or(name))
                        .to_string_lossy()
                        .replace('\\', "/");

                    match VoiceEmbedding::load(path, device, dtype, max_speakers) {
                        Ok(emb) => {
                            tracing::debug!(voice = %voice_name, "Loaded voice embedding");
                            voices.insert(voice_name, emb);
                        }
                        Err(e) => {
                            tracing::warn!(voice = %voice_name, error = ?e, "Failed to load voice");
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Tokenize text into entries for the state machine
    /// Implements the same logic as Python script_to_entries in moshi/models/tts.py
    fn tokenize_text(&self, text: &[String]) -> Result<Vec<Entry>> {
        let mut entries = Vec::new();

        // Multi-speaker handling - matches Python tts.py:447
        // Python: multi_speaker = 'speaker_wavs' in self.lm.condition_provider.conditioners
        let multi_speaker = !self.cross_attention_cache.is_empty();
        let speaker_tokens = [self.token_ids.main, self.token_ids.other]; // [1, 2]
        let mut last_speaker: Option<usize> = None;

        for (line_idx, line) in text.iter().enumerate() {
            let line = line.replace('\u{2019}', "'").replace(':', " ").replace('(', "").replace(')', "");
            tracing::debug!(line = %line, "tokenize_text processing line");

            // Track first word of each line (like Python's first_content)
            let mut first_content = true;

            for word in line.split_whitespace() {
                let pieces = self.text_tokenizer.encode(word)?;
                tracing::debug!(word = %word, num_pieces = pieces.len(), "tokenize_text encoded word");

                // Use raw token IDs from SentencePiece - Python does the same
                let mut tokens: Vec<u32> = pieces
                    .iter()
                    .map(|p| p.id as u32)
                    .collect();

                // Insert speaker token at start of first word if multi-speaker
                // Matches Python script_to_entries lines 283-286:
                //   if first_content:
                //       speaker = idx % len(speaker_tokens)
                //       if multi_speaker and last_speaker != speaker:
                //           last_speaker = speaker
                //           tokens.insert(0, speaker_tokens[speaker])
                //       first_content = False
                if first_content {
                    let speaker = line_idx % speaker_tokens.len();
                    if multi_speaker && last_speaker != Some(speaker) {
                        last_speaker = Some(speaker);
                        tokens.insert(0, speaker_tokens[speaker]);
                        tracing::debug!(word = %word, speaker_token = speaker_tokens[speaker], "Inserted speaker token");
                    }
                    first_content = false;
                }

                tracing::debug!(word = %word, tokens = ?tokens, "tokenize_text tokens");

                let padding = if self.config.padding_between > 0 {
                    (self.config.padding_between + tokens.len()).saturating_sub(1)
                } else {
                    0
                };

                entries.push(Entry::new(tokens, word.to_string(), padding));
            }
        }

        Ok(entries)
    }

    /// Get voice embedding for a voice name
    fn get_voice(&self, voice: Option<&str>) -> Option<&VoiceEmbedding> {
        let voice_name = voice.unwrap_or(&self.default_voice);
        self.voices.get(voice_name)
    }

    /// Run TTS generation for a query (non-streaming)
    pub fn run(
        &self,
        query: &crate::TtsQuery,
    ) -> Result<(Vec<u8>, Vec<WordWithTimestamps>)> {
        // Tokenize text
        let entries = self.tokenize_text(&query.text)?;
        if entries.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        tracing::info!(num_entries = entries.len(), "Tokenized text into entries");

        // Get voice embedding
        let voice = self.get_voice(query.voice.as_deref());
        if voice.is_none() && !self.voices.is_empty() {
            tracing::warn!(
                voice = ?query.voice,
                default = %self.default_voice,
                "Voice not found, using default"
            );
        }

        tracing::debug!(
            n_q = self.config.n_q,
            dep_q = self.config.dep_q,
            delays_len = self.config.delays.len(),
            "TTS config values"
        );

        let num_codebooks = self.config.delays.len();
        let mut gen_state = GenState::new(
            num_codebooks,
            &self.config.delays,
            entries,
            &self.machine,
            &self.device,
            query.temperature,
            self.config.temp_text,
            query.top_k,
            self.config.top_k_text,
            query.seed,
        )?;

        // Lock models for generation
        let mut lm_model = self.lm_model.lock().map_err(|e| anyhow::anyhow!("lm_model lock poisoned: {}", e))?;
        let mut mimi = self.mimi.lock().map_err(|e| anyhow::anyhow!("mimi lock poisoned: {}", e))?;

        // Reset LM model state
        lm_model.reset_state();
        // Reset Mimi decoder state so decode_step maintains continuity across frames
        mimi.reset_state();

        // Calculate "missing" codebooks - audio streams beyond what depformer handles
        // num_codebooks = 33 (text + 32 audio), dep_q = 32 (depformer outputs)
        // missing = (num_codebooks - 1) - dep_q = 32 - 32 = 0 for TTS
        let zero_token = self.token_ids.zero;
        let num_codebooks = self.config.delays.len();
        let num_audio_codebooks = num_codebooks.saturating_sub(1);
        let missing = num_audio_codebooks.saturating_sub(self.config.dep_q);
        let input_tokens = if missing > 0 {
            Tensor::full(
                zero_token as i64,
                (1, missing, 1),
                &self.device,
            )?.to_dtype(DType::I64)?
        } else {
            // No extra codebooks to handle - create dummy empty tensor
            Tensor::zeros((1, 0, 1), DType::I64, &self.device)?
        };

        // Get cross-attention source for voice conditioning
        // Python pre-computes this via _get_cross_attention_source() and stores in cross_attention_cache
        let voice_name = query.voice.as_deref().unwrap_or(&self.default_voice);
        let ca_src = self.cross_attention_cache.get(voice_name).cloned();
        if ca_src.is_some() {
            tracing::info!(voice = %voice_name, "Using cross-attention for voice conditioning");
        } else if !self.cross_attention_cache.is_empty() {
            tracing::warn!(voice = %voice_name, "Voice not in cross-attention cache, using without voice conditioning");
        }

        // NOTE: No per-request warmup needed. Python's warmup is a GLOBAL server-level init
        // that runs once during __post_init__ for CUDA graph warmup. When a client resets,
        // Python resets the LM offset back to 0 via reset_streaming(). Since we create fresh
        // LM state per-request, we start at offset 0 directly.

        // Generation loop
        // Python moshi/models/tts.py:385 (max_gen_length: int = 30000)
        let max_gen_length = query.max_seq_len.unwrap_or(30000);
        let mut all_pcm = Vec::new();
        let mut timestamps = Vec::new();
        // Generation starts at offset 0, matching Python's client.offset = 0
        for offset in 0..max_gen_length {
            // Check if we're done - Python: real_end = end_step + delay_steps + final_padding + max_delay
            if let Some(end_step) = gen_state.machine_state.end_step {
                let real_end = end_step + self.config.delay_steps + self.config.final_padding + gen_state.max_delay;
                if offset >= real_end {
                    tracing::debug!(offset, end_step, real_end, "Generation complete");
                    break;
                }
            }

            // Run one step of generation
            let frame = self.step_inner(&mut lm_model, &mut gen_state, &input_tokens, offset, ca_src.as_ref())?;

            if let Some(frame) = frame {

                // Decode audio if past delay
                let real_offset = offset as i64 - gen_state.max_delay as i64;
                if real_offset >= self.config.delay_steps as i64 {
                    // Only take first n_q audio layers for Mimi (not all dep_q slices)
                    let n_q = self.config.n_q;
                    let audio_frame = frame.i((0..1, 1..(1 + n_q), 0..1))?;

                    // Handle negative token values to match Python's behavior:
                    // Python's F.embedding wraps negative indices, so -1 maps to the last embedding (card-1).
                    // We replicate this by converting -1 -> card-1 (2047) before decoding.
                    // This is important for per-codebook delay masking where zero_token=-1.
                    let card = self.token_ids.card as i64;
                    let audio_frame_vec: Vec<i64> = audio_frame.flatten_all()?.to_vec1()?;
                    let audio_frame_wrapped: Vec<i64> = audio_frame_vec.iter().map(|&t| {
                        // Clamp to [0, card-1] range, matching Python's clamp_(min=0)
                        t.clamp(0, card - 1)
                    }).collect();

                    let audio_frame = Tensor::from_slice(
                        &audio_frame_wrapped,
                        (1, n_q, 1),
                        &self.device
                    )?;
                    // Convert to u32 for mimi decode
                    let audio_frame = audio_frame.to_dtype(DType::U32)?;

                    let pcm = mimi.decode_step(&audio_frame.into(), &().into())?;
                    if let Some(pcm) = pcm.as_option() {
                        let pcm: Vec<f32> = pcm.i((0, 0))?.to_vec1()?;
                        all_pcm.extend(pcm);
                    }
                }
            }

            // Update transcript timestamps
            for (word, step) in &gen_state.machine_state.transcript {
                let start_s = *step as f64 / self.config.frame_rate;
                let stop_s = start_s + 0.1; // Approximate duration
                if timestamps.iter().all(|t: &WordWithTimestamps| t.text != *word || (t.start_s - start_s).abs() > 0.01) {
                    timestamps.push(WordWithTimestamps {
                        text: word.clone(),
                        start_s,
                        stop_s,
                    });
                }
            }

            gen_state.offset = offset + 1;
        }

        // Encode final audio
        let mut buffer = Vec::new();
        kaudio::wav::write_pcm_as_wav(&mut buffer, &all_pcm, 24000, 1)?;

        Ok((buffer, timestamps))
    }

    /// Run one step of generation (internal method with borrowed lm_model)
    fn step_inner(
        &self,
        lm_model: &mut moshi::lm::LmModel,
        state: &mut GenState,
        input_tokens: &Tensor,
        offset: usize,
        ca_src: Option<&Tensor>,  // Cross-attention source for voice conditioning
    ) -> Result<Option<Tensor>> {
        use moshi::StreamMask;

        let ct = state.cache.dim(2)?;
        let num_codebooks = self.config.delays.len();
        let dep_q = self.config.dep_q;

        if offset == 0 {
            tracing::debug!(
                ct = ct,
                num_codebooks = num_codebooks,
                dep_q = dep_q,
                in_audio_codebooks = lm_model.in_audio_codebooks(),
                input_tokens_shape = ?input_tokens.dims(),
                "step_inner first step debug"
            );
        }

        // Write input tokens to cache with delays (only for codebooks beyond dep_q)
        let extra_cb_start = dep_q + 1;
        let delays = if extra_cb_start < self.config.delays.len() {
            &self.config.delays[extra_cb_start..]
        } else {
            &[] as &[usize]
        };
        for (i, &delay) in delays.iter().enumerate() {
            let write_pos = (offset + delay) % ct;
            let token = input_tokens.i((0, i, 0))?.to_scalar::<i64>()?;
            // Update cache at position
            let mut cache_vec: Vec<i64> = state.cache.i((0, dep_q + 1 + i, ..))?.to_vec1()?;
            cache_vec[write_pos] = token;
            let new_row = Tensor::from_slice(&cache_vec, ct, &self.device)?;
            state.cache = state.cache.slice_assign(
                &[0..1, (dep_q + 1 + i)..(dep_q + 2 + i), 0..ct],
                &new_row.unsqueeze(0)?.unsqueeze(0)?,
            )?;
        }

        // Get input from cache at current position
        // Python reads ALL codebooks from position (offset % CT), not delay-corrected.
        // The delay handling is done via:
        // 1. Writing user tokens at position (offset + delay)
        // 2. Reading all tokens from position offset
        // 3. Using is_init check to return initial token when offset <= delay
        //
        // Initial tokens (from Python's _get_initial_token()):
        // - Text (codebook 0): text_card = 8000
        // - Audio (codebooks 1+): card = 2048
        let text_initial_token = self.dsm_config.text_card as i64;
        let audio_initial_token = self.dsm_config.card as i64;

        let read_pos = offset % ct;

        // Read tokens from cache and track which audio codebooks should be zeroed
        // Python's ScaledEmbedding with zero_idx=-1 zeros output for -1 tokens
        // We achieve the same by passing None for those audio codebooks
        let mut text_token: i64 = 0;
        let mut audio_tokens: Vec<Option<i64>> = Vec::with_capacity(num_codebooks - 1);

        for cb in 0..num_codebooks {
            let delay = self.config.delays[cb];
            if offset <= delay {
                // is_init case: Python uses state.initial which contains initial_token_id = card = 2048
                if cb == 0 {
                    text_token = text_initial_token;
                } else {
                    // Python's state.initial for audio = initial_token_id = card = 2048
                    audio_tokens.push(Some(audio_initial_token));
                }
            } else {
                let token = state.cache.i((0, cb, read_pos))?.to_scalar::<i64>()?;
                if cb == 0 {
                    // For text tokens read from cache, match Python's ScaledEmbedding behavior:
                    // - Python clamps negative tokens to 0 before embedding lookup
                    // - For -1 (zero token), output is zeroed: embedding[0] * 0 = 0
                    // - For -2 (ungenerated), output is embedding[0] (clamped, not zeroed)
                    // Since we can't zero the embedding here, we use 0 for both -1 and -2
                    // This matches Python's clamp(min=0) behavior
                    text_token = if token < 0 { 0 } else { token };
                } else {
                    // For audio, -1 (zero token) should result in zero embedding
                    // We achieve this by passing None to the LM forward
                    // Python's ScaledEmbedding zeros output for zero_idx=-1
                    if token == -1 {
                        audio_tokens.push(None);  // Zero embedding contribution
                    } else if token == -2 {
                        // Ungenerated - Python clamps -2 to 0 in ScaledEmbedding
                        // This means we use embedding[0], not embedding[card]
                        audio_tokens.push(Some(0i64));
                    } else {
                        audio_tokens.push(Some(token));
                    }
                }
            }
        }

        // Create text tensor
        let text_ids = Tensor::from_slice(&[text_token], (1, 1), &self.device)?.to_dtype(DType::I64)?;

        // Create audio_ids Vec<Option<Tensor>> - None means zero embedding contribution
        let mut audio_ids = Vec::new();
        for cb in 0..lm_model.in_audio_codebooks() {
            if let Some(token) = audio_tokens.get(cb).copied().flatten() {
                let audio_id = Tensor::from_slice(&[token as u32], (1, 1), &self.device)?;
                audio_ids.push(Some(audio_id));
            } else {
                audio_ids.push(None);  // Zero embedding contribution
            }
        }

        // Compute text embedding using our multiplexed embedding (handles demuxing)
        // This replaces the LM's standard text_emb when second_stream_ahead > 0
        // The text_ids can be multiplexed tokens: (second+1) * card + first
        let text_emb = self.multiplexed_text_emb.forward(&text_ids)?;

        // Combine text_emb with sum_condition (cfg + control LUT embeddings)
        // Python combines all AddToInput conditions by summing them
        let combined_emb = match &self.sum_condition {
            Some(sum_cond) => {
                // sum_condition has shape [1, 1, dim], text_emb has shape [1, 1, dim]
                text_emb.broadcast_add(sum_cond)?
            }
            None => text_emb,
        };

        // Pass the combined embedding via conditioner, with text_ids=None to skip LM's text embedding
        let condition = moshi::conditioner::Condition::AddToInput(combined_emb);

        // Run main transformer (batch size 1, all active)
        let mask = StreamMask::new(vec![true], &self.device)?;

        // Use forward_ca if we have cross-attention source (voice conditioning)
        // Otherwise use forward_cond (no voice conditioning)
        let (text_logits, transformer_out) = if let Some(ca_tensor) = ca_src {
            // Use cross-attention for voice conditioning
            // Python: lm.forward_ca(emb, ca_src, mask)
            let ca_src = moshi::transformer::CaSrc::Tokens(ca_tensor.clone());
            lm_model.forward_ca(
                None,  // Don't use LM's text_emb, we're providing via condition
                audio_ids,
                &ca_src,
                Some(&condition),
                &mask,
            )?
        } else {
            // No cross-attention (no voice conditioning)
            lm_model.forward_cond(
                None,  // Don't use LM's text_emb, we're providing via condition
                audio_ids,
                Some(&condition),
                &mask,
            )?
        };

        // Sample text token - text_logits shape is [B, T, vocab] = [1, 1, vocab]
        let text_logits_1d = text_logits.i((0, 0, ..))?.to_dtype(DType::F32)?;

        // Apply padding bonus
        let text_logits_1d = if self.config.padding_bonus != 0.0 {
            let mut logits_vec: Vec<f32> = text_logits_1d.to_vec1()?;
            if let Some(v) = logits_vec.get_mut(self.token_ids.pad as usize) {
                *v += self.config.padding_bonus as f32;
            }
            Tensor::from_vec(logits_vec, text_logits_1d.shape(), &self.device)?
        } else {
            text_logits_1d
        };

        let sampled_text = state.text_lp.sample(&text_logits_1d)?;

        // Process through state machine
        let (out_text, _consumed) = self.machine.process(offset, &mut state.machine_state, sampled_text);

        // Sample audio tokens via TTS depformer
        let mut audio_tokens;
        if offset >= self.config.delay_steps {
            // transformer_out has shape [1, 1, 1, D] or [1, 1, D] - ensure [1, 1, D] for depformer
            // Reshape to 3D by keeping batch, last dim for seq len, and feature dim
            let main_hidden = match transformer_out.rank() {
                4 => transformer_out.squeeze(1)?,  // [1, 1, 1, D] -> [1, 1, D]
                3 => transformer_out.clone(),
                _ => anyhow::bail!("Unexpected transformer_out rank: {}", transformer_out.rank()),
            };
            // NOTE: Pass the FULL multiplexed text token to depformer.
            // Python passes the multiplexed token directly to graphed_depth (lm.py line 746).
            // The depformer's MultiplexedTextEmbedding handles demultiplexing internally.
            // Do NOT demultiplex here - that would break the embedding lookup!
            audio_tokens = self.tts_depformer.sample(
                &main_hidden,
                out_text,  // Full multiplexed token, depformer handles demux
                &mut state.audio_lp,
                &self.device,
            )?;

        } else {
            // Before delay_steps, use zero tokens
            audio_tokens = vec![self.token_ids.zero as u32; dep_q];
        }

        // Apply per-codebook masking (matches Python's _on_audio_hook)
        // Python: mask = offsets < delays[1:dep_q+1] + delay_steps
        // Where delays[cb] is the delay for audio codebook cb (cb starts at 1 for first audio)
        let zero_token = self.token_ids.zero as u32;
        for cb in 0..dep_q {
            // Audio codebook cb corresponds to delays index cb+1 (since delays[0] is text)
            let delay = self.config.delays.get(cb + 1).copied().unwrap_or(0);
            if offset < delay + self.config.delay_steps {
                audio_tokens[cb] = zero_token;
            }
        }

        // Build output frame [1, 1 + dep_q, 1]
        // NOTE: audio_tokens is Vec<u32>, but zero_token (-1) was cast from i32 to u32 (4294967295).
        // When building the frame, we need to convert back to proper signed i64 representation.
        // For zero tokens (4294967295u32 from -1i32), we need -1i64.
        // For valid tokens (0..card), they stay the same.
        let zero_token_u32 = self.token_ids.zero as u32; // -1i32 as u32 = 4294967295
        let zero_token_i64 = self.token_ids.zero as i64; // -1i32 as i64 = -1 (sign-extended)
        let ungenerated_token_id: i64 = -2;  // Python's lm_model.ungenerated_token_id

        // Python (lm.py:775-776): mask = (offsets <= max_delay); out[mask, :, :] = ungenerated_token_id
        // Python's offsets is post-increment (offsets + 1 after line 753), so:
        // - mask = (offset + 1) <= max_delay = offset < max_delay
        // During the delay period, the ENTIRE output frame is set to -2 (ungenerated)
        let frame_vec = if offset < state.max_delay {
            // All outputs masked to ungenerated during delay period
            vec![ungenerated_token_id; 1 + dep_q]
        } else {
            let mut fv = vec![out_text as i64];
            fv.extend(audio_tokens.iter().map(|&t| {
                if t == zero_token_u32 {
                    zero_token_i64 // Properly sign-extended -1
                } else {
                    t as i64 // Normal token, zero-extend is fine
                }
            }));
            fv
        };
        let _frame = Tensor::from_slice(&frame_vec, (1, 1 + dep_q, 1), &self.device)?;

        // Write outputs to cache
        // NOTE: Python increments state.offsets BEFORE writing, so writes happen at (offset+1) % ct.
        // See lm.py line 753: state.offsets += 1, then line 756: positions = (state.offsets % CT)
        let write_pos = (offset + 1) % ct;

        // Use Vec-based cache modification to avoid slice_assign issues on GPU
        // Read entire cache into Vec, modify, then rebuild tensor
        let mut full_cache: Vec<Vec<i64>> = Vec::with_capacity(num_codebooks);
        for cb in 0..num_codebooks {
            let row: Vec<i64> = state.cache.i((0, cb, ..))?.to_vec1()?;
            full_cache.push(row);
        }

        // Text token
        full_cache[0][write_pos] = out_text as i64;

        // Audio tokens
        // Note: token_ids.zero is -1, but audio_tokens are u32 so -1 becomes 4294967295
        // We need to convert back to signed for proper cache handling
        let zero_token_u32 = self.token_ids.zero as u32; // -1 as u32 = 4294967295
        let zero_token_i64 = self.token_ids.zero as i64; // -1
        let delay_steps = self.config.delay_steps;
        for (i, &token) in audio_tokens.iter().enumerate() {
            if i + 1 < num_codebooks {
                // Python's on_audio_hook forces early audio tokens to zero (-1)
                // This matches: if offset < delay + self.delay_steps: audio_tokens[:, q] = self.machine.token_ids.zero
                let audio_cb_delay = self.config.delays.get(i + 1).copied().unwrap_or(0);
                let force_zero = offset < audio_cb_delay + delay_steps;

                // Convert back to signed representation for special tokens
                let token_i64 = if force_zero {
                    zero_token_i64 // Force early audio tokens to zero (-1)
                } else if token == zero_token_u32 {
                    zero_token_i64 // Store as -1
                } else {
                    token as i64
                };
                full_cache[i + 1][write_pos] = token_i64;
            }
        }

        // Rebuild cache tensor from Vec
        let flat_cache: Vec<i64> = full_cache.iter().flatten().copied().collect();
        state.cache = Tensor::from_slice(&flat_cache, (1, num_codebooks, ct), &self.device)?;

        // Return delay-corrected frame
        // NOTE: Python's _step() increments offset BEFORE checking the delay mask,
        // so Python's check `offset_after <= max_delay` is equivalent to our `offset < max_delay`
        // (since offset_after = offset + 1).
        let max_delay = state.max_delay;
        if offset < max_delay {
            // Not enough data yet, return None (matching Python's return None)
            return Ok(None);
        }

        // Gather delay-corrected output
        // NOTE: Python uses (state.offsets - max_delay + delay) where state.offsets is POST-increment
        // (already incremented by +1 at start of step). So we use (offset+1) here.
        let mut out_vec = Vec::new();
        for cb in 0..(dep_q + 1) {
            let delay = self.config.delays[cb];
            let read_pos = ((offset + 1) as isize - max_delay as isize + delay as isize).rem_euclid(ct as isize) as usize;
            let token = state.cache.i((0, cb, read_pos))?.to_scalar::<i64>()?;
            out_vec.push(token);
        }

        Ok(Some(Tensor::from_slice(&out_vec, (1, dep_q + 1, 1), &self.device)?))
    }

    /// Handle a WebSocket connection for streaming TTS
    pub async fn handle_socket(
        &self,
        mut socket: ws::WebSocket,
        query: crate::TtsStreamingQuery,
    ) -> Result<()> {
        use futures_util::StreamExt;

        tracing::info!(voice = ?query.voice, temp = query.temperature, "Starting streaming TTS");

        // Receive text from client until we get null byte
        let mut text_parts: Vec<String> = Vec::new();
        while let Some(msg) = socket.next().await {
            match msg {
                Ok(ws::Message::Text(text)) => {
                    text_parts.push(text.to_string());
                }
                Ok(ws::Message::Binary(data)) => {
                    // Check for null byte signal (end of text)
                    if data.len() == 1 && data[0] == 0 {
                        break;
                    }
                    // Otherwise treat as text
                    if let Ok(text) = String::from_utf8(data.to_vec()) {
                        text_parts.push(text);
                    }
                }
                Ok(ws::Message::Close(_)) => {
                    tracing::debug!("Client closed connection");
                    return Ok(());
                }
                Ok(_) => continue,
                Err(e) => {
                    tracing::warn!(err = ?e, "WebSocket receive error");
                    return Err(e.into());
                }
            }
        }

        let full_text = text_parts.join("");
        if full_text.is_empty() {
            tracing::warn!("No text received from client");
            return Ok(());
        }

        tracing::info!(text = %full_text, "Received text for TTS");

        // Create TTS query and run synchronously
        let tts_query = crate::TtsQuery {
            text: vec![full_text],
            seed: query.seed,
            temperature: query.temperature,
            top_k: query.top_k,
            voice: query.voice.clone(),
            voices: query.voices.clone(),
            max_seq_len: query.max_seq_len,
            return_timestamps: Some(true),
            cfg_alpha: query.cfg_alpha,
        };

        // Run generation (blocking)
        let (wav_data, timestamps) = self.run(&tts_query)?;

        // Decode WAV to get raw PCM
        let pcm = self.decode_wav_to_pcm(&wav_data)?;

        // Create encoder for output format
        let mut encoder = Encoder::new(query.format.clone())?;

        // Send header if any (for OggOpus)
        if let Some(header) = encoder.header()? {
            socket.send(ws::Message::Binary(header.into())).await?;
        }

        // Send timestamps
        for wwts in timestamps {
            if let Some(word_msg) = encoder.encode_word(wwts)? {
                socket.send(ws::Message::Binary(word_msg.into())).await?;
            }
        }

        // Send audio in chunks
        const PCM_CHUNK_SIZE: usize = 2400; // 100ms at 24kHz
        let total_chunks = pcm.len().div_ceil(PCM_CHUNK_SIZE);
        for (i, chunk) in pcm.chunks(PCM_CHUNK_SIZE).enumerate() {
            let encoded = encoder.encode(chunk.to_vec())?;
            socket.send(ws::Message::Binary(encoded.into())).await?;
            if i % 20 == 0 {
                tracing::debug!(chunk = i, total = total_chunks, "Sending audio chunk");
            }
        }

        tracing::info!(total_samples = pcm.len(), total_chunks, "Streaming TTS completed, sending close frame");

        // Send close frame to properly terminate the WebSocket connection
        socket.send(ws::Message::Close(None)).await?;

        // Wait for client's close response to ensure all data is flushed
        // The WebSocket protocol requires both sides to acknowledge the close
        while let Some(msg) = socket.next().await {
            match msg {
                Ok(ws::Message::Close(_)) => {
                    tracing::debug!("Received close frame from client");
                    break;
                }
                Err(e) => {
                    tracing::debug!(err = ?e, "Connection closed while waiting for close ack");
                    break;
                }
                _ => {
                    // Ignore other messages while closing
                    continue;
                }
            }
        }

        Ok(())
    }

    /// Decode WAV data to raw PCM f32 samples
    /// Assumes 16-bit PCM WAV format (as produced by kaudio::wav::write_pcm_as_wav)
    fn decode_wav_to_pcm(&self, wav_data: &[u8]) -> Result<Vec<f32>> {
        // Simple WAV parser - skip 44-byte header and read 16-bit PCM samples
        if wav_data.len() < 44 {
            anyhow::bail!("WAV data too short");
        }

        // Verify RIFF header
        if &wav_data[0..4] != b"RIFF" || &wav_data[8..12] != b"WAVE" {
            anyhow::bail!("Invalid WAV header");
        }

        // Find data chunk - typically at byte 44 but can vary
        let mut pos = 12;
        while pos + 8 < wav_data.len() {
            let chunk_id = &wav_data[pos..pos + 4];
            let chunk_size = u32::from_le_bytes([
                wav_data[pos + 4],
                wav_data[pos + 5],
                wav_data[pos + 6],
                wav_data[pos + 7],
            ]) as usize;

            if chunk_id == b"data" {
                let data_start = pos + 8;
                let data_end = (data_start + chunk_size).min(wav_data.len());
                let pcm_data = &wav_data[data_start..data_end];

                // Convert 16-bit PCM to f32
                let samples: Vec<f32> = pcm_data
                    .chunks_exact(2)
                    .map(|bytes| {
                        let sample = i16::from_le_bytes([bytes[0], bytes[1]]);
                        sample as f32 / 32768.0
                    })
                    .collect();

                return Ok(samples);
            }

            pos += 8 + chunk_size;
            // Align to even boundary
            if chunk_size % 2 == 1 {
                pos += 1;
            }
        }

        anyhow::bail!("No data chunk found in WAV")
    }

    /// Get the device
    #[allow(dead_code)]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the dtype
    #[allow(dead_code)]
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

