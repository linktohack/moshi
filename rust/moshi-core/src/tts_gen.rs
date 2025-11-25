// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

//! TTS Generation module for streaming TTS with the DSM model.
//!
//! This module implements the LMGen-like generation logic specifically for TTS,
//! matching the Python implementation in moshi/models/lm.py.

use candle::{DType, Device, Result, Tensor};
use candle_transformers::generation::LogitsProcessor;

use crate::tts_state_machine::{Entry, State, StateMachine, TokenIds};

/// Configuration for TTS generation
#[derive(Debug, Clone)]
pub struct TtsGenConfig {
    /// Temperature for audio token sampling
    pub temp: f64,
    /// Temperature for text token sampling
    pub temp_text: f64,
    /// Top-k for audio sampling
    pub top_k: usize,
    /// Top-k for text sampling
    pub top_k_text: usize,
    /// CFG coefficient (1.0 = no CFG)
    pub cfg_coef: f64,
    /// Whether CFG applies to no-text condition
    pub cfg_is_no_text: bool,
    /// Maximum padding tokens between words
    pub max_padding: i32,
    /// Initial padding at start
    pub initial_padding: i32,
    /// Final padding after last word
    pub final_padding: usize,
    /// Padding between words
    pub padding_between: usize,
    /// Bonus to add to pad token logits
    pub padding_bonus: f64,
    /// Number of delay steps before audio starts
    pub delay_steps: usize,
    /// Number of active audio codebooks (n_q in config)
    pub n_q: usize,
    /// Lookahead for second text stream
    pub second_stream_ahead: usize,
}

impl Default for TtsGenConfig {
    fn default() -> Self {
        Self {
            temp: 0.6,
            temp_text: 0.7,
            top_k: 250,
            top_k_text: 0,
            cfg_coef: 2.0,
            cfg_is_no_text: true,
            max_padding: 8,
            initial_padding: 2,
            final_padding: 4,
            padding_between: 1,
            padding_bonus: 0.0,
            delay_steps: 16,
            n_q: 24,
            second_stream_ahead: 2,
        }
    }
}

/// Streaming state for TTS generation
#[allow(dead_code)]
pub struct TtsGenState {
    /// Cache for tokens with delay handling
    cache: Tensor,
    /// Initial token tensor
    initial: Tensor,
    /// Current offset per batch element
    offsets: Vec<usize>,
    /// Delays for each codebook (on device)
    delays_cuda: Tensor,
    /// Maximum delay across all codebooks
    max_delay: usize,
    /// Batch size
    batch_size: usize,
    /// Device
    device: Device,
    /// DType
    dtype: DType,
    /// Sum condition (added to embeddings)
    condition_sum: Option<Tensor>,
    /// Cross-attention condition
    condition_cross: Option<Tensor>,
}

impl TtsGenState {
    /// Create a new streaming state
    pub fn new(
        batch_size: usize,
        num_codebooks: usize,
        delays: &[usize],
        device: &Device,
        dtype: DType,
        condition_sum: Option<Tensor>,
        condition_cross: Option<Tensor>,
    ) -> Result<Self> {
        let max_delay = delays.iter().copied().max().unwrap_or(0);
        let cache_size = max_delay + 2;

        // Initialize cache with ungenerated token ID
        let ungenerated: i64 = -2;
        let cache = Tensor::full(
            ungenerated,
            (batch_size, num_codebooks, cache_size),
            device,
        )?.to_dtype(DType::I64)?;

        // Create initial token tensor (all zeros for padding)
        let initial = Tensor::zeros((batch_size, num_codebooks, 1), DType::I64, device)?;

        let offsets = vec![0; batch_size];

        let delays_cuda = Tensor::from_slice(
            &delays.iter().map(|&d| d as i64).collect::<Vec<_>>(),
            delays.len(),
            device,
        )?;

        Ok(Self {
            cache,
            initial,
            offsets,
            delays_cuda,
            max_delay,
            batch_size,
            device: device.clone(),
            dtype,
            condition_sum,
            condition_cross,
        })
    }

    /// Reset streaming state for specific batch indices
    pub fn reset(&mut self, batch_indices: &[usize]) -> Result<()> {
        for &idx in batch_indices {
            if idx < self.batch_size {
                self.offsets[idx] = 0;
            }
        }
        Ok(())
    }
}

/// TTS Generator - handles streaming generation
pub struct TtsGen {
    config: TtsGenConfig,
    token_ids: TokenIds,
    machine: StateMachine,
    audio_lp: LogitsProcessor,
    text_lp: LogitsProcessor,
}

impl TtsGen {
    pub fn new(config: TtsGenConfig, text_card: usize) -> Self {
        let token_ids = TokenIds::new(text_card + 1);
        let machine = StateMachine::new(
            token_ids,
            config.second_stream_ahead,
            config.max_padding,
            config.initial_padding,
        );

        let audio_lp = LogitsProcessor::new(0, Some(config.temp), Some(config.top_k as f64));
        let text_lp = LogitsProcessor::new(0, Some(config.temp_text), Some(config.top_k_text as f64));

        Self {
            config,
            token_ids,
            machine,
            audio_lp,
            text_lp,
        }
    }

    /// Get a reference to the state machine
    pub fn machine(&self) -> &StateMachine {
        &self.machine
    }

    /// Get a reference to the token IDs
    pub fn token_ids(&self) -> &TokenIds {
        &self.token_ids
    }

    /// Get the config
    pub fn config(&self) -> &TtsGenConfig {
        &self.config
    }

    /// Sample text token from logits
    pub fn sample_text(&mut self, logits: &Tensor) -> Result<u32> {
        // Apply padding bonus if configured
        let logits = if self.config.padding_bonus != 0.0 {
            let mut logits_vec = logits.to_vec1::<f32>()?;
            if let Some(v) = logits_vec.get_mut(self.token_ids.pad as usize) {
                *v += self.config.padding_bonus as f32;
            }
            Tensor::from_vec(logits_vec, logits.shape(), logits.device())?
        } else {
            logits.clone()
        };

        let token = self.text_lp.sample(&logits)?;
        Ok(token)
    }

    /// Sample audio token from logits
    pub fn sample_audio(&mut self, logits: &Tensor) -> Result<u32> {
        let token = self.audio_lp.sample(logits)?;
        Ok(token)
    }

    /// Create a new state for generation
    pub fn new_state(&self, entries: Vec<Entry>) -> State {
        self.machine.new_state(entries)
    }

    /// Process a text token through the state machine
    pub fn process_text(
        &self,
        step: usize,
        state: &mut State,
        token: u32,
    ) -> (u32, bool) {
        self.machine.process(step, state, token)
    }
}

/// Helper function to compute CFG blended logits
pub fn apply_cfg(
    logits: &Tensor,
    logits_null: &Tensor,
    cfg_coef: f64,
) -> Result<Tensor> {
    // logits = logits_null + (logits - logits_null) * cfg_coef
    let diff = (logits - logits_null)?;
    let scaled = (diff * cfg_coef)?;
    logits_null + scaled
}

/// Load delays from model config
pub fn get_delays_from_config(delays: &[usize], n_q: usize) -> Vec<usize> {
    // For TTS DSM model, delays are configured in the model config
    // First element is text, rest are audio codebooks
    // We need delays for: [text, audio_0, audio_1, ..., audio_{n_q-1}]
    if delays.len() >= n_q + 1 {
        delays[..n_q + 1].to_vec()
    } else {
        // Fallback: no delays
        vec![0; n_q + 1]
    }
}

