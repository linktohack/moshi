// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

//! TTS State Machine for Delayed Streams Modeling (DSM) based TTS.
//!
//! This module implements the logic for the state machine around the DSM-based TTS model.
//! For TTS, we start from pure text (not text properly padded with time alignment) and
//! co-generate the padded text sequence along with the audio output. The model signals
//! when it thinks the next step will be the start of a word, and we then pop a word
//! and feed it the token representation of the word over the next few steps.

use std::collections::VecDeque;
use tracing;

/// Special token IDs used by the TTS state machine.
#[derive(Debug, Clone, Copy)]
pub struct TokenIds {
    /// Text cardinality, including the initial token (1 + tokenizer cardinality).
    /// This is used for multiplexing multiple input tokens into the text stream.
    pub card: usize,
    /// A new word is starting.
    pub new_word: u32,
    /// Padding, nothing happens.
    pub pad: u32,
    /// Indicates the start of turn of the main speaker.
    pub main: u32,
    /// Indicates the start of turn of the other speaker.
    pub other: u32,
    /// Special value that is embedded to exactly 0.
    pub zero: i32,
    /// Indicate that a value is not yet generated but should be.
    pub ungenerated: i32,
}

impl TokenIds {
    /// Create new TokenIds with the given text card (vocabulary size + 1).
    pub fn new(card: usize) -> Self {
        Self {
            card,
            new_word: 0,
            pad: 3,
            main: 1,
            other: 2,
            zero: -1,
            ungenerated: -2,
        }
    }
}

/// One word to generate.
#[derive(Debug, Clone)]
pub struct Entry {
    /// List of tokens for this word.
    pub tokens: Vec<u32>,
    /// Word as string.
    pub text: String,
    /// If > 0, we will prevent the model from sampling a new word for that
    /// many steps after the current word. Note that even for `padding=0`, the model
    /// will be forbidden to sample a new word until all the tokens for the current word are consumed.
    pub padding: usize,
}

impl Entry {
    pub fn new(tokens: Vec<u32>, text: String, padding: usize) -> Self {
        Self { tokens, text, padding }
    }

    pub fn pause(padding: usize) -> Self {
        Self { tokens: Vec::new(), text: String::new(), padding }
    }
}

/// State of the TTS Machine.
#[derive(Debug, Clone)]
pub struct State {
    /// Queue containing the entries to generate.
    pub entries: VecDeque<Entry>,
    /// How many times the model can still sample a pad.
    pub remaining_padding: i32,
    /// How many times the model is still forced to sample a pad.
    pub forced_padding: i32,
    /// Queue containing the main stream text tokens to feed.
    pub queued: VecDeque<u32>,
    /// Queue containing the lookahead text tokens to feed.
    pub lookahead_queued: VecDeque<u32>,
    /// Once we reach the end of the generation, this is set to the current step.
    /// The end of the generation is once the model samples a `word` but `entries` is empty.
    pub end_step: Option<usize>,
    /// List of steps at which each entry in `entries` was consumed.
    pub consumption_times: Vec<usize>,
    /// List of tuples `(word, step)`, at which each word was consumed.
    pub transcript: Vec<(String, usize)>,
}

impl State {
    pub fn new(entries: VecDeque<Entry>, remaining_padding: i32, forced_padding: i32) -> Self {
        Self {
            entries,
            remaining_padding,
            forced_padding,
            queued: VecDeque::new(),
            lookahead_queued: VecDeque::new(),
            end_step: None,
            consumption_times: Vec::new(),
            transcript: Vec::new(),
        }
    }

    /// Get the tokens for the Nth entry with tokens (used for lookahead).
    pub fn get_tokens_ahead(&self, mut lookahead: usize) -> Vec<u32> {
        assert!(lookahead > 0);
        for entry in &self.entries {
            if !entry.tokens.is_empty() {
                lookahead -= 1;
                if lookahead == 0 {
                    return entry.tokens.clone();
                }
            }
        }
        Vec::new()
    }
}

/// State machine that manipulates the `State` based on the model prediction.
/// In particular, every time the model predicts a `word` (see `TokenIds`) special token,
/// the state machine will pop the next word to synthesize and start feeding it.
/// The model is optionally equipped with a second input text stream providing a lookahead
/// into the future text.
#[derive(Debug, Clone)]
pub struct StateMachine {
    /// Special token values.
    pub token_ids: TokenIds,
    /// If > 0, the model needs a second stream for lookahead.
    pub second_stream_ahead: usize,
    /// Maximum number of padding tokens that can be sampled in a row.
    pub max_padding: i32,
    /// Number of padding tokens at the beginning, to prevent the first word from being cut.
    pub initial_padding: i32,
}

impl StateMachine {
    pub fn new(
        token_ids: TokenIds,
        second_stream_ahead: usize,
        max_padding: i32,
        initial_padding: i32,
    ) -> Self {
        Self {
            token_ids,
            second_stream_ahead,
            max_padding,
            initial_padding,
        }
    }

    /// Create a new state from the given entries.
    pub fn new_state(&self, entries: impl IntoIterator<Item = Entry>) -> State {
        State::new(
            entries.into_iter().collect(),
            self.initial_padding,
            self.initial_padding,
        )
    }

    /// Process the output of the model.
    ///
    /// # Arguments
    /// * `step` - Current step index.
    /// * `state` - State to act upon.
    /// * `token` - Model prediction.
    ///
    /// # Returns
    /// * `output_token` - Value to use as the text input for the model at the next step.
    /// * `consumed_new_word` - True if a new word was consumed.
    pub fn process(&self, step: usize, state: &mut State, token: u32) -> (u32, bool) {
        let mut consumed_new_word = false;
        let mut token = token;

        // Validate token - only new_word and pad are valid
        if token != self.token_ids.new_word && token != self.token_ids.pad {
            token = self.token_ids.pad;
        }

        // Determine what token to use based on state

        if !state.queued.is_empty() {
            // Some text tokens are yet to be fed, we must PAD.
            token = self.token_ids.pad;
        } else if state.forced_padding > 0 {
            // We are forced to pad, we must PAD.
            token = self.token_ids.pad;
        } else if state.remaining_padding <= 0 {
            // We are not allowed to pad, we must ask for a new WORD.
            token = self.token_ids.new_word;
        }

        if token == self.token_ids.new_word {
            if let Some(entry) = state.entries.pop_front() {
                state.consumption_times.push(step);
                consumed_new_word = true;

                if !entry.tokens.is_empty() {
                    state.transcript.push((entry.text.clone(), step));
                    // We queue the tokens to be fed to the model.
                    state.queued.extend(entry.tokens.iter().copied());
                    if self.second_stream_ahead > 0 {
                        // We queue the tokens for the N+lookahead word into the second text stream.
                        state
                            .lookahead_queued
                            .extend(state.get_tokens_ahead(self.second_stream_ahead));
                    }
                    // Entry contains a new word, we reset the max padding counter.
                    state.remaining_padding = self.max_padding;
                } else {
                    // Entry is only here to insert a break, pretend the token was a PAD.
                    token = self.token_ids.pad;
                }
                state.forced_padding = entry.padding as i32;
            } else {
                token = self.token_ids.pad;
                if self.second_stream_ahead > 0 && state.end_step.is_none() {
                    token = self.token_ids.new_word;
                }
                // Trying to consume past the last word, we reached the end.
                if state.end_step.is_none() {
                    tracing::debug!(step, "Setting end_step");
                    state.end_step = Some(step);
                }
            }
        }

        let output: u32;
        if token == self.token_ids.pad {
            // Decrement the counters for remaining and forced pads.
            if state.remaining_padding > 0 {
                state.remaining_padding -= 1;
            }
            if state.forced_padding > 0 {
                state.forced_padding -= 1;
            }
            if let Some(queued_token) = state.queued.pop_front() {
                // We have some text tokens to feed to the model.
                output = queued_token;
            } else {
                output = self.token_ids.pad;
            }
        } else if token == self.token_ids.new_word {
            output = self.token_ids.new_word;
        } else if token == self.token_ids.zero as u32 {
            output = token;
        } else {
            panic!("Invalid token {}", token);
        }

        // Handle second stream (lookahead)
        let final_output = if self.second_stream_ahead > 0 {
            let second: i32;
            let mut actual_output = output;

            if actual_output == self.token_ids.new_word {
                // If sampled the `word` special token, we put it on the
                // second text stream instead of the main one.
                second = self.token_ids.new_word as i32;
                if let Some(queued_token) = state.queued.pop_front() {
                    // This allows us to pass the current word tokens faster.
                    actual_output = queued_token;
                } else {
                    actual_output = self.token_ids.pad;
                }
            } else if let Some(lookahead_token) = state.lookahead_queued.pop_front() {
                // Otherwise if we have some lookahead tokens we feed them.
                second = lookahead_token as i32;
            } else {
                second = -1;
            }

            // Multiplex the two tokens. We add `+1` to `second` so that
            // we can encode -1, which would translate to an all 0s embedding.
            // This will get de-multiplexed in the embedding in lm.
            let multiplexed = ((second + 1) as u32) * (self.token_ids.card as u32) + actual_output;
            tracing::debug!(
                step,
                output,
                second,
                actual_output,
                multiplexed,
                card = self.token_ids.card,
                "StateMachine::process multiplex"
            );
            multiplexed
        } else {
            output
        };

        (final_output, consumed_new_word)
    }
}
