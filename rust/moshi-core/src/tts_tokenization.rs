// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

//! TTS Tokenization utilities for converting text scripts to entries.
//!
//! This module provides standalone functions for text preprocessing and tokenization
//! that can be unit tested independently of the TTS model.

use crate::tts_state_machine::{Entry, TokenIds};
use regex::Regex;

/// Configuration for script-to-entries conversion.
#[derive(Debug, Clone)]
pub struct ScriptConfig {
    /// Whether the model supports multiple speakers.
    pub multi_speaker: bool,
    /// Amount of padding to force between words (makes articulation clearer).
    pub padding_between: usize,
    /// Frame rate for calculating break durations.
    pub frame_rate: f64,
}

impl Default for ScriptConfig {
    fn default() -> Self {
        Self {
            multi_speaker: true,
            padding_between: 0,
            frame_rate: 12.5,
        }
    }
}

/// Preprocess a single line of text by applying character replacements.
/// Matches Python's preprocessing in script_to_entries.
pub fn preprocess_line(line: &str) -> String {
    line.replace('\u{2019}', "'")  // Right single quotation mark to apostrophe
        .replace(':', " ")          // Colon to space
        .replace('(', "")           // Remove open parenthesis
        .replace(')', "")           // Remove close parenthesis
}

/// Result of parsing a line segment.
#[derive(Debug, Clone, PartialEq)]
pub enum LineSegment {
    /// A word to tokenize.
    Word(String),
    /// A break/pause with duration in seconds.
    Break(f64),
}

/// Parse a line into segments, handling break tags.
/// Matches Python's regex parsing: `(?:<break\s+time="([0-9]+(?:.[0-9]*)?)s"\s*/?>)|(?:\s+)`
pub fn parse_line_segments(line: &str) -> Vec<LineSegment> {
    // Regex for break tags and whitespace
    let re = Regex::new(r#"(?:<break\s+time="([0-9]+(?:\.[0-9]*)?)s"\s*/?>)|(?:\s+)"#).unwrap();

    let mut segments = Vec::new();
    let mut last_end = 0;

    for cap in re.captures_iter(line) {
        let m = cap.get(0).unwrap();

        // Text before the match
        let before = &line[last_end..m.start()];
        if !before.is_empty() {
            segments.push(LineSegment::Word(before.to_string()));
        }

        // Check if this is a break tag (group 1 captured the time)
        if let Some(time_match) = cap.get(1) {
            let duration: f64 = time_match.as_str().parse().unwrap_or(0.0);
            segments.push(LineSegment::Break(duration));
        }
        // If it's just whitespace, we don't add anything (words are separated by whitespace)

        last_end = m.end();
    }

    // Text after the last match
    if last_end < line.len() {
        let remaining = &line[last_end..];
        if !remaining.is_empty() {
            segments.push(LineSegment::Word(remaining.to_string()));
        }
    }

    segments
}

/// Calculate padding for a word given the padding_between setting.
/// Formula: if padding_between > 0, padding = max(0, padding_between + num_tokens - 1)
pub fn calculate_padding(num_tokens: usize, padding_between: usize) -> usize {
    if padding_between > 0 {
        (padding_between + num_tokens).saturating_sub(1)
    } else {
        0
    }
}

/// Convert a script (list of lines/turns) to entries using a tokenizer function.
///
/// This is a generic version that works with any tokenizer implementation.
///
/// # Arguments
/// * `script` - List of lines, each line is a turn (alternating speakers in multi-speaker mode)
/// * `token_ids` - Special token IDs
/// * `config` - Script configuration
/// * `tokenize` - Function that converts a word to token IDs
pub fn script_to_entries<F>(
    script: &[String],
    token_ids: &TokenIds,
    config: &ScriptConfig,
    mut tokenize: F,
) -> Vec<Entry>
where
    F: FnMut(&str) -> Vec<u32>,
{
    let speaker_tokens = [token_ids.main, token_ids.other];
    let mut last_speaker: Option<usize> = None;
    let mut entries = Vec::new();

    for (line_idx, line) in script.iter().enumerate() {
        // Preprocess the line
        let line = preprocess_line(line);

        // Track first content of each line (for speaker token insertion)
        let mut first_content = true;

        // Parse line into segments
        let segments = parse_line_segments(&line);

        for segment in segments {
            match segment {
                LineSegment::Word(word) => {
                    // Tokenize the word
                    let mut tokens = tokenize(&word);

                    // Insert speaker token at start of first word if multi-speaker
                    if first_content {
                        let speaker = line_idx % speaker_tokens.len();
                        if config.multi_speaker && last_speaker != Some(speaker) {
                            last_speaker = Some(speaker);
                            tokens.insert(0, speaker_tokens[speaker]);
                        }
                        first_content = false;
                    }

                    // Calculate padding
                    let padding = calculate_padding(tokens.len(), config.padding_between);

                    entries.push(Entry::new(tokens, word, padding));
                }
                LineSegment::Break(duration) => {
                    // Create a pause entry
                    let padding = (duration * config.frame_rate).round() as usize;
                    entries.push(Entry::pause(padding));

                    // After a break, next word is still considered first_content for speaker
                    // This matches Python behavior where break doesn't reset first_content
                }
            }
        }
    }

    entries
}

