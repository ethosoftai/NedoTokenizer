use nedo_morph_bundle::{NativeAnalysis, NativeMorpheme, NativeMorphology};

use super::{alignment::align_analysis, TokenizerError};

struct NormalizedCandidate {
    text: String,
    boundaries: Vec<(usize, usize)>,
}

impl NormalizedCandidate {
    fn original_boundary(&self, candidate_boundary: usize) -> Option<usize> {
        if candidate_boundary == 0 {
            return Some(0);
        }
        self.boundaries.iter().find_map(|(candidate, original)| {
            (*candidate == candidate_boundary).then_some(*original)
        })
    }
}

/// `NedoFormer` input-side analysis with a byte-mapped normalization shadow.
///
/// Original surface bytes are never rewritten.  The shadow is tried only after
/// the normal production analyzer/quality fallbacks fail, and every recovered
/// cut is mapped back onto a boundary in the original token.
pub fn analyze_token_with_nedoformer_shadow(
    morphology: &NativeMorphology<'_>,
    token: &str,
) -> Result<Vec<NativeAnalysis>, TokenizerError> {
    let direct = analyze_token_with_quality_fallback(morphology, token)?;
    if !direct.is_empty() {
        return Ok(direct);
    }

    let Some(lowered) = turkish_lower_candidate(token) else {
        return Ok(Vec::new());
    };
    let mut recovered = Vec::new();
    if let Some(analysis) = analysis_from_candidate(morphology, token, &lowered, "ShadowLower")? {
        recovered.push(analysis);
    }
    for candidate in deasciify_candidates(&lowered, 64) {
        if let Some(analysis) =
            analysis_from_candidate(morphology, token, &candidate, "ShadowDeasciify")?
        {
            if !recovered.iter().any(|existing| {
                existing.canonical == analysis.canonical
                    && existing
                        .morphemes
                        .iter()
                        .map(|m| m.surface.as_str())
                        .eq(analysis.morphemes.iter().map(|m| m.surface.as_str()))
            }) {
                recovered.push(analysis);
            }
        }
    }
    Ok(recovered)
}

pub(super) fn analyze_token_with_quality_fallback(
    morphology: &NativeMorphology<'_>,
    token: &str,
) -> Result<Vec<NativeAnalysis>, TokenizerError> {
    if token.is_empty() || token.chars().count() > 512 {
        return Ok(Vec::new());
    }
    if let Some(analysis) = numeric_time_with_suffix_analysis(token) {
        return Ok(vec![analysis]);
    }
    let analyses = morphology.analyze_token(token)?;
    if !analyses.is_empty() {
        return Ok(analyses);
    }

    if let Some(analysis) = attached_question_analysis(morphology, token)? {
        return Ok(vec![analysis]);
    }
    if let Some(candidate) = turkish_composition_candidate(token) {
        if let Some(analysis) =
            analysis_from_candidate(morphology, token, &candidate, "Normalized")?
        {
            return Ok(vec![analysis]);
        }
    }
    if let Some(candidate) = trailing_emphasis_candidate(token) {
        if let Some(analysis) = analysis_from_candidate(morphology, token, &candidate, "Emphasis")?
        {
            return Ok(vec![analysis]);
        }
    }
    if turkish_single_letter(token) {
        return Ok(vec![synthetic_analysis(
            token,
            &[],
            token,
            "Noun",
            "Abbrv",
            "Letter",
            false,
        )]);
    }
    Ok(Vec::new())
}

fn numeric_time_with_suffix_analysis(token: &str) -> Option<NativeAnalysis> {
    let (apostrophe_start, apostrophe) = token.char_indices().find(|(_, value)| {
        matches!(
            *value,
            '\'' | '\u{2018}' | '\u{2019}' | '\u{02bc}' | '\u{ff07}'
        )
    })?;
    let apostrophe_end = apostrophe_start + apostrophe.len_utf8();
    let clock = &token[..apostrophe_start];
    let suffix = &token[apostrophe_end..];
    if suffix.is_empty() || !suffix.chars().all(char::is_alphabetic) || !looks_like_clock(clock) {
        return None;
    }
    let dictionary_id = "NEDO_ClockSuffix_Fallback".to_owned();
    let morphemes = vec![
        NativeMorpheme {
            name: "ClockRoot".to_owned(),
            id: "ClockRoot".to_owned(),
            surface: clock.to_owned(),
            derivational: false,
            informal: false,
            pos: Some("Num".to_owned()),
            mapped_id: None,
        },
        NativeMorpheme {
            name: "ClockSuffix".to_owned(),
            id: "ClockSuffix".to_owned(),
            surface: suffix.to_owned(),
            derivational: false,
            informal: false,
            pos: None,
            mapped_id: None,
        },
    ];
    let mut canonical = dictionary_id.clone();
    canonical.push('\u{1}');
    for morpheme in &morphemes {
        canonical.push_str(&morpheme.id);
        canonical.push('=');
        canonical.push_str(&morpheme.surface);
        canonical.push('\u{2}');
    }
    Some(NativeAnalysis {
        canonical,
        dictionary_id,
        lemma: clock.to_owned(),
        primary_pos: "Num".to_owned(),
        secondary_pos: "Clock".to_owned(),
        surface_form: format!("{clock}{suffix}"),
        stem: clock.to_owned(),
        ending: suffix.to_owned(),
        morphemes,
    })
}

fn looks_like_clock(value: &str) -> bool {
    let parts = value.split(':').collect::<Vec<_>>();
    if !(2..=3).contains(&parts.len()) {
        return false;
    }
    parts.iter().enumerate().all(|(index, part)| {
        !part.is_empty()
            && part.len() <= 2
            && part.bytes().all(|byte| byte.is_ascii_digit())
            && (index == 0 || part.len() == 2)
    })
}

fn attached_question_analysis(
    morphology: &NativeMorphology<'_>,
    token: &str,
) -> Result<Option<NativeAnalysis>, TokenizerError> {
    let lowered = turkish_lower(token);
    if !["mı", "mi", "mu", "mü"]
        .iter()
        .any(|suffix| lowered.ends_with(suffix))
    {
        return Ok(None);
    }
    let character_count = token.chars().count();
    if character_count <= 3 {
        return Ok(None);
    }
    let Some(split) = byte_boundary_after_characters(token, character_count - 2) else {
        return Ok(None);
    };
    let prefix = &token[..split];
    let analyses = morphology.analyze_token(prefix)?;
    for base in analyses {
        let Ok(aligned) = align_analysis(prefix, 0, &base) else {
            continue;
        };
        let mut cuts = aligned
            .cuts
            .into_iter()
            .map(|value| usize::try_from(value))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| TokenizerError::LengthOverflow("informal question cut"))?;
        cuts.push(split);
        cuts.sort_unstable();
        cuts.dedup();
        return Ok(Some(synthetic_analysis(
            token,
            &cuts,
            &base.lemma,
            &base.primary_pos,
            &base.secondary_pos,
            "QuestionAttached",
            true,
        )));
    }
    Ok(None)
}

fn analysis_from_candidate(
    morphology: &NativeMorphology<'_>,
    original: &str,
    candidate: &NormalizedCandidate,
    label: &str,
) -> Result<Option<NativeAnalysis>, TokenizerError> {
    if candidate.text == original {
        return Ok(None);
    }
    let analyses = morphology.analyze_token(&candidate.text)?;
    if analyses.is_empty() && turkish_single_letter(&candidate.text) {
        return Ok(Some(synthetic_analysis(
            original,
            &[],
            &candidate.text,
            "Noun",
            "Abbrv",
            label,
            false,
        )));
    }
    for base in analyses {
        let Ok(aligned) = align_analysis(&candidate.text, 0, &base) else {
            continue;
        };
        let mut cuts = Vec::with_capacity(aligned.cuts.len());
        let mut valid = true;
        for cut in aligned.cuts {
            let candidate_boundary = usize::try_from(cut)
                .map_err(|_| TokenizerError::LengthOverflow("normalized quality cut"))?;
            let Some(original_boundary) = candidate.original_boundary(candidate_boundary) else {
                valid = false;
                break;
            };
            cuts.push(original_boundary);
        }
        if !valid {
            continue;
        }
        cuts.sort_unstable();
        cuts.dedup();
        return Ok(Some(synthetic_analysis(
            original,
            &cuts,
            &base.lemma,
            &base.primary_pos,
            &base.secondary_pos,
            label,
            label == "Emphasis",
        )));
    }
    Ok(None)
}

fn synthetic_analysis(
    token: &str,
    cuts: &[usize],
    lemma: &str,
    primary_pos: &str,
    secondary_pos: &str,
    label: &str,
    informal: bool,
) -> NativeAnalysis {
    let mut boundaries = Vec::with_capacity(cuts.len() + 2);
    boundaries.push(0);
    boundaries.extend(
        cuts.iter()
            .copied()
            .filter(|cut| *cut > 0 && *cut < token.len() && token.is_char_boundary(*cut)),
    );
    boundaries.push(token.len());
    boundaries.sort_unstable();
    boundaries.dedup();

    let mut morphemes = Vec::with_capacity(boundaries.len().saturating_sub(1));
    for (index, window) in boundaries.windows(2).enumerate() {
        let surface = token[window[0]..window[1]].to_owned();
        let id = if index == 0 {
            format!("{label}Root")
        } else {
            format!("{label}Part{index}")
        };
        morphemes.push(NativeMorpheme {
            name: id.clone(),
            id,
            surface,
            derivational: false,
            informal,
            pos: (index == 0).then(|| primary_pos.to_owned()),
            mapped_id: None,
        });
    }
    let stem_end = boundaries.get(1).copied().unwrap_or(token.len());
    let stem = token[..stem_end].to_owned();
    let ending = token[stem_end..].to_owned();
    let dictionary_id = format!("NEDO_{label}_Fallback");
    let mut canonical = format!("{dictionary_id}\u{1}");
    for morpheme in &morphemes {
        canonical.push_str(&morpheme.id);
        canonical.push('=');
        canonical.push_str(&morpheme.surface);
        canonical.push('\u{2}');
    }
    NativeAnalysis {
        canonical,
        dictionary_id,
        lemma: lemma.to_owned(),
        primary_pos: primary_pos.to_owned(),
        secondary_pos: secondary_pos.to_owned(),
        surface_form: token.to_owned(),
        stem,
        ending,
        morphemes,
    }
}

fn byte_boundary_after_characters(token: &str, count: usize) -> Option<usize> {
    if count == 0 {
        return Some(0);
    }
    token.char_indices().nth(count).map_or_else(
        || (token.chars().count() == count).then_some(token.len()),
        |entry| Some(entry.0),
    )
}

fn turkish_lower_candidate(token: &str) -> Option<NormalizedCandidate> {
    let mut text = String::with_capacity(token.len());
    let mut boundaries = Vec::with_capacity(token.chars().count());
    let mut changed = false;
    let characters = token.char_indices().collect::<Vec<_>>();
    for (position, (_, value)) in characters.iter().copied().enumerate() {
        let original_end = characters
            .get(position + 1)
            .map_or(token.len(), |entry| entry.0);
        let lowered = match value {
            'I' => 'ı',
            'İ' => 'i',
            _ => {
                let mut values = value.to_lowercase();
                let first = values.next()?;
                if values.next().is_some() {
                    return None;
                }
                first
            }
        };
        changed |= lowered != value;
        text.push(lowered);
        boundaries.push((text.len(), original_end));
    }
    (changed || text == token).then_some(NormalizedCandidate { text, boundaries })
}

fn deasciify_candidates(base: &NormalizedCandidate, maximum: usize) -> Vec<NormalizedCandidate> {
    if maximum == 0 {
        return Vec::new();
    }
    let mut characters = Vec::new();
    for (index, value) in base.text.char_indices() {
        let end = index + value.len_utf8();
        let Some(original_end) = base.original_boundary(end) else {
            return Vec::new();
        };
        characters.push((value, original_end));
    }
    let mut variants = vec![(String::new(), Vec::<(usize, usize)>::new(), false)];
    for (value, original_end) in characters {
        let alternate = deasciify_alternate(value);
        let mut next = Vec::with_capacity(variants.len().saturating_mul(2).min(maximum));
        for (text, boundaries, changed) in variants {
            if next.len() < maximum {
                let mut unchanged_text = text.clone();
                unchanged_text.push(value);
                let mut unchanged_boundaries = boundaries.clone();
                unchanged_boundaries.push((unchanged_text.len(), original_end));
                next.push((unchanged_text, unchanged_boundaries, changed));
            }
            if let Some(alternate) = alternate {
                if next.len() < maximum {
                    let mut changed_text = text;
                    changed_text.push(alternate);
                    let mut changed_boundaries = boundaries;
                    changed_boundaries.push((changed_text.len(), original_end));
                    next.push((changed_text, changed_boundaries, true));
                }
            }
        }
        variants = next;
    }
    variants
        .into_iter()
        .filter_map(|(text, boundaries, changed)| {
            (changed && text != base.text).then_some(NormalizedCandidate { text, boundaries })
        })
        .collect()
}

const fn deasciify_alternate(value: char) -> Option<char> {
    match value {
        'c' => Some('ç'),
        'g' => Some('ğ'),
        'i' => Some('ı'),
        'o' => Some('ö'),
        's' => Some('ş'),
        'u' => Some('ü'),
        _ => None,
    }
}

fn turkish_composition_candidate(token: &str) -> Option<NormalizedCandidate> {
    let characters = token.char_indices().collect::<Vec<_>>();
    let mut text = String::with_capacity(token.len());
    let mut boundaries = Vec::with_capacity(characters.len());
    let mut changed = false;
    let mut index = 0_usize;
    while index < characters.len() {
        let (start, value) = characters[index];
        let end = characters
            .get(index + 1)
            .map_or(token.len(), |entry| entry.0);
        if let Some((_, mark)) = characters.get(index + 1).copied() {
            if let Some(composed) = compose_turkish(value, mark) {
                let original_end = characters
                    .get(index + 2)
                    .map_or(token.len(), |entry| entry.0);
                text.push(composed);
                boundaries.push((text.len(), original_end));
                changed = true;
                index += 2;
                continue;
            }
        }
        let _ = start;
        text.push(value);
        boundaries.push((text.len(), end));
        index += 1;
    }
    changed.then_some(NormalizedCandidate { text, boundaries })
}

fn trailing_emphasis_candidate(token: &str) -> Option<NormalizedCandidate> {
    let characters = token.char_indices().collect::<Vec<_>>();
    let last = characters.last()?.1;
    let mut run_start = characters.len() - 1;
    while run_start > 0 && characters[run_start - 1].1.eq_ignore_ascii_case(&last) {
        run_start -= 1;
    }
    let run_length = characters.len() - run_start;
    if run_start == 0 || !(3..=8).contains(&run_length) || !last.is_alphabetic() {
        return None;
    }
    let mut text = String::with_capacity(token.len());
    let mut boundaries = Vec::with_capacity(run_start + 1);
    for (position, (_, value)) in characters.iter().copied().enumerate() {
        if position < run_start {
            let original_end = characters
                .get(position + 1)
                .map_or(token.len(), |entry| entry.0);
            text.push(value);
            boundaries.push((text.len(), original_end));
        } else if position == run_start {
            text.push(value);
            boundaries.push((text.len(), token.len()));
            break;
        }
    }
    Some(NormalizedCandidate { text, boundaries })
}

const fn compose_turkish(base: char, mark: char) -> Option<char> {
    match (base, mark) {
        ('g', '\u{0306}') => Some('ğ'),
        ('G', '\u{0306}') => Some('Ğ'),
        ('s', '\u{0327}') => Some('ş'),
        ('S', '\u{0327}') => Some('Ş'),
        ('c', '\u{0327}') => Some('ç'),
        ('C', '\u{0327}') => Some('Ç'),
        ('o', '\u{0308}') => Some('ö'),
        ('O', '\u{0308}') => Some('Ö'),
        ('u', '\u{0308}') => Some('ü'),
        ('U', '\u{0308}') => Some('Ü'),
        ('a', '\u{0302}') => Some('â'),
        ('A', '\u{0302}') => Some('Â'),
        ('i', '\u{0302}') => Some('î'),
        ('I', '\u{0302}') => Some('Î'),
        ('u', '\u{0302}') => Some('û'),
        ('U', '\u{0302}') => Some('Û'),
        ('I', '\u{0307}') => Some('İ'),
        ('i', '\u{0307}') => Some('i'),
        _ => None,
    }
}

fn turkish_single_letter(token: &str) -> bool {
    let mut characters = token.chars();
    let Some(value) = characters.next() else {
        return false;
    };
    characters.next().is_none()
        && matches!(
            value,
            'a'..='z'
                | 'A'..='Z'
                | 'ç'
                | 'Ç'
                | 'ğ'
                | 'Ğ'
                | 'ı'
                | 'İ'
                | 'ö'
                | 'Ö'
                | 'ş'
                | 'Ş'
                | 'ü'
                | 'Ü'
        )
}

fn turkish_lower(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    for value in input.chars() {
        match value {
            'I' => output.push('ı'),
            'İ' => output.push('i'),
            _ => output.extend(value.to_lowercase()),
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::{
        numeric_time_with_suffix_analysis, trailing_emphasis_candidate,
        turkish_composition_candidate,
    };

    #[test]
    fn clock_suffix_fallback_is_exact() {
        let analysis = numeric_time_with_suffix_analysis("14:30:05'te").expect("clock suffix");
        assert_eq!(
            analysis
                .morphemes
                .iter()
                .map(|morpheme| morpheme.surface.as_str())
                .collect::<String>(),
            "14:30:05te"
        );
        assert!(numeric_time_with_suffix_analysis("oran'la").is_none());
    }

    #[test]
    fn normalization_candidates_preserve_original_boundaries() {
        let composed = turkish_composition_candidate("g\u{0306}eldi").expect("composition");
        assert_eq!(composed.text, "ğeldi");
        assert_eq!(
            composed.original_boundary(composed.text.len()),
            Some("g\u{0306}eldi".len())
        );
        let emphasis = trailing_emphasis_candidate("tamamdırrr").expect("emphasis");
        assert_eq!(emphasis.text, "tamamdır");
        assert_eq!(
            emphasis.original_boundary(emphasis.text.len()),
            Some("tamamdırrr".len())
        );
    }
}
