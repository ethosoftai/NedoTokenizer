//! High-precision explicit and unmarked code-span detection.

use nedo_core::CodeScanHints;
use nedo_format::ByteSpan;

const STRONG_CODE_SCORE: i16 = 7;
const MODERATE_CODE_SCORE: i16 = 3;
const ADJACENT_CODE_SCORE: i16 = 8;
const CONTINUATION_DISTANCE: usize = 2;

/// Finds explicit fenced/inline code and high-confidence unmarked code blocks.
///
/// Explicit backtick regions are always retained. Unmarked detection is line based:
/// one strong code-shaped line or two adjacent moderate code-shaped lines establish
/// a block, and at most two delimiter/comment continuation lines are attached on
/// either side. The lexical scanner supplies whole-document presence hints, so no
/// fixed prefix window is needed and ordinary prose avoids a second byte scan.
/// Returned spans are ordered, merged, non-overlapping, and byte exact.
#[must_use]
pub fn auto_code_spans(raw: &[u8], hints: CodeScanHints) -> Vec<ByteSpan> {
    let mut spans = if hints.has_backtick() {
        explicit_code_spans(raw)
    } else {
        Vec::new()
    };
    if hints.may_contain_unmarked_code() {
        spans.extend(inferred_code_spans(raw));
    }
    if hints.may_contain_inline_artifact() || hints.may_contain_unmarked_code() {
        spans.extend(inline_artifact_spans(raw));
    }
    merge_spans(spans)
}

fn inline_artifact_spans(raw: &[u8]) -> Vec<ByteSpan> {
    let mut spans = Vec::new();
    let mut index = 0_usize;
    while index < raw.len() {
        while index < raw.len() && inline_separator(raw[index]) {
            index += 1;
        }
        let start = index;
        while index < raw.len() && !inline_separator(raw[index]) {
            index += 1;
        }
        let mut left = start;
        let mut right = index;
        while left < right && matches!(raw[left], b'(' | b'[' | b'{' | b'<' | b'"' | b'\'') {
            left += 1;
        }
        while right > left
            && matches!(
                raw[right - 1],
                b'.' | b',' | b';' | b'!' | b'?' | b')' | b']' | b'}' | b'>' | b'"' | b'\''
            )
        {
            right -= 1;
        }
        let candidate = &raw[left..right];
        if !candidate.is_empty()
            && (looks_like_url(candidate)
                || looks_like_email(candidate)
                || looks_like_inline_assignment(candidate)
                || looks_like_iso_timestamp(candidate)
                || looks_like_programming_number(candidate))
        {
            if let (Ok(start), Ok(end)) = (u64::try_from(left), u64::try_from(right)) {
                spans.push(ByteSpan { start, end });
            }
        }
    }
    spans
}

#[inline]
const fn inline_separator(byte: u8) -> bool {
    byte.is_ascii_whitespace() || byte < 0x20 || byte == 0x7f || matches!(byte, b'`')
}

fn looks_like_url(bytes: &[u8]) -> bool {
    [b"http://".as_slice(), b"https://", b"ftp://", b"www."]
        .iter()
        .any(|prefix| bytes.starts_with(prefix))
        && bytes.len() > 8
}

fn looks_like_email(bytes: &[u8]) -> bool {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return false;
    };
    let mut parts = text.split('@');
    let Some(local) = parts.next() else {
        return false;
    };
    let Some(domain) = parts.next() else {
        return false;
    };
    if parts.next().is_some()
        || local.is_empty()
        || domain.is_empty()
        || !domain.contains('.')
        || domain.starts_with('.')
        || domain.ends_with('.')
    {
        return false;
    }
    local
        .chars()
        .all(|value| value.is_alphanumeric() || matches!(value, '.' | '_' | '+' | '-'))
        && domain
            .chars()
            .all(|value| value.is_alphanumeric() || matches!(value, '.' | '-'))
}

fn looks_like_inline_assignment(bytes: &[u8]) -> bool {
    let Some(equal) = bytes.iter().position(|byte| *byte == b'=') else {
        return false;
    };
    if equal == 0
        || equal + 1 >= bytes.len()
        || bytes.get(equal + 1) == Some(&b'=')
        || bytes[..equal].ends_with(b"!")
        || bytes[..equal].ends_with(b"<")
        || bytes[..equal].ends_with(b">")
    {
        return false;
    }
    let lhs = &bytes[..equal];
    let valid_identifier = lhs
        .first()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'_')
        && lhs
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(*byte, b'_' | b'.' | b'-'));
    let structured_name = lhs.iter().any(|byte| matches!(*byte, b'_' | b'.' | b'-'))
        || matches!(
            lhs,
            b"id"
                | b"key"
                | b"url"
                | b"path"
                | b"host"
                | b"port"
                | b"status"
                | b"error"
                | b"level"
                | b"time"
                | b"date"
        );
    valid_identifier && structured_name
}

fn looks_like_iso_timestamp(bytes: &[u8]) -> bool {
    if bytes.len() < 19
        || bytes.get(4) != Some(&b'-')
        || bytes.get(7) != Some(&b'-')
        || !matches!(bytes.get(10), Some(b'T' | b't'))
        || bytes.get(13) != Some(&b':')
        || bytes.get(16) != Some(&b':')
    {
        return false;
    }
    for index in [0_usize, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18] {
        if !bytes.get(index).is_some_and(u8::is_ascii_digit) {
            return false;
        }
    }
    bytes[19..].iter().all(|byte| {
        byte.is_ascii_digit() || matches!(*byte, b'.' | b',' | b'+' | b'-' | b':' | b'Z' | b'z')
    })
}

fn looks_like_programming_number(bytes: &[u8]) -> bool {
    let value = bytes
        .strip_prefix(b"+")
        .or_else(|| bytes.strip_prefix(b"-"))
        .unwrap_or(bytes);
    for (prefix, valid) in [
        (b"0x".as_slice(), 16_u8),
        (b"0X".as_slice(), 16),
        (b"0b".as_slice(), 2),
        (b"0B".as_slice(), 2),
        (b"0o".as_slice(), 8),
        (b"0O".as_slice(), 8),
    ] {
        if let Some(digits) = value.strip_prefix(prefix) {
            return !digits.is_empty()
                && digits.iter().all(|byte| {
                    *byte == b'_'
                        || match valid {
                            2 => matches!(*byte, b'0' | b'1'),
                            8 => matches!(*byte, b'0'..=b'7'),
                            _ => byte.is_ascii_hexdigit(),
                        }
                });
        }
    }

    let mut index = 0_usize;
    while value
        .get(index)
        .is_some_and(|byte| byte.is_ascii_digit() || *byte == b'_')
    {
        index += 1;
    }
    if index == 0 {
        return false;
    }
    if value.get(index) == Some(&b'.') {
        index += 1;
        let start = index;
        while value
            .get(index)
            .is_some_and(|byte| byte.is_ascii_digit() || *byte == b'_')
        {
            index += 1;
        }
        if index == start {
            return false;
        }
    }
    let mut specialized = false;
    if matches!(value.get(index), Some(b'e' | b'E')) {
        specialized = true;
        index += 1;
        if matches!(value.get(index), Some(b'+' | b'-')) {
            index += 1;
        }
        let start = index;
        while value
            .get(index)
            .is_some_and(|byte| byte.is_ascii_digit() || *byte == b'_')
        {
            index += 1;
        }
        if index == start {
            return false;
        }
    }
    if matches!(
        value.get(index),
        Some(b'f' | b'F' | b'i' | b'I' | b'u' | b'U')
    ) {
        specialized = true;
        index += 1;
        let start = index;
        while value.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
        if index == start {
            return false;
        }
    }
    specialized && index == value.len()
}

/// Finds explicit fenced and inline backtick code spans.
///
/// Unclosed delimiters are deliberately ignored to avoid turning ordinary prose
/// into code mode. Returned spans are ordered, non-overlapping, and include the
/// delimiters so round-trip metadata remains simple.
#[must_use]
pub fn explicit_code_spans(raw: &[u8]) -> Vec<ByteSpan> {
    let mut spans = Vec::new();
    let mut index = 0_usize;
    while index < raw.len() {
        if raw[index] != b'`' {
            index += 1;
            continue;
        }
        let run = backtick_run(raw, index);
        let delimiter = if run >= 3 { run } else { 1 };
        let content_start = index + delimiter;
        if let Some(close) = find_closing(raw, content_start, delimiter) {
            let end = close + delimiter;
            if let (Ok(start), Ok(end)) = (u64::try_from(index), u64::try_from(end)) {
                spans.push(ByteSpan { start, end });
            }
            index = end;
        } else {
            index += run;
        }
    }
    spans
}

/// Returns deterministic byte cuts for code identifiers.
///
/// Separators become their own pieces; camel-case, acronym-to-word, and
/// letter/digit transitions become boundaries. The returned offsets are
/// strictly inside `text` and sorted.
#[must_use]
pub fn identifier_cuts(text: &str) -> Vec<usize> {
    let chars = text.char_indices().collect::<Vec<_>>();
    let mut cuts = Vec::new();
    for (position, &(index, value)) in chars.iter().enumerate() {
        let end = index + value.len_utf8();
        if matches!(value, '_' | '-') {
            if index > 0 {
                cuts.push(index);
            }
            if end < text.len() {
                cuts.push(end);
            }
            continue;
        }
        if position == 0 {
            continue;
        }
        let previous = chars[position - 1].1;
        let next = chars.get(position + 1).map(|entry| entry.1);
        let camel = previous.is_lowercase() && value.is_uppercase();
        let acronym =
            previous.is_uppercase() && value.is_uppercase() && next.is_some_and(char::is_lowercase);
        let digit_transition = previous.is_numeric() != value.is_numeric()
            && (previous.is_alphanumeric() && value.is_alphanumeric());
        if camel || acronym || digit_transition {
            cuts.push(index);
        }
    }
    cuts.sort_unstable();
    cuts.dedup();
    cuts.retain(|cut| *cut > 0 && *cut < text.len());
    cuts
}

#[derive(Clone, Copy)]
struct CodeLine {
    start: usize,
    end: usize,
    score: i16,
    hard: bool,
    weak: bool,
}

fn code_lines(raw: &[u8]) -> Vec<CodeLine> {
    let mut lines = Vec::new();
    let mut start = 0_usize;
    while start < raw.len() {
        let end = raw[start..]
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(raw.len(), |relative| start + relative + 1);
        let content_end = raw[start..end]
            .iter()
            .rposition(|byte| !matches!(*byte, b'\r' | b'\n'))
            .map_or(start, |relative| start + relative + 1);
        let analysis = analyze_code_line(&raw[start..content_end]);
        lines.push(CodeLine {
            start,
            end,
            score: analysis.score,
            hard: analysis.hard,
            weak: analysis.weak,
        });
        start = end;
    }
    lines
}

fn inferred_code_spans(raw: &[u8]) -> Vec<ByteSpan> {
    let mut lines = code_lines(raw);
    if lines.is_empty() {
        return Vec::new();
    }
    if let Some(line) = lines.iter().find(|line| !line_is_blank(raw, **line)) {
        let content = line_content(raw, *line);
        if trim_ascii(content).starts_with(b"#!") {
            if let Ok(end) = u64::try_from(raw.len()) {
                return vec![ByteSpan { start: 0, end }];
            }
        }
    }

    let mut selected = lines
        .iter()
        .map(|line| line.hard || line.score >= STRONG_CODE_SCORE)
        .collect::<Vec<_>>();
    for index in 0..lines.len().saturating_sub(1) {
        let current = lines[index].score;
        let next = lines[index + 1].score;
        if current >= MODERATE_CODE_SCORE
            && next >= MODERATE_CODE_SCORE
            && current.saturating_add(next) >= ADJACENT_CODE_SCORE
        {
            selected[index] = true;
            selected[index + 1] = true;
        }
    }

    let mut first_anchor = None;
    let mut last_anchor = None;
    let mut anchors = 0_usize;
    let mut nonblank = 0_usize;
    for (index, line) in lines.iter().copied().enumerate() {
        if !line_is_blank(raw, line) {
            nonblank += 1;
        }
        if line.hard || line.score >= MODERATE_CODE_SCORE {
            first_anchor.get_or_insert(index);
            last_anchor = Some(index);
            anchors += 1;
        }
    }
    if anchors >= 3 && nonblank > 0 && anchors.saturating_mul(100) >= nonblank.saturating_mul(35) {
        if let (Some(first), Some(last)) = (first_anchor, last_anchor) {
            for index in first..=last {
                if selected[index]
                    || line_is_safe_continuation(raw, lines[index])
                    || !line_is_natural_prose(raw, lines[index])
                {
                    selected[index] = true;
                }
            }
        }
    }

    for index in 0..lines.len() {
        if !selected[index] {
            continue;
        }
        for distance in 1..=CONTINUATION_DISTANCE {
            let Some(previous) = index.checked_sub(distance) else {
                break;
            };
            if !line_is_safe_continuation(raw, lines[previous]) {
                break;
            }
            selected[previous] = true;
        }
        for distance in 1..=CONTINUATION_DISTANCE {
            let next = index + distance;
            if next >= lines.len() || !line_is_safe_continuation(raw, lines[next]) {
                break;
            }
            selected[next] = true;
        }
    }

    let mut spans = Vec::new();
    let mut index = 0_usize;
    while index < lines.len() {
        if !selected[index] {
            index += 1;
            continue;
        }
        let start = lines[index].start;
        let mut end = lines[index].end;
        index += 1;
        while index < lines.len() && selected[index] {
            end = lines[index].end;
            index += 1;
        }
        if let (Ok(start), Ok(end)) = (u64::try_from(start), u64::try_from(end)) {
            spans.push(ByteSpan { start, end });
        }
    }
    lines.clear();
    spans
}

#[derive(Clone, Copy, Default)]
struct CodeSignals {
    semicolon: bool,
    brace: bool,
    open_paren: bool,
    close_paren: bool,
    open_bracket: bool,
    close_bracket: bool,
    underscore: bool,
    camel: bool,
    code_punctuation: bool,
    operators: u8,
}

#[derive(Clone, Copy, Default)]
struct LineAnalysis {
    score: i16,
    hard: bool,
    weak: bool,
}

fn analyze_code_line(raw: &[u8]) -> LineAnalysis {
    let stripped_bytes = trim_ascii(raw);
    if stripped_bytes.is_empty() {
        return LineAnalysis {
            weak: true,
            ..LineAnalysis::default()
        };
    }
    if stripped_bytes.starts_with(b"```")
        || (stripped_bytes.starts_with(b"`") && stripped_bytes.ends_with(b"`"))
    {
        return LineAnalysis::default();
    }

    let indented = raw.first().is_some_and(u8::is_ascii_whitespace);
    let cheap_candidate = starts_with_code_keyword_bytes(stripped_bytes)
        || stripped_bytes.iter().any(|byte| {
            matches!(
                *byte,
                b'=' | b';'
                    | b'{'
                    | b'}'
                    | b'('
                    | b')'
                    | b'['
                    | b']'
                    | b'<'
                    | b'>'
                    | b':'
                    | b'_'
                    | b'#'
                    | b'/'
                    | b'$'
            )
        });
    if !cheap_candidate {
        return LineAnalysis {
            weak: stripped_bytes.starts_with(b"* ")
                || stripped_bytes.starts_with(b"*/")
                || stripped_bytes.starts_with(b"# "),
            ..LineAnalysis::default()
        };
    }

    let Ok(stripped) = std::str::from_utf8(stripped_bytes) else {
        return LineAnalysis::default();
    };
    let signals = scan_code_signals(stripped_bytes);
    let hard_start = starts_with_any(
        stripped,
        &[
            "#!",
            "#[",
            "//",
            "/*",
            "#include",
            "using namespace",
            "<?",
            "<!DOCTYPE",
            "<html",
            "</",
        ],
    );
    let keyword_start = starts_with_code_keyword(stripped);
    let assignment = looks_like_assignment(stripped);
    let structured = match stripped_bytes.first().copied() {
        Some(b'"') => looks_like_json_key(stripped),
        Some(b'{') => stripped.contains(':') && stripped.contains('"'),
        Some(b'[') => looks_like_section(stripped),
        Some(b'<') => looks_like_markup_tag(stripped),
        Some(b'&') => looks_like_escaped_markup_tag(stripped),
        Some(b'.' | b'#' | b'@') => looks_like_css_selector(stripped),
        _ => assignment && looks_like_toml_assignment(stripped),
    };
    let hard = hard_start || keyword_start || structured;
    let mut score = if hard { 4_i16 } else { 0_i16 };
    score = score.saturating_add(i16::from(signals.operators.min(2)) * 2);
    if assignment {
        score = score.saturating_add(4);
    }
    if signals.open_paren && signals.close_paren && looks_like_call(stripped) {
        score = score.saturating_add(6);
    }
    if signals.semicolon {
        score = score.saturating_add(2);
    }
    if signals.brace {
        score = score.saturating_add(1);
    }
    if signals.open_paren && signals.close_paren {
        score = score.saturating_add(1);
    }
    if signals.open_bracket && signals.close_bracket {
        score = score.saturating_add(1);
    }
    if signals.underscore && stripped_bytes.iter().any(|byte| is_identifier_start(*byte)) {
        score = score.saturating_add(1);
    }
    if signals.camel {
        score = score.saturating_add(1);
    }

    if !hard && score >= MODERATE_CODE_SCORE {
        let (word_count, punctuation_count, has_turkish) = prose_stats(stripped);
        if word_count >= 12 {
            score = score.saturating_sub(10);
        } else if word_count >= 9 {
            score = score.saturating_sub(6);
        }
        if word_count >= 4 && has_turkish {
            score = score.saturating_sub(if word_count >= 8 { 6 } else { 3 });
        } else if word_count >= 6 && punctuation_count <= 2 {
            score = score.saturating_sub(3);
        }
        if stripped.ends_with(['.', '!', '?'])
            && !stripped.ends_with(");")
            && !stripped.ends_with("};")
        {
            score = score.saturating_sub(2);
        }
    }

    let delimiter_only = stripped_bytes
        .iter()
        .all(|byte| matches!(*byte, b'{' | b'}' | b'[' | b']' | b'(' | b')' | b',' | b';'));
    let comment_continuation = starts_with_any(stripped, &["//", "/*", "*/", "* ", "# "]);
    let suffix_continuation =
        stripped.ends_with(['{', '(', '[', ',', ';']) || stripped.ends_with("=>");
    let weak = delimiter_only
        || comment_continuation
        || (indented && signals.code_punctuation)
        || suffix_continuation
        || structured;
    LineAnalysis { score, hard, weak }
}

fn scan_code_signals(bytes: &[u8]) -> CodeSignals {
    let mut signals = CodeSignals::default();
    let mut previous = None::<u8>;
    for (index, byte) in bytes.iter().copied().enumerate() {
        match byte {
            b';' => signals.semicolon = true,
            b'{' | b'}' => signals.brace = true,
            b'(' => signals.open_paren = true,
            b')' => signals.close_paren = true,
            b'[' => signals.open_bracket = true,
            b']' => signals.close_bracket = true,
            b'_' => signals.underscore = true,
            _ => {}
        }
        if matches!(
            byte,
            b'=' | b';' | b'{' | b'}' | b'(' | b')' | b'[' | b']' | b'<' | b'>' | b':'
        ) {
            signals.code_punctuation = true;
        }
        if previous.is_some_and(|before| before.is_ascii_lowercase() && byte.is_ascii_uppercase()) {
            signals.camel = true;
        }
        if let Some(next) = bytes.get(index + 1).copied() {
            if is_code_operator_pair(byte, next) {
                signals.operators = signals.operators.saturating_add(1).min(2);
            }
        }
        previous = Some(byte);
    }
    signals
}

fn is_code_operator_pair(first: u8, second: u8) -> bool {
    matches!(
        (first, second),
        (b'=', b'>')
            | (b':', b':')
            | (b'-', b'>')
            | (b'+', b'=')
            | (b'-', b'=')
            | (b'*', b'=')
            | (b'/', b'=')
            | (b'%', b'=')
            | (b'=', b'=')
            | (b'!', b'=')
            | (b'<', b'=')
            | (b'>', b'=')
            | (b'&', b'&')
            | (b'|', b'|')
            | (b'?', b'?')
            | (b'?', b'.')
            | (b':', b'=')
            | (b'<', b'<')
            | (b'>', b'>')
            | (b'.', b'.')
    )
}

fn starts_with_code_keyword_bytes(bytes: &[u8]) -> bool {
    const PREFIXES: &[&[u8]] = &[
        b"pub ",
        b"pub(",
        b"async ",
        b"unsafe ",
        b"extern ",
        b"fn ",
        b"struct ",
        b"enum ",
        b"trait ",
        b"impl ",
        b"mod ",
        b"use ",
        b"let ",
        b"const ",
        b"static ",
        b"type ",
        b"def ",
        b"class ",
        b"import ",
        b"from ",
        b"function ",
        b"var ",
        b"package ",
        b"namespace ",
        b"interface ",
        b"export ",
        b"if ",
        b"for ",
        b"while ",
        b"match ",
        b"loop ",
        b"else",
        b"return ",
        b"break",
        b"continue",
        b"pass",
        b"raise ",
        b"try",
        b"except",
        b"finally",
        b"with ",
        b"yield ",
        b"await ",
        b"SELECT ",
        b"INSERT ",
        b"UPDATE ",
        b"DELETE ",
        b"CREATE ",
        b"ALTER ",
        b"WITH ",
        b"#!",
        b"#[",
        b"//",
        b"/*",
        b"#include",
        b"using namespace",
        b"<?",
        b"<!DOCTYPE",
        b"<html",
        b"</",
    ];
    PREFIXES.iter().any(|prefix| bytes.starts_with(prefix))
}

fn prose_stats(text: &str) -> (usize, usize, bool) {
    let mut words = 0_usize;
    let mut punctuation = 0_usize;
    let mut in_word = false;
    let mut has_turkish = false;
    for value in text.chars() {
        if value.is_alphabetic() {
            if !in_word {
                words += 1;
                in_word = true;
            }
            has_turkish |= is_turkish_letter(value);
        } else {
            in_word = false;
            if !value.is_alphanumeric() && !value.is_whitespace() {
                punctuation += 1;
            }
        }
    }
    (words, punctuation, has_turkish)
}

fn starts_with_code_keyword(text: &str) -> bool {
    let mut value = text;
    loop {
        let trimmed = value.trim_start();
        if let Some(rest) = trimmed.strip_prefix("pub ") {
            value = rest;
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("pub(") {
            if let Some(end) = rest.find(')') {
                value = &rest[end + 1..];
                continue;
            }
        }
        let mut stripped_prefix = false;
        for prefix in ["async ", "unsafe "] {
            if let Some(rest) = trimmed.strip_prefix(prefix) {
                value = rest;
                stripped_prefix = true;
                break;
            }
        }
        if stripped_prefix {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("extern \"") {
            if let Some(end) = rest.find('"') {
                value = &rest[end + 1..];
                continue;
            }
        }
        if let Some(rest) = trimmed.strip_prefix("extern ") {
            value = rest;
            continue;
        }
        value = trimmed;
        break;
    }

    let declaration = [
        "fn",
        "struct",
        "enum",
        "trait",
        "impl",
        "mod",
        "use",
        "let",
        "const",
        "static",
        "type",
        "macro_rules",
        "def",
        "class",
        "import",
        "from",
        "function",
        "var",
        "package",
        "namespace",
        "interface",
        "export",
    ]
    .iter()
    .any(|keyword| starts_with_word(value, keyword));
    if declaration {
        return true;
    }
    if [
        "return", "break", "continue", "pass", "raise", "yield", "await",
    ]
    .iter()
    .any(|keyword| starts_with_word(value, keyword))
    {
        return true;
    }
    let control = [
        "if", "for", "while", "match", "loop", "else", "try", "except", "finally", "with",
    ]
    .iter()
    .any(|keyword| starts_with_word(value, keyword));
    if control
        && (value.contains('{')
            || value.contains(':')
            || value.contains(';')
            || value.contains('(')
            || value.contains("=>"))
    {
        return true;
    }
    [
        "SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "WITH",
    ]
    .iter()
    .any(|keyword| starts_with_word(value, keyword))
}

fn looks_like_assignment(text: &str) -> bool {
    let bytes = text.as_bytes();
    let Some(mut index) = bytes
        .first()
        .copied()
        .filter(|byte| is_identifier_start(*byte))
        .map(|_| 1)
    else {
        return false;
    };
    while index < bytes.len() && is_assignment_lhs_byte(bytes[index]) {
        index += 1;
    }
    while index < bytes.len() && bytes[index].is_ascii_whitespace() {
        index += 1;
    }
    for operator in [
        b"+=".as_slice(),
        b"-=".as_slice(),
        b"*=".as_slice(),
        b"/=".as_slice(),
        b"%=".as_slice(),
        b"=".as_slice(),
    ] {
        if bytes[index..].starts_with(operator) {
            let after = index + operator.len();
            return bytes.get(after).is_some_and(|byte| *byte != b'=');
        }
    }
    false
}

fn looks_like_toml_assignment(text: &str) -> bool {
    if !looks_like_assignment(text) {
        return false;
    }
    let Some(equal) = text.find('=') else {
        return false;
    };
    let rhs = text[equal + 1..].trim_start();
    rhs.starts_with(['"', '\'', '[', '{', '+', '-'])
        || rhs
            .chars()
            .next()
            .is_some_and(|value| value.is_ascii_digit())
        || starts_with_word(rhs, "true")
        || starts_with_word(rhs, "false")
}

fn looks_like_json_key(text: &str) -> bool {
    if !text.starts_with('"') {
        return false;
    }
    let mut escaped = false;
    for (index, value) in text.char_indices().skip(1) {
        if escaped {
            escaped = false;
            continue;
        }
        if value == '\\' {
            escaped = true;
            continue;
        }
        if value == '"' {
            return text[index + value.len_utf8()..]
                .trim_start()
                .starts_with(':');
        }
    }
    false
}

fn looks_like_section(text: &str) -> bool {
    let inner = if let Some(value) = text
        .strip_prefix("[[")
        .and_then(|value| value.strip_suffix("]]"))
    {
        value
    } else if let Some(value) = text
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
    {
        value
    } else {
        return false;
    };
    !inner.is_empty()
        && inner
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b':' | b'-'))
}

fn looks_like_escaped_markup_tag(text: &str) -> bool {
    let Some(close) = text.find("&gt;") else {
        return false;
    };
    if !text.starts_with("&lt;") {
        return false;
    }
    let suffix = text[close + 4..].trim_start();
    suffix.is_empty() || suffix.starts_with("&lt;") || suffix.contains("&lt;/")
}

fn looks_like_markup_tag(text: &str) -> bool {
    let value = text.strip_prefix("</").or_else(|| text.strip_prefix('<'));
    value.is_some_and(|rest| {
        rest.bytes()
            .next()
            .is_some_and(|byte| byte.is_ascii_alphabetic())
            && text.contains('>')
    })
}

fn looks_like_css_selector(text: &str) -> bool {
    (text.starts_with('.') || text.starts_with('#') || text.starts_with('@'))
        && text.contains('{')
        && text.bytes().any(|byte| byte.is_ascii_alphabetic())
}

fn looks_like_call(text: &str) -> bool {
    let bytes = text.as_bytes();
    let mut index = 0_usize;
    while index < bytes.len() {
        let valid_boundary =
            index == 0 || (bytes[index - 1].is_ascii() && !is_identifier_byte(bytes[index - 1]));
        if !valid_boundary || !is_identifier_start(bytes[index]) {
            index += 1;
            continue;
        }
        let identifier_start = index;
        index += 1;
        while index < bytes.len() && is_identifier_byte(bytes[index]) {
            index += 1;
        }
        let identifier = &bytes[identifier_start..index];
        let path_member =
            identifier_start > 0 && matches!(bytes[identifier_start - 1], b'.' | b':');
        let code_name = identifier
            .first()
            .is_some_and(|byte| byte.is_ascii_lowercase() || matches!(*byte, b'_' | b'$'))
            || identifier.contains(&b'_')
            || identifier
                .windows(2)
                .any(|pair| pair[0].is_ascii_lowercase() && pair[1].is_ascii_uppercase())
            || path_member
            || matches!(identifier, b"Some" | b"None" | b"Ok" | b"Err");
        if !code_name {
            continue;
        }
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        if bytes.get(index) == Some(&b'!') {
            index += 1;
        }
        if bytes.get(index) != Some(&b'(') {
            continue;
        }
        let Some(close) = bytes[index + 1..].iter().position(|byte| *byte == b')') else {
            continue;
        };
        let suffix = text[index + close + 2..].trim_start();
        if suffix.is_empty()
            || suffix.starts_with('{')
            || suffix.starts_with(';')
            || suffix.starts_with("->")
            || suffix.starts_with("=>")
            || suffix.starts_with('?')
        {
            return true;
        }
        index += close + 2;
    }
    false
}

fn starts_with_any(text: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|prefix| text.starts_with(prefix))
}

fn starts_with_word(text: &str, word: &str) -> bool {
    let Some(rest) = text.strip_prefix(word) else {
        return false;
    };
    rest.chars()
        .next()
        .is_none_or(|value| !value.is_alphanumeric() && value != '_')
}

fn is_turkish_letter(value: char) -> bool {
    matches!(
        value,
        'ç' | 'ğ' | 'ı' | 'ö' | 'ş' | 'ü' | 'Ç' | 'Ğ' | 'İ' | 'Ö' | 'Ş' | 'Ü'
    )
}

fn is_identifier_start(value: u8) -> bool {
    value.is_ascii_alphabetic() || matches!(value, b'_' | b'$')
}

fn is_identifier_byte(value: u8) -> bool {
    value.is_ascii_alphanumeric() || matches!(value, b'_' | b'$')
}

fn is_assignment_lhs_byte(value: u8) -> bool {
    is_identifier_byte(value) || matches!(value, b'.' | b'[' | b']' | b'-')
}

fn line_content(raw: &[u8], line: CodeLine) -> &[u8] {
    let mut end = line.end;
    while end > line.start && matches!(raw[end - 1], b'\r' | b'\n') {
        end -= 1;
    }
    &raw[line.start..end]
}

fn line_is_safe_continuation(raw: &[u8], line: CodeLine) -> bool {
    line.weak && !line_is_natural_prose(raw, line)
}

fn line_is_natural_prose(raw: &[u8], line: CodeLine) -> bool {
    let bytes = trim_ascii(line_content(raw, line));
    if bytes.is_empty()
        || starts_with_code_keyword_bytes(bytes)
        || bytes.starts_with(b"//")
        || bytes.starts_with(b"/*")
        || bytes.starts_with(b"* ")
        || bytes.starts_with(b"# ")
        || bytes.starts_with(b"&lt;")
        || bytes.starts_with(b"<")
    {
        return false;
    }
    let Ok(text) = std::str::from_utf8(bytes) else {
        return false;
    };
    if text.ends_with([';', '{', '}', ','])
        || text.ends_with("=>")
        || text.contains(" = ")
        || text.contains("::")
        || text.contains("->")
    {
        return false;
    }
    let (words, punctuation, has_turkish) = prose_stats(text);
    (has_turkish && words >= 2)
        || words >= 8
        || (words >= 1 && punctuation <= 8 && text.ends_with(['.', '!', '?']))
}

fn line_is_blank(raw: &[u8], line: CodeLine) -> bool {
    trim_ascii(line_content(raw, line)).is_empty()
}

fn trim_ascii(mut value: &[u8]) -> &[u8] {
    while value.first().is_some_and(u8::is_ascii_whitespace) {
        value = &value[1..];
    }
    while value.last().is_some_and(u8::is_ascii_whitespace) {
        value = &value[..value.len() - 1];
    }
    value
}

fn merge_spans(mut spans: Vec<ByteSpan>) -> Vec<ByteSpan> {
    spans.sort_unstable_by_key(|span| (span.start, span.end));
    let mut merged = Vec::<ByteSpan>::with_capacity(spans.len());
    for span in spans {
        if span.start >= span.end {
            continue;
        }
        if let Some(previous) = merged.last_mut() {
            if span.start <= previous.end {
                previous.end = previous.end.max(span.end);
                continue;
            }
        }
        merged.push(span);
    }
    merged
}

fn backtick_run(raw: &[u8], start: usize) -> usize {
    raw[start..]
        .iter()
        .take_while(|byte| **byte == b'`')
        .count()
}

fn find_closing(raw: &[u8], mut index: usize, delimiter: usize) -> Option<usize> {
    while index < raw.len() {
        if raw[index] == b'`' && backtick_run(raw, index) >= delimiter {
            return Some(index);
        }
        index += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{auto_code_spans, explicit_code_spans, inline_artifact_spans};
    use nedo_core::scan;

    fn automatic_spans(
        raw: &[u8],
    ) -> Result<Vec<nedo_format::ByteSpan>, Box<dyn std::error::Error>> {
        let lexical = scan(raw.to_vec())?;
        Ok(auto_code_spans(raw, lexical.code_hints()))
    }

    fn slices<'a>(
        raw: &'a [u8],
        spans: &[nedo_format::ByteSpan],
    ) -> Result<Vec<&'a [u8]>, Box<dyn std::error::Error>> {
        spans
            .iter()
            .map(|span| {
                let start = usize::try_from(span.start)?;
                let end = usize::try_from(span.end)?;
                Ok(&raw[start..end])
            })
            .collect()
    }

    #[test]
    fn detects_high_confidence_inline_artifacts() {
        let raw = "mail ad.soyad+test@örnek.istanbul url https://örnek.istanbul/a?x=1&y=çığ log request_id=abc-123 2026-08-04T00:03:12.345+03:00 0xDEAD_BEEF 1e-9 3.14f64".as_bytes();
        let spans = inline_artifact_spans(raw);
        let values = spans
            .iter()
            .map(|span| &raw[span.start as usize..span.end as usize])
            .collect::<Vec<_>>();
        assert!(values.contains(&"ad.soyad+test@örnek.istanbul".as_bytes()));
        assert!(values.contains(&"https://örnek.istanbul/a?x=1&y=çığ".as_bytes()));
        assert!(values.contains(&b"request_id=abc-123".as_slice()));
        assert!(values.contains(&b"2026-08-04T00:03:12.345+03:00".as_slice()));
        assert!(values.contains(&b"0xDEAD_BEEF".as_slice()));
        assert!(values.contains(&b"1e-9".as_slice()));
        assert!(values.contains(&b"3.14f64".as_slice()));
    }

    #[test]
    fn detects_only_closed_explicit_code_regions() -> Result<(), Box<dyn std::error::Error>> {
        let raw = b"metin `x += 1` ve ```rust\nfn main() {}\n``` son";
        let spans = explicit_code_spans(raw);
        assert_eq!(spans.len(), 2);
        let detected = slices(raw, &spans)?;
        assert_eq!(detected[0], b"`x += 1`");
        assert!(std::str::from_utf8(detected[1])?.starts_with("```rust"));
        assert!(explicit_code_spans(b"normal ` kapanmamis").is_empty());
        Ok(())
    }

    #[test]
    fn detects_unmarked_code_without_absorbing_prose() -> Result<(), Box<dyn std::error::Error>> {
        let raw = b"Turkce aciklama.\nfn main() {\n    let foo_bar = 42;\n    println!(\"{foo_bar}\");\n}\nSon cumle.\n";
        let detected = slices(raw, &automatic_spans(raw)?)?;
        assert_eq!(
            detected,
            vec![b"fn main() {\n    let foo_bar = 42;\n    println!(\"{foo_bar}\");\n}\n"]
        );
        Ok(())
    }

    #[test]
    fn detects_small_assignment_and_call_blocks() -> Result<(), Box<dyn std::error::Error>> {
        let raw = b"x = value\nprint(x)\n";
        assert_eq!(slices(raw, &automatic_spans(raw)?)?, vec![raw.as_slice()]);
        assert_eq!(
            slices(b"foo_bar += 42\n", &automatic_spans(b"foo_bar += 42\n")?)?,
            vec![b"foo_bar += 42\n"]
        );
        Ok(())
    }

    #[test]
    fn keeps_turkish_prose_and_formula_out_of_code_mode() -> Result<(), Box<dyn std::error::Error>>
    {
        for raw in [
            "geliyom gidiyom, birazdan dönerim.",
            "Lucian Pintilie (d. 9 Kasım 1933; Bükreş), Rumen film yönetmenidir.",
            "O=PCl3 + 3 ROH → O=P(OR)3 + 3 HCl",
            "Aşağıdaki şehirler ele geçirildi: Akhisar (22 Haziran); Bursa (8 Temmuz).",
            "- Bu yalnızca normal bir listedir.",
            "Birinci Sekreter: (genel sekreter 1948-1953);\nBaşkan (sadece resmi olarak, 1950 yılında kaldırıldı);",
            "Terk (2008)",
            "• \"Weeping Cherry\" 28 Nisan 2015 - Barbes Records (ABD)",
        ] {
            assert!(automatic_spans(raw.as_bytes())?.is_empty(), "false code span for {raw}");
        }
        Ok(())
    }

    #[test]
    fn shebang_marks_the_complete_document() -> Result<(), Box<dyn std::error::Error>> {
        let raw = b"#!/bin/sh\necho geliyom\nexit 0\n";
        assert_eq!(slices(raw, &automatic_spans(raw)?)?, vec![raw.as_slice()]);
        Ok(())
    }

    #[test]
    fn detects_common_unmarked_programming_languages() -> Result<(), Box<dyn std::error::Error>> {
        for raw in [
            b"def add(x, y):
    return x + y
"
            .as_slice(),
            b"const result = foo.bar(42);
"
            .as_slice(),
            br#"{"name":"nedo","enabled":true}
"#
            .as_slice(),
            br#"[package]
name = "nedo"
"#
            .as_slice(),
            b"SELECT id FROM users WHERE active = true;
"
            .as_slice(),
            br#"<div class="card">hello</div>
"#
            .as_slice(),
            b"<div>hello</div>\n".as_slice(),
            b".card { display: grid; }
"
            .as_slice(),
            b"#!/bin/sh
printf '%s\n' hello
"
            .as_slice(),
        ] {
            let spans = automatic_spans(raw)?;
            assert!(
                !spans.is_empty(),
                "code language was not detected: {:?}",
                String::from_utf8_lossy(raw)
            );
            let detected = slices(raw, &spans)?;
            assert_eq!(
                detected.iter().map(|value| value.len()).sum::<usize>(),
                raw.len(),
                "partial code detection for {:?}: {:?}",
                String::from_utf8_lossy(raw),
                detected
            );
        }
        Ok(())
    }

    #[test]
    fn explicit_code_anywhere_is_detected() -> Result<(), Box<dyn std::error::Error>> {
        let mut raw = vec![b'a'; 800];
        raw.extend_from_slice(
            b"
`foo_bar += 42`
",
        );
        let spans = automatic_spans(&raw)?;
        assert_eq!(slices(&raw, &spans)?, vec![b"`foo_bar += 42`".as_slice()]);
        Ok(())
    }

    #[test]
    fn unmarked_code_after_the_old_prefix_limit_is_detected(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut raw = vec![b'a'; 800];
        raw.extend_from_slice(b"\nfn main() {\n    let value = 42;\n}\n");
        let spans = automatic_spans(&raw)?;
        let detected = slices(&raw, &spans)?;
        assert_eq!(
            detected,
            vec![b"fn main() {\n    let value = 42;\n}\n".as_slice()]
        );
        Ok(())
    }

    #[test]
    fn escaped_markup_after_the_old_prefix_limit_is_detected(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut raw = vec![b'a'; 800];
        raw.extend_from_slice(b"\n&lt;SyncML version=\"1.2\"&gt;\n&lt;/SyncML&gt;\n");
        let spans = automatic_spans(&raw)?;
        let detected = slices(&raw, &spans)?;
        assert_eq!(
            detected,
            vec![b"&lt;SyncML version=\"1.2\"&gt;\n&lt;/SyncML&gt;\n".as_slice()]
        );
        Ok(())
    }

    #[test]
    fn inline_escaped_tag_does_not_absorb_turkish_text() -> Result<(), Box<dyn std::error::Error>> {
        let raw = "&lt;br&gt;Türkçe açıklama.\n".as_bytes();
        assert!(automatic_spans(raw)?.is_empty());
        Ok(())
    }

    #[test]
    fn dense_markup_does_not_absorb_turkish_prose() -> Result<(), Box<dyn std::error::Error>> {
        let raw = "&lt;a&gt;\nB Grubu.\n&lt;b&gt;\nFinal aşaması.\n&lt;c&gt;\n5.-8.'lik.\n&lt;d&gt;\naçıklama uğruna):\n&lt;e&gt;\n".as_bytes();
        let spans = automatic_spans(raw)?;
        let detected = slices(raw, &spans)?;
        assert_eq!(
            detected,
            vec![
                b"&lt;a&gt;\n".as_slice(),
                b"&lt;b&gt;\n".as_slice(),
                b"&lt;c&gt;\n".as_slice(),
                b"&lt;d&gt;\n".as_slice(),
                b"&lt;e&gt;\n".as_slice(),
            ]
        );
        Ok(())
    }

    #[test]
    fn dense_source_document_keeps_internal_comments_and_literals(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let raw = br#"use std::io;
// explanation
fn main() {
    let message = "hello";
    // internal prose
    println!("{message}");
}
"#;
        assert_eq!(slices(raw, &automatic_spans(raw)?)?, vec![raw.as_slice()]);
        Ok(())
    }

    #[test]
    fn recognizes_chained_rust_declaration_prefixes() -> Result<(), Box<dyn std::error::Error>> {
        for raw in [
            b"pub(crate) async fn run() {}\n".as_slice(),
            b"pub unsafe extern \"C\" fn call() {}\n".as_slice(),
        ] {
            assert_eq!(slices(raw, &automatic_spans(raw)?)?, vec![raw]);
        }
        Ok(())
    }

    #[test]
    fn splits_identifiers_without_changing_bytes() {
        assert_eq!(
            super::identifier_cuts("parseHttpRequest2XX"),
            vec![5, 9, 16, 17]
        );
        assert_eq!(
            super::identifier_cuts("parse_http-header"),
            vec![5, 6, 10, 11]
        );
        assert!(super::identifier_cuts("simple").is_empty());
    }
}
