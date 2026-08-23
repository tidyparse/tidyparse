use std::{
    alloc::{Layout, alloc, dealloc},
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    slice, str,
};

use rg_analysis::{Analysis, DocumentSymbol};
use rg_parse::{Span, parse_source_file};
use rg_syntax::{AstNode, SyntaxKind, ast};
use rg_text::RustEdition;
use serde::Serialize;

const GLANCER_REVISION: &str = "7c68f1afb5ace31026013dee3a75a8ea4cf1684f";

thread_local! {
    static OUTPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EngineInfo {
    name: &'static str,
    revision: &'static str,
    mode: &'static str,
    capabilities: [&'static str; 4],
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AnalysisResponse {
    ok: bool,
    engine: EngineInfo,
    diagnostics: Vec<Diagnostic>,
    symbols: Vec<BrowserSymbol>,
    occurrences: Vec<Occurrence>,
    completions: Vec<Completion>,
    error: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Diagnostic {
    start: u32,
    end: u32,
    severity: &'static str,
    message: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserSymbol {
    name: String,
    kind: String,
    start: u32,
    end: u32,
    selection_start: u32,
    selection_end: u32,
    detail: String,
    children: Vec<BrowserSymbol>,
}

#[derive(Clone)]
struct Declaration {
    name: String,
    canonical_name: String,
    kind: String,
    start: u32,
    end: u32,
    detail: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Occurrence {
    name: String,
    start: u32,
    end: u32,
    declaration_start: Option<u32>,
    declaration_end: Option<u32>,
    kind: String,
    detail: String,
    is_declaration: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Completion {
    label: String,
    kind: String,
    detail: String,
}

fn engine_info() -> EngineInfo {
    EngineInfo {
        name: "Rust Glancer",
        revision: GLANCER_REVISION,
        mode: "syntax-only browser prototype",
        capabilities: [
            "syntax diagnostics",
            "document symbols",
            "same-file definitions",
            "lexical completion",
        ],
    }
}

fn utf16_offset(source: &str, byte_offset: u32) -> u32 {
    let requested = usize::try_from(byte_offset).unwrap_or(source.len());
    let mut boundary = requested.min(source.len());
    while boundary > 0 && !source.is_char_boundary(boundary) {
        boundary -= 1;
    }
    source[..boundary]
        .encode_utf16()
        .count()
        .try_into()
        .unwrap_or(u32::MAX)
}

fn browser_range(source: &str, span: Span) -> (u32, u32) {
    (
        utf16_offset(source, span.text.start),
        utf16_offset(source, span.text.end),
    )
}

fn syntax_range(source: &str, range: rg_syntax::TextRange) -> (u32, u32) {
    (
        utf16_offset(source, u32::from(range.start())),
        utf16_offset(source, u32::from(range.end())),
    )
}

fn source_fragment(source: &str, start: u32, end: u32) -> &str {
    let Ok(mut start) = usize::try_from(start) else {
        return "";
    };
    let Ok(mut end) = usize::try_from(end) else {
        return "";
    };
    start = start.min(source.len());
    end = end.min(source.len()).max(start);
    while start < source.len() && !source.is_char_boundary(start) {
        start += 1;
    }
    while end > start && !source.is_char_boundary(end) {
        end -= 1;
    }
    &source[start..end]
}

fn declaration_detail(source: &str, span: Span) -> String {
    let fragment = source_fragment(source, span.text.start, span.text.end).trim();
    let header = fragment
        .split_once('{')
        .map(|(head, _)| head)
        .unwrap_or(fragment)
        .split(';')
        .next()
        .unwrap_or(fragment);
    let compact = header.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut characters = compact.chars();
    let shortened = characters.by_ref().take(240).collect::<String>();
    if characters.next().is_some() {
        format!("{shortened}…")
    } else {
        compact
    }
}

fn convert_symbol(source: &str, symbol: &DocumentSymbol) -> BrowserSymbol {
    let (start, end) = browser_range(source, symbol.span);
    let (selection_start, selection_end) = browser_range(source, symbol.selection_span);
    BrowserSymbol {
        name: symbol.name.clone(),
        kind: symbol.kind.to_string(),
        start,
        end,
        selection_start,
        selection_end,
        detail: declaration_detail(source, symbol.span),
        children: symbol
            .children
            .iter()
            .map(|child| convert_symbol(source, child))
            .collect(),
    }
}

fn collect_symbol_declarations(symbols: &[BrowserSymbol], declarations: &mut Vec<Declaration>) {
    for symbol in symbols {
        declarations.push(Declaration {
            name: symbol.name.clone(),
            canonical_name: canonical_name(&symbol.name),
            kind: symbol.kind.clone(),
            start: symbol.selection_start,
            end: symbol.selection_end,
            detail: symbol.detail.clone(),
        });
        collect_symbol_declarations(&symbol.children, declarations);
    }
}

fn canonical_name(name: &str) -> String {
    name.strip_prefix("r#").unwrap_or(name).to_owned()
}

fn ast_name_text(name: &ast::Name) -> Option<String> {
    name.ident_token()
        .or_else(|| name.self_token())
        .map(|token| token.text().to_owned())
}

fn ast_name_ref_text(name: &ast::NameRef) -> Option<String> {
    name.ident_token()
        .or_else(|| name.Self_token())
        .or_else(|| name.self_token())
        .map(|token| token.text().to_owned())
}

fn syntax_name_kind(name: &ast::Name) -> &'static str {
    for ancestor in name.syntax().ancestors().skip(1) {
        match ancestor.kind() {
            SyntaxKind::RECORD_FIELD | SyntaxKind::TUPLE_FIELD => return "field",
            SyntaxKind::VARIANT => return "variant",
            SyntaxKind::PARAM | SyntaxKind::SELF_PARAM | SyntaxKind::LET_STMT => return "variable",
            SyntaxKind::FN => return "fn",
            SyntaxKind::STRUCT => return "struct",
            SyntaxKind::ENUM => return "enum",
            SyntaxKind::TRAIT => return "trait",
            SyntaxKind::MODULE => return "module",
            SyntaxKind::TYPE_ALIAS => return "type_alias",
            SyntaxKind::CONST => return "const",
            SyntaxKind::STATIC => return "static",
            _ => {}
        }
    }
    "variable"
}

fn enclosing_detail(source: &str, name: &ast::Name) -> String {
    let range = name.syntax().text_range();
    for ancestor in name.syntax().ancestors().skip(1) {
        if matches!(
            ancestor.kind(),
            SyntaxKind::LET_STMT
                | SyntaxKind::PARAM
                | SyntaxKind::RECORD_FIELD
                | SyntaxKind::VARIANT
                | SyntaxKind::FN
                | SyntaxKind::STRUCT
                | SyntaxKind::ENUM
                | SyntaxKind::TRAIT
                | SyntaxKind::MODULE
                | SyntaxKind::TYPE_ALIAS
                | SyntaxKind::CONST
                | SyntaxKind::STATIC
        ) {
            let ancestor_range = ancestor.text_range();
            let span = Span::from_text_range(ancestor_range);
            return declaration_detail(source, span);
        }
    }
    source_fragment(source, u32::from(range.start()), u32::from(range.end())).to_owned()
}

fn collect_declarations(
    source: &str,
    syntax: &rg_syntax::SourceFile,
    symbols: &[BrowserSymbol],
) -> Vec<Declaration> {
    let mut declarations = Vec::new();
    collect_symbol_declarations(symbols, &mut declarations);
    let mut seen = declarations
        .iter()
        .map(|declaration| (declaration.start, declaration.end))
        .collect::<BTreeSet<_>>();

    for node in syntax.syntax().descendants() {
        let Some(name) = ast::Name::cast(node) else {
            continue;
        };
        let Some(text) = ast_name_text(&name) else {
            continue;
        };
        let (start, end) = syntax_range(source, name.syntax().text_range());
        if !seen.insert((start, end)) {
            continue;
        }
        declarations.push(Declaration {
            canonical_name: canonical_name(&text),
            name: text,
            kind: syntax_name_kind(&name).to_owned(),
            start,
            end,
            detail: enclosing_detail(source, &name),
        });
    }

    declarations.sort_by_key(|declaration| (declaration.start, declaration.end));
    declarations
}

fn resolved_declaration<'a>(
    name: &str,
    offset: u32,
    declarations: &'a [Declaration],
) -> Option<&'a Declaration> {
    let canonical = canonical_name(name);
    let matching = declarations
        .iter()
        .filter(|declaration| declaration.canonical_name == canonical)
        .collect::<Vec<_>>();
    matching
        .iter()
        .copied()
        .filter(|declaration| declaration.start <= offset)
        .max_by_key(|declaration| declaration.start)
        .or_else(|| matching.first().copied())
}

fn collect_occurrences(
    source: &str,
    syntax: &rg_syntax::SourceFile,
    declarations: &[Declaration],
) -> Vec<Occurrence> {
    let mut occurrences = declarations
        .iter()
        .map(|declaration| Occurrence {
            name: declaration.name.clone(),
            start: declaration.start,
            end: declaration.end,
            declaration_start: Some(declaration.start),
            declaration_end: Some(declaration.end),
            kind: declaration.kind.clone(),
            detail: declaration.detail.clone(),
            is_declaration: true,
        })
        .collect::<Vec<_>>();

    for node in syntax.syntax().descendants() {
        let Some(name_ref) = ast::NameRef::cast(node) else {
            continue;
        };
        let Some(name) = ast_name_ref_text(&name_ref) else {
            continue;
        };
        let (start, end) = syntax_range(source, name_ref.syntax().text_range());
        let declaration = resolved_declaration(&name, start, declarations);
        occurrences.push(Occurrence {
            name,
            start,
            end,
            declaration_start: declaration.map(|item| item.start),
            declaration_end: declaration.map(|item| item.end),
            kind: declaration
                .map(|item| item.kind.clone())
                .unwrap_or_else(|| "reference".to_owned()),
            detail: declaration
                .map(|item| item.detail.clone())
                .unwrap_or_else(|| "Unresolved in this single-file syntax model".to_owned()),
            is_declaration: false,
        });
    }

    occurrences.sort_by_key(|occurrence| (occurrence.start, occurrence.end));
    occurrences
}

fn completion_items(declarations: &[Declaration]) -> Vec<Completion> {
    let mut unique = BTreeMap::<String, Completion>::new();
    for declaration in declarations {
        unique
            .entry(declaration.canonical_name.clone())
            .or_insert_with(|| Completion {
                label: declaration.name.clone(),
                kind: declaration.kind.clone(),
                detail: declaration.detail.clone(),
            });
    }
    unique.into_values().collect()
}

fn analyze(source: &str) -> AnalysisResponse {
    let parsed = parse_source_file(source, RustEdition::Edition2024);
    let diagnostics = parsed
        .errors()
        .into_iter()
        .map(|error| {
            let (start, end) = syntax_range(source, error.range());
            Diagnostic {
                start,
                end,
                severity: "error",
                message: error.to_string(),
            }
        })
        .collect();
    let syntax = parsed.tree();
    let symbols = Analysis::document_symbols_from_syntax(&syntax)
        .iter()
        .map(|symbol| convert_symbol(source, symbol))
        .collect::<Vec<_>>();
    let declarations = collect_declarations(source, &syntax, &symbols);
    let occurrences = collect_occurrences(source, &syntax, &declarations);
    let completions = completion_items(&declarations);

    AnalysisResponse {
        ok: true,
        engine: engine_info(),
        diagnostics,
        symbols,
        occurrences,
        completions,
        error: None,
    }
}

fn failure(message: impl Into<String>) -> AnalysisResponse {
    AnalysisResponse {
        ok: false,
        engine: engine_info(),
        diagnostics: Vec::new(),
        symbols: Vec::new(),
        occurrences: Vec::new(),
        completions: Vec::new(),
        error: Some(message.into()),
    }
}

fn write_output(response: &AnalysisResponse) {
    let output = serde_json::to_vec(response).unwrap_or_else(|_| {
        br#"{"ok":false,"error":"Rust Glancer response serialization failed"}"#.to_vec()
    });
    OUTPUT.with(|buffer| *buffer.borrow_mut() = output);
}

#[unsafe(no_mangle)]
pub extern "C" fn tidyparse_rust_alloc(length: usize) -> *mut u8 {
    if length == 0 {
        return std::ptr::NonNull::<u8>::dangling().as_ptr();
    }
    let Ok(layout) = Layout::array::<u8>(length) else {
        return std::ptr::null_mut();
    };
    // SAFETY: The caller returns this allocation through `tidyparse_rust_dealloc` with the same
    // byte length. The WebAssembly host writes no more than `length` bytes into it.
    unsafe { alloc(layout) }
}

#[unsafe(no_mangle)]
pub extern "C" fn tidyparse_rust_dealloc(pointer: *mut u8, length: usize) {
    if length == 0 || pointer.is_null() {
        return;
    }
    let Ok(layout) = Layout::array::<u8>(length) else {
        return;
    };
    // SAFETY: `pointer` was returned by `tidyparse_rust_alloc(length)` and is released once.
    unsafe { dealloc(pointer, layout) };
}

#[unsafe(no_mangle)]
pub extern "C" fn tidyparse_rust_analyze(pointer: *const u8, length: usize) -> i32 {
    if pointer.is_null() && length != 0 {
        write_output(&failure("The browser supplied a null source pointer"));
        return 1;
    }
    // SAFETY: The browser host owns a live allocation of exactly `length` bytes for this call.
    let bytes = unsafe { slice::from_raw_parts(pointer, length) };
    match str::from_utf8(bytes) {
        Ok(source) => {
            write_output(&analyze(source));
            0
        }
        Err(error) => {
            write_output(&failure(format!("Source is not valid UTF-8: {error}")));
            1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tidyparse_rust_output_ptr() -> *const u8 {
    OUTPUT.with(|buffer| buffer.borrow().as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn tidyparse_rust_output_len() -> usize {
    OUTPUT.with(|buffer| buffer.borrow().len())
}

#[unsafe(no_mangle)]
pub extern "C" fn tidyparse_rust_abi_version() -> u32 {
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browser_offsets_use_utf16_code_units() {
        let source = "a🦀b";
        assert_eq!(utf16_offset(source, 1), 1);
        assert_eq!(utf16_offset(source, 2), 1);
        assert_eq!(utf16_offset(source, 5), 3);
        assert_eq!(utf16_offset(source, 6), 4);
    }

    #[test]
    fn glancer_produces_outline_and_syntax_diagnostics() {
        let valid = analyze("struct Point { x: f64 }\nfn main() { let point = Point { x: 1.0 }; }");
        assert!(valid.diagnostics.is_empty());
        assert!(valid.symbols.iter().any(|symbol| symbol.name == "Point"));
        assert!(valid.symbols.iter().any(|symbol| symbol.name == "main"));
        assert!(valid.completions.iter().any(|item| item.label == "point"));

        let invalid = analyze("fn main( {");
        assert!(!invalid.diagnostics.is_empty());
    }
}
