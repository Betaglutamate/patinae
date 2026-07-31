//! Splitting an assistant reply into the spans the panel renders.
//!
//! Models answer in markdown, and Slint has no markdown widget. Rather than
//! build one, this recognises the single distinction that earns its keep in a
//! molecular viewer: **code is not prose**. A fenced block is something you
//! might run or copy, and it wants a monospace face on its own surface;
//! everything else is a sentence and wants to read like one.
//!
//! Inline markers (`**bold**`, `*italic*`) are deliberately left in the text.
//! Stripping them without being able to render the emphasis would lose
//! information, and a stray asterisk is a smaller cost than a silently mangled
//! reply.

use patinae_framework::plugin_ui::PanelMessageBlock;

/// The fence marker. Only backticks — tilde fences are vanishingly rare in
/// model output and supporting them would mean tracking which style opened.
const FENCE: &str = "```";

/// Split `body` into alternating prose and code blocks.
///
/// Fences are recognised only at the start of a line, so a stray triple
/// backtick mid-sentence does not open a code block.
///
/// An **unterminated fence** — the normal state of affairs while a reply is
/// still streaming — closes at the end of the text. The block renders as code
/// from the moment it opens, so a command appears in its final form as it
/// arrives instead of sitting in prose and reflowing into a card when the
/// closing fence lands.
pub fn blocks(body: &str) -> Vec<PanelMessageBlock> {
    let mut blocks = Vec::new();
    let mut prose: Vec<&str> = Vec::new();
    let mut code: Vec<&str> = Vec::new();
    let mut language = String::new();
    let mut in_code = false;

    for line in body.split('\n') {
        match (in_code, line.trim_start().starts_with(FENCE)) {
            // Opening a block: flush whatever prose came before it.
            (false, true) => {
                push_prose(&mut blocks, &mut prose);
                language = line.trim_start()[FENCE.len()..].trim().to_string();
                in_code = true;
            }
            // Closing one.
            (true, true) => {
                push_code(&mut blocks, &mut code, &language);
                language.clear();
                in_code = false;
            }
            (true, false) => code.push(line),
            (false, false) => prose.push(line),
        }
    }

    // Whichever run we were in when the text ran out.
    if in_code {
        push_code(&mut blocks, &mut code, &language);
    } else {
        push_prose(&mut blocks, &mut prose);
    }

    // A message always renders as something, even if it is only the empty
    // string the assistant has produced so far.
    if blocks.is_empty() {
        blocks.push(PanelMessageBlock::prose(String::new()));
    }
    blocks
}

/// Flush pending prose, dropping a run that is only blank lines.
///
/// Those blank lines are the separators around a fence, not content; keeping
/// them would leave an empty paragraph above and below every code block.
fn push_prose(blocks: &mut Vec<PanelMessageBlock>, prose: &mut Vec<&str>) {
    let text = std::mem::take(prose).join("\n");
    if !text.trim().is_empty() {
        blocks.push(PanelMessageBlock::prose(text.trim_matches('\n')));
    }
}

/// Flush a pending code run.
///
/// Unlike prose, an empty code block is kept: a fence the model opened is a
/// statement that code is coming, and showing the empty card as it streams is
/// steadier than having one appear a token later.
fn push_code(blocks: &mut Vec<PanelMessageBlock>, code: &mut Vec<&str>, language: &str) {
    let text = std::mem::take(code).join("\n");
    blocks.push(PanelMessageBlock::code(language, text.trim_matches('\n')));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compact rendering of a block list, so expectations read as data.
    fn shape(body: &str) -> Vec<(String, String)> {
        blocks(body)
            .into_iter()
            .map(|block| match block {
                PanelMessageBlock::Prose(text) => ("prose".to_string(), text),
                PanelMessageBlock::Code { language, text } => {
                    (format!("code:{language}"), text)
                }
            })
            .collect()
    }

    #[test]
    fn plain_prose_is_one_block() {
        assert_eq!(
            shape("Loaded 1crn and coloured it by chain."),
            [(
                "prose".to_string(),
                "Loaded 1crn and coloured it by chain.".to_string()
            )]
        );
    }

    #[test]
    fn a_fenced_block_is_split_out_with_its_language() {
        assert_eq!(
            shape("Run this:\n```python\ncmd.zoom()\n```\nThen look."),
            [
                ("prose".to_string(), "Run this:".to_string()),
                ("code:python".to_string(), "cmd.zoom()".to_string()),
                ("prose".to_string(), "Then look.".to_string()),
            ]
        );
    }

    #[test]
    fn a_fence_without_a_language_still_reads_as_code() {
        assert_eq!(
            shape("```\nload 1crn\n```"),
            [("code:".to_string(), "load 1crn".to_string())]
        );
    }

    #[test]
    fn an_unterminated_fence_renders_as_code_while_it_streams() {
        // The common case mid-reply. Leaving it as prose would make the block
        // reflow into a card the moment the closing fence arrives.
        assert_eq!(
            shape("Here you go:\n```python\ncmd.load("),
            [
                ("prose".to_string(), "Here you go:".to_string()),
                ("code:python".to_string(), "cmd.load(".to_string()),
            ]
        );
    }

    #[test]
    fn a_fence_that_has_only_just_opened_shows_an_empty_card() {
        // One token further on than the test above: the fence is open and no
        // code has arrived yet.
        assert_eq!(
            shape("```python"),
            [("code:python".to_string(), String::new())]
        );
    }

    #[test]
    fn multiple_blocks_alternate_in_order() {
        assert_eq!(
            shape("one\n```\na\n```\ntwo\n```\nb\n```\nthree"),
            [
                ("prose".to_string(), "one".to_string()),
                ("code:".to_string(), "a".to_string()),
                ("prose".to_string(), "two".to_string()),
                ("code:".to_string(), "b".to_string()),
                ("prose".to_string(), "three".to_string()),
            ]
        );
    }

    #[test]
    fn blank_separator_lines_do_not_become_empty_paragraphs() {
        assert_eq!(
            shape("intro\n\n```\ncode\n```\n\noutro"),
            [
                ("prose".to_string(), "intro".to_string()),
                ("code:".to_string(), "code".to_string()),
                ("prose".to_string(), "outro".to_string()),
            ]
        );
    }

    #[test]
    fn a_backtick_run_mid_sentence_does_not_open_a_block() {
        // Only a line-leading fence counts, or inline code would tear the
        // sentence around it in half.
        assert_eq!(
            shape("use ```load``` for that"),
            [("prose".to_string(), "use ```load``` for that".to_string())]
        );
    }

    #[test]
    fn blank_lines_inside_a_code_block_are_kept() {
        // They are part of the program, not separators.
        assert_eq!(
            shape("```python\na = 1\n\nb = 2\n```"),
            [("code:python".to_string(), "a = 1\n\nb = 2".to_string())]
        );
    }

    #[test]
    fn an_empty_body_still_yields_one_block() {
        // A message that renders as nothing at all would collapse its row.
        assert_eq!(shape(""), [("prose".to_string(), String::new())]);
    }

    #[test]
    fn an_indented_fence_still_opens_a_block() {
        // Models routinely indent fences inside list items.
        assert_eq!(
            shape("  ```python\n  cmd.zoom()\n  ```"),
            [("code:python".to_string(), "  cmd.zoom()".to_string())]
        );
    }
}
