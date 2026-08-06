# WDFA Frontier Decoder

When `wdfa != null`, `WGPUv1.kt` skips `enum_words_wor`. Instead, it enumerates and ranks directly over the parse chart/backpointer structure while carrying the WDFA state. The backpointers are treated as a compact expression for the accepted completion language, and the WDFA supplies token costs during enumeration.

## State

A frontier particle stores:

- a stack of chart cells still to expand,
- the emitted token packet,
- the current WDFA state $q$,
- accumulated WDFA cost $c$,
- edit distance $\Delta$ from the accepting root,
- status: active, done, or dead.

Each root is an accepting parse-chart cell from `startIdxs`. Initialization creates one particle per root while the frontier fits, or samples roots if there are more roots than the frontier budget.

## Backpointer Support

For a chart cell $d$, the decoder gets its next alternatives from two sources:

- `dp_in[d]` encodes terminal support for the cell. `count_tms` and `wdfa_nth_lex_tok` enumerate the concrete token ids.
- `bpCount/bpOffset/bpStorage` encodes binary backpointers. Each backpointer is a pair $(l,r)$ representing concatenation, so expansion pushes `r` then `l` onto the stack.

Thus a particle whose stack top is $d$ has immediate support

$$
S(d) = \mathrm{Lex}(d) \cup \mathrm{Bin}(d)
$$

where lexical successors emit one token and binary successors only rewrite the stack. This is the regex-like part: the parse forest is a DAG expression made of union over alternatives and concatenation through binary backpointers.

## Exact Then Sampled Expansion

Each iteration first counts successors for every active particle. If the total successor count still fits in `MAX_WDFA_FRONTIER`, `wdfa_frontier_write_exact` materializes every successor. This is exact enumeration over the backpointer expression.

Once the successor set would exceed the frontier cap, the decoder switches to particle compression:

1. assign each parent a bounded weight based on WDFA cost and branching count,
2. prefix-sum those weights into a parent CDF,
3. sample `MAX_WDFA_FRONTIER` parents,
4. sample one immediate successor from each selected parent.

Lexical successor mass is

$$
m(t \mid q) = \exp(-\mathrm{cost}(q,t) / \mathrm{scale})
$$

where `cost(q,t)` is the WDFA edge cost, or `missingCost` if the edge is absent. Binary alternatives get unit mass. These are local proposal probabilities for preserving useful frontier diversity, not a globally normalized probability over all completions.

## WDFA Scoring

When a lexical successor emits token $t$, the decoder updates

$$
(q,c) \mapsto (\delta(q,t), c + w(q,t)).
$$

If no WDFA edge exists, it stays in the same state and adds `missingCost`. When the stack becomes empty, the particle is finalized by adding the WDFA final cost:

$$
c_\mathrm{final} = c + F(q).
$$

Infinite final cost marks the packet invalid. Valid completions are ranked by

$$
\mathrm{rank} = c_\mathrm{final} + (\Delta + 1) \cdot 10^7.
$$

This preserves the existing edit-distance-first convention while using the WDFA as the within-distance model.

## Completion Reservoir

Finished particles are not kept alive in the frontier. `wdfa_frontier_emit_done_packets` appends completed packets to a bounded completion reservoir and marks those particles dead. This prevents one cheap completed string from repeatedly occupying the compressed frontier.

After expansion stops, `select_top_k_unique` selects the best packets from the completion reservoir. Its uniqueness hash ignores the epsilon token because the UI strips `ε`; this makes top-k diversity match the displayed completion strings. `gather_top_k` then copies the selected packets back for normal `decodePackets`.

## Summary

The decoder is exact while the active successor set fits the frontier cap. Past that point, it becomes a WDFA-weighted particle approximation over the same backpointer expression. The next-token support always comes from the parse-chart terminal literals and binary backpointers; the probabilities used during compression come from WDFA transition costs plus bounded branch-aware parent weights.
