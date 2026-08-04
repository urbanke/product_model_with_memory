# Tokenizer for text8 / enwik8 / enwik9 — specification v3 (locked)

Settled after three rounds of review.  Where a choice is settled by
measurement it says so; where it is settled by simplicity or
generality because the measured difference is immaterial, it says
that instead.  No rule contains a data-derived constant.  The
defaults below are justified for Wikipedia-like corpora — the files
this paper reports on — and are not claimed to be universal.

## 1. Segmentation

Scan left to right into maximal segments:

* **L-segment** — a maximal run of ASCII letters `A–Z a–z`
* **N-segment** — a maximal run of ASCII digits `0–9`
* **B-segment** — any single other byte

Concatenation reproduces the file exactly; no separators are needed.

## 2. The two switches

Both alternatives are implemented; **all four combinations are
measured and reported**, not only the winner.

**Settled, 31 July 2026 (enwik8, memoryless, bits/character).**
intern + conditioned **3.1282** · intern + folded 3.1413 ·
compositional + conditioned 3.1902 · compositional + folded 3.2032.
Interning wins by 0.062, conditioning by 0.013 — and the second
margin reproduces in both number settings, which is why it counts.
The defaults are therefore `numbers=intern`, `case=conditioned`.  The
alternatives stay implemented and tested: they are the control.

**`numbers` ∈ {intern, compositional}.**
*intern*: an N-segment is one token; a digit string seen for the
first time is coded as `ESC-NUM` and spelled into S2b, then joins the
vocabulary.  *compositional*: N-segments are not formed at all —
digits are ordinary B-segments, one token per digit.

Why this is a switch rather than a decision: digits are 2.18% of
enwik8 and 3.45% of enwik9 by volume, so the coding-cost difference
is small either way.  The live argument is *alignment*: our memory
families index distance in tokens, and compositional coding makes the
stream 4.6% longer on enwik9 while pushing 13.4M contexts apart.
That argument is real but model-dependent, and both segmentations
preserve the same information (byte offsets are recoverable from
either), so the question is empirical.

**`case` ∈ {conditioned, folded}.**
*conditioned*: token identity is the lowercased form; the case class
travels in stream S3, modelled with the current token as its state —
which is our existing per-state machinery with a four-symbol
alphabet, not new logic.  *folded*: case is part of the token
identity, S3 does not exist.

Independent (unconditioned) case coding was to be **not offered**, on
the argument that it is dominated.  Measurement corrected this on both
counts and it is now always evaluated alongside the conditioned model,
with the cheaper of the two selected at a cost of log2(3) bits.

On enwik8 the independent model costs 0.8711 bits per letter run
(0.1261 bpc) and the conditioned model 0.4640 (0.0672 bpc), so
conditioning saves 0.0589 bpc — a real gain, but the premise that case
is "nearly determined by the word" overstates it: half a bit per run
survives.  On text8 the sign reverses.  Every letter run there is
lowercase, so the stream is constant; the pooled model codes all
17,005,207 symbols in **28.7 bits**, while conditioning on the token
costs **547,004** bits, because each of 135,336 states pays its own
start-up.  Conditioning is not free, and a corpus that does not need
it must not be made to pay.

The same argument one level down applies to the masks, and the same
caution: conditioning them on the token saves 0.0009 bpc on enwik8,
real but an order of magnitude smaller than the case gain.

## 3. Streams

Concatenated, in order.  Physical layout is an implementation detail:
with per-stream models, concatenation and interleaving give identical
codelengths, since neither reorders the symbols within a stream.  The
real axis is independent streams versus cross-stream conditioning,
and `case=conditioned` is the first instance of the latter.

* **S1 tokens** — one symbol per segment over
  `{byte 0…255} ∪ {ESC-WORD} ∪ [ESC-NUM] ∪ {vocabulary} ∪ {EOF}`.
  Vocabulary indices are assigned in order of first occurrence; the
  decoder applies the identical rule, so nothing is transmitted.
* **S2a word spellings** — per `ESC-WORD`, the lowercased letters over
  `{a…z, END}`.
* **S2b number spellings** (`numbers=intern` only) — per `ESC-NUM`,
  the digits over `{0…9, END}`.
* **S3 case** (`case=conditioned` only) — one symbol per L-segment
  from `{lower, Cap, UPPER, mixed}`, partitioned as: `lower` = all
  lowercase; `Cap` = first uppercase and the rest lowercase (includes
  a single uppercase letter); `UPPER` = all uppercase with length ≥ 2;
  `mixed` = otherwise.
* **S4 masks** — per `mixed` L-segment, one binary symbol per letter
  (1 = uppercase), coded, not raw.

The spec fixes *what* S4 contains, not how it is modelled, and the
argument that settled S3 applies once more one level down: given the
token, the capitalisation pattern is nearly determined — `iPhone` is
`0100000` every time — so a repeat should not pay for its pattern
again.  The token also fixes the word length, so state *s* contributes
one binary stream per letter position; the pooled state −1 (first
occurrences, no fixed length) keeps a single binary model.  This is
decodable: when a mask is read the decoder already has the token, the
class `mixed`, and for a first occurrence the spelling, hence the
length.  Both models are measured and reported (`token_baseline.py`
prints three complete schemes, and every one of them is a valid code),
so nothing here is assumed.

## 4. Framing

No header.  S1 ends with `EOF`; the counts then follow: number of
`ESC-WORD` gives S2a, `ESC-NUM` gives S2b, L-segments gives S3,
`mixed` symbols in S3 gives S4 (each mask's length is its word's
length, known by then).  Coder flush is at most 16 bits per stream
and is reported, not ignored.

## 5. Acceptance tests — before any number is reported

1. **Round trip**: text8, enwik8, enwik9 reassembled byte for byte,
   for every switch combination.
2. **Framing self-sufficiency**: decode from the streams alone and
   check all recovered counts against the encoder's.
3. **Accounting closure**: stream costs plus overhead equal the
   reported total, to the bit.
4. **text8 comparison**: report the new total against the existing
   word+spelling accounting.  These are *not* the same code — the new
   scheme spends a token on each space, which the old one folded into
   the word — so the memoryless total is expected to be higher, with
   the gap shrinking under memory.  A gap that does not shrink
   indicates a modelling error.

## 6. Reported with every result

Stream costs separately (bits and bits/character); vocabulary growth
and S2's share of the total (the price of adaptivity); L/N/B segment
shares and the case histogram; and the **computational cost** of each
variant — time and peak memory — so that a small coding gain bought
with a large resource cost is visible as such.

## 7. Excluded

No XML- or Wikipedia-specific transforms; no shipped dictionary; no
vocabulary pruning; no subword/BPE vocabulary yet (the natural later
variant: it bounds the alphabet and removes S2).

**The subword variant is no longer optional — 31 July 2026.**  Run
against `cl100k_base` on the same estimator, this tokenizer loses by
0.173 bpc on enwik8 *even after the BPE vocabulary is charged* for its
zipped size (776,019 bytes, 0.0621 bpc at 10^8).  Only a third of that
gap is dictionary accounting (ours 0.1238 bpc against their 0.0621);
the rest is segmentation.  cl100k packs 3.877 bytes into a token where
we average 2.405, because **63% of our segments are single "other"
bytes** — the brackets, slashes, equals signs and quotes of XML, one
symbol each, which BPE merges.  Deleting our spelling streams outright
would still leave us at 3.004 bpc, above the uncharged BPE figure.

So the next representation-level gain is an adaptive subword
segmentation — merges learned from the past, so still nothing shipped
— and the target is the B-segment run, not the vocabulary cost.
