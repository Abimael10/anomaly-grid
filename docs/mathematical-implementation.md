# Mathematical Implementation

How the variable-order Markov model computes probabilities and anomaly
scores. The crate-level rustdoc has the same content in API-shaped form;
this page is the textbook companion.

## Variable-order Markov chain

For a context `c` of length `k ≤ max_order` and next symbol `x`, the
conditional probability is the **Witten-Bell interpolation**:

```
P_wb(x | c) = λ(c) · P_ml(x | c) + (1 − λ(c)) · P_wb(x | suffix(c))
λ(c)        = N(c) / (N(c) + T(c))
```

where:

- `N(c)` is the total number of times context `c` was observed during
  training;
- `T(c)` is the number of *distinct* continuations seen after `c`;
- `P_ml(x | c) = (count(c, x) + α) / (N(c) + α · |Σ|)` is the
  Laplace-smoothed maximum-likelihood estimate at this order;
- `suffix(c)` drops the first symbol of `c`, recursing toward order 0.

The order-0 base case is the smoothed unigram:

```
P(x) = (count(x) + α) / (N + α · |Σ|)
```

Probabilities are normalised over the **global** alphabet `Σ` (all
symbols seen at any point during training), not just over the symbols
seen after `c`. This means `Σ_x P(x | c) = 1` exactly for every
observed context.

## Information-theoretic score

For a window `[x₁, …, xₙ]`, the implementation computes:

```
information_score = (1 / (n − 1)) · Σ_{i=1..n−1} −log₂ P_wb(x_{i+1} | x₁..x_i)   [bits]
likelihood        = ∏_{i=1..n−1} P_wb(x_{i+1} | x₁..x_i)                          [chain rule, ∈ [0, 1]]
log_likelihood    = ln(likelihood)                                                 [nats; −∞ on underflow]
anomaly_strength  = tanh((w_l + w_i) · information_score / normalization_factor)   [∈ [0, 1)]
```

The `tanh` envelope keeps the score bounded and monotonic. The two
weights `w_l` and `w_i` (for likelihood and information components)
were separate fields in v0.5 with mismatched units; v0.6 unifies them
in bits and uses their sum as a single scale.

## Numerical stability

Long sequences underflow if probabilities are multiplied directly. The
detection hot path therefore stays in log-space:

```
log_likelihood_bits_per_symbol = (1 / (n − 1)) · Σ log₂ P_wb(...)
```

`MarkovModel::log_likelihood_bits_per_symbol` is the public entry point;
`AnomalyScore::information_score` is the per-window value.

A `min_probability` floor (default `1e-12`) prevents `log(0)` from any
zero-count edge case.

## Memory layout

- **String interning**: every distinct symbol is stored once as
  `Arc<str>` and represented by a `StateId(u32)` everywhere else.
  Hot-path comparisons are integer equality.
- **Arena trie**: contexts share prefixes through a
  `Vec<TrieNode>` indexed by `NodeId(u32)`. Each node carries a
  `SmallVec<[(StateId, NodeId); 4]>` of children — typical alphabets
  keep most nodes inline.
- **Transition counts**: per-context counts live in a
  `TransitionCounts` enum that stays inline in `SmallVec` for ≤ 4
  distinct continuations and spills to `HashMap` only when needed.

## Smoothing rationale

Witten-Bell interpolation was chosen over plain Laplace at higher
orders because it is much better-behaved when `N(c)` is small (rare
contexts get more weight on the lower-order estimate, exactly when ML
estimates would be noisy). The order-0 base case keeps Laplace because
it has a closed-form normalisation over the full alphabet that doesn't
need iteration.

## Verification

Concrete-input regressions live in `tests/math.rs`; randomised
properties (sums to 1, entropy bounds, parallel determinism) live in
`tests/proptest.rs`. Both are run on every CI build under
`#![deny(clippy::pedantic, clippy::nursery, clippy::unwrap_used,
clippy::expect_used)]`.
