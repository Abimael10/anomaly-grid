# Use Cases

This directory contains focused use cases that stress finite alphabets (≤20 symbols) with rich temporal structure. Each example trains on normal data, then runs a separate anomaly set and simply reports how many anomalies were caught (no metrics jargon).

## Technical Fit Criteria

**Limitations of examples are kept computational:**
- Scale: Processing large volumes of data
- Performance: Real-time analysis requirements
- Memory: Large vocabularies and pattern libraries
- Distribution: Analysis across multiple systems

## Examples

### 1. Network Protocol State Machine (16 states)
**File**: `network_protocol_analysis.rs`

- **What it does**: Trains on compliant TCP-like sessions; counts how many anomalous sessions (skipped handshakes, resets in data, impossible jumps) are flagged.

```bash
cargo run --example network_protocol_analysis
```

### 2. Protein Folding Sequences (20 amino acids)
**File**: `protein_folding_sequences.rs`

- **What it does**: Trains on biologically plausible folds; counts how many misfolded sequences (hydrophobic exposure, broken disulfides, charge clusters, proline kinks) are flagged.

```bash
cargo run --example protein_folding_sequences
```

### 3. Communication Protocol Timing (12 symbols)
**File**: `communication_protocol_analysis.rs`

- **What it does**: Trains on START/SYNC/DATA/ACK flows; counts how many timing/symbol/flow attacks are flagged.

```bash
cargo run --example communication_protocol_analysis
```

## Why These Scenarios
- **Finite alphabets** (12–20 symbols) that naturally suit Markov contexts.
- **Sequence-first** problems where temporal order carries the signal.
- **Subtle deviations** that benefit from higher-order context rather than aggregate statistics.