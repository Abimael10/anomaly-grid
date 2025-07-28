# anomaly-grid

```
     █████╗ ███╗   ██╗ ██████╗ ███╗   ███╗ █████╗ ██╗  ██╗   ██╗
    ██╔══██╗████╗  ██║██╔═══██╗████╗ ████║██╔══██╗██║  ╚██╗ ██╔╝
    ███████║██╔██╗ ██║██║   ██║██╔████╔██║███████║██║   ╚████╔╝ 
    ██╔══██║██║╚██╗██║██║   ██║██║╚██╔╝██║██╔══██║██║    ╚██╔╝  
    ██║  ██║██║ ╚████║╚██████╔╝██║ ╚═╝ ██║██║  ██║███████╗██║   
    ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝   
    [ANOMALY-GRID v0.1.6] - SEQUENCE ANOMALY DETECTION ENGINE
```

[![Crates.io](https://img.shields.io/crates/v/anomaly-grid)](https://crates.io/crates/anomaly-grid)
[![Documentation](https://docs.rs/anomaly-grid/badge.svg)](https://docs.rs/anomaly-grid/0.1.6)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Sequential pattern analysis through variable-order Markov chains with simplified spectral decomposition and quantum state modeling. Built for detecting deviations in finite-alphabet sequences.**

---

## ⚠️ Development Status

This library is currently in active development and represents ongoing exploration of advanced anomaly detection methodologies. While the core algorithms are mathematically sound and extensively tested, there are areas that require further optimization and refinement.

**Known areas for improvement:**
- Spectral analysis stability (currently simplified to prevent infinite loops)
- Memory optimization for large state spaces (max 500 contexts, sequences limited to 200 elements)
- Performance tuning for specific use cases
- Documentation clarity and completeness

**Current Safety Limits:**
- Maximum sequence length: 200 elements
- Maximum contexts: 500 per model
- Maximum order: Capped at 3 regardless of input (This for simple tests at the beginning since it can scale quickly without it and I do not count with enough computing to process that right now! This will change quickly and can be edited fairly easy too).
- Spectral analysis: Simplified fallback implementation

Contact: Please file issues on the repository or reach out directly for technical discussions, bug reports, or collaboration opportunities.

## 🚀 Quick Start

```rust
use anomaly_grid::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize detection engine
    let mut detector = AdvancedTransitionModel::new(3);

    // Train on normal patterns
    let baseline: Vec<String> = vec!["connect", "auth", "query", "disconnect"]
        .into_iter()
        .map(String::from)
        .collect();
    detector.build_context_tree(&baseline)?;

    // Detect anomalies in suspicious activity
    let suspect: Vec<String> = vec!["connect", "auth", "admin_escalate", "dump_db"]
        .into_iter()
        .map(String::from)
        .collect();
    let threats = detector.detect_advanced_anomalies(&suspect, 0.01);

    // Analyze results
    for threat in threats {
        if threat.likelihood < 1e-6 {
            println!("🚨 HIGH THREAT: {:?}", threat.state_sequence);
            println!("   Risk Score: {:.2e}", 1.0 - threat.likelihood);
            println!(
                "   Confidence: [{:.2e}, {:.2e}]",
                threat.confidence_interval.0, threat.confidence_interval.1
            );
        }
    }

    Ok(())
}
```

## 🔬 Core Technology Stack

### Mathematical Foundation
- **Variable-Order Markov Models**: Context Tree Weighting with adaptive order selection (capped at order 3)
- **Simplified Spectral Analysis**: Basic eigenvalue representation with fallback uniform distributions
- **Information Theory**: Shannon entropy, KL divergence, and surprise quantification
- **Quantum Modeling**: Equal superposition states with entropy-based phase encoding (experimental, will be removed in future versions)
- **Topological Features**: Basic cycle detection and clustering analysis

### Multi-Dimensional Scoring
Each anomaly receives **5 independent scores**:

1. **Likelihood Score**: Variable-order context probability - Lower = more anomalous
2. **Information Score**: `(surprise + entropy) / length` - Higher = more anomalous  
3. **Spectral Score**: `|observed - stationary|` - Deviation from uniform distribution (simplified)
4. **Quantum Coherence**: `1 - trace/n_states` - Superposition measurement (experimental)
5. **Topological Signature**: `[components, cycles, clustering]` - Structural complexity

## 🎯 Proven Use Cases

### Network Security
```rust
// Port scan detection
let normal_traffic = vec![
    "TCP_SYN", "TCP_ACK", "HTTP_GET", "HTTP_200", "TCP_FIN"
].into_iter().map(String::from).collect();

let attack_pattern = vec![
    "TCP_SYN", "TCP_RST", "TCP_SYN", "TCP_RST", "TCP_SYN", "TCP_RST"
].into_iter().map(String::from).collect();

let mut detector = AdvancedTransitionModel::new(2);
detector.build_context_tree(&normal_traffic)?;
let anomalies = detector.detect_advanced_anomalies(&attack_pattern, 0.01);
```

### User Behavior Analysis
```rust
// Privilege escalation detection
let normal_session = vec![
    "LOGIN", "DASHBOARD", "PROFILE", "SETTINGS", "LOGOUT"
].into_iter().map(String::from).collect();

let suspicious_session = vec![
    "LOGIN", "ADMIN_PANEL", "USER_LIST", "DELETE_USER", "DELETE_USER"
].into_iter().map(String::from).collect();
```

### Financial Fraud
```rust
// Velocity attack detection
let normal_transactions = vec![
    "AUTH", "PURCHASE", "CONFIRM", "SETTLEMENT"
].into_iter().map(String::from).collect();

let fraud_pattern = vec![
    "VELOCITY_ALERT", "AUTH", "AUTH", "AUTH", "AUTH"
].into_iter().map(String::from).collect();
```

## ⚡ Performance Characteristics

### Computational Complexity
```
Training:   O(n × k × order)     where n=sequence_length (max 200), k=alphabet_size
Detection:  O(m × k × log(k))    where m=test_length
Memory:     O(contexts)          limited to 500 contexts maximum
```

### Current Safety Limits
```rust
// Built-in safety limits in current implementation
let max_sequence_length = 200;  // Hard limit in build_context_tree
let max_contexts = 500;         // Memory protection
let effective_max_order = 3;    // Capped regardless of input
```

### Parallel Processing
```rust
// Batch analysis across multiple sequences
let sequences = vec![
    vec!["GET", "200", "POST", "201"].into_iter().map(String::from).collect(),
    vec!["SELECT", "INSERT", "COMMIT"].into_iter().map(String::from).collect(),
    vec!["SYN", "ACK", "DATA", "FIN"].into_iter().map(String::from).collect(),
];

let results = batch_process_sequences(&sequences, 2, 0.05);
// Processes all sequences in parallel using Rayon
```

## 🛠️ Installation & Dependencies

```toml
[dependencies]
anomaly-grid = "0.1.6"

# Core dependencies included:
nalgebra = "0.33.2"  # Linear algebra operations
ndarray = "0.16.1"   # N-dimensional arrays
rayon = "1.10.0"     # Parallel processing
```

## 📊 Advanced Usage

### Model Configuration
```rust
// All models effectively use max_order=3 due to current safety limits
let detector = AdvancedTransitionModel::new(5);  // Will be capped at 3 internally
```

### Training Requirements
```rust
// Current implementation requirements
let min_sequence_length = 2;           // Minimum for processing
let max_sequence_length = 200;         // Hard limit
let max_contexts = 500;                // Memory protection
let recommended_order = 2;             // For best performance
```

### Result Interpretation
```rust
for anomaly in anomalies {
    let risk_score = 1.0 - anomaly.likelihood;
    
    match risk_score {
        r if r > 0.999 => println!("🔴 CRITICAL: {:.2e}", r),
        r if r > 0.99  => println!("🟡 HIGH: {:.2e}", r),
        r if r > 0.9   => println!("🟢 MEDIUM: {:.2e}", r),
        _              => println!("ℹ️  LOW: {:.2e}", risk_score),
    }
    
    // Multi-dimensional analysis
    println!("Information entropy: {:.4}", anomaly.information_theoretic_score);
    println!("Spectral deviation: {:.4}", anomaly.spectral_anomaly_score);
    println!("Quantum coherence: {:.4}", anomaly.quantum_coherence_measure);
    println!("Topological complexity: {:?}", anomaly.topological_signature);
}
```

## 🧪 Testing & Validation

### Comprehensive Test Suite
```bash
# Run all tests with detailed output
cargo test -- --nocapture

# Individual test categories
cargo test func_tests                  # Core mathematical functions
cargo test tests                       # Basic functionality tests
```

### Mathematical Validation
The library automatically validates:
- **Probability Conservation**: All context probabilities sum to 1.0
- **Entropy Bounds**: 0 ≤ entropy ≤ log₂(alphabet_size)
- **Memory Limits**: Prevents excessive context creation
- **Numerical Precision**: No NaN/infinity propagation

## 🚨 Known Limitations

### Current Implementation Constraints
```rust
// Hard-coded limits in current version
let max_sequence_processing = 200;     // Elements
let max_contexts_per_model = 500;      // Memory protection
let effective_max_order = 3;           // Performance protection
let simplified_spectral = true;        // No complex eigenvalue computation
```

### Spectral Analysis Status
- **Simplified Implementation**: Uses uniform distributions as fallbacks
- **No Power Iteration**: Removed to prevent infinite loops
- **Fixed Values**: Mixing time and spectral gap are constants
- **Memory Safe**: No large matrix operations

### Quantum Features Disclaimer
- **Experimental**: Highly speculative implementations
- **Simplified**: Equal superposition with basic phase encoding
- **Removal Planned**: Will be moved to separate experimental library

## 🔧 Configuration Tuning

### Sensitivity vs. False Positives
```rust
let threshold = match use_case {
    "critical_security" => 0.001,    // High sensitivity
    "fraud_detection"   => 0.01,     // Balanced
    "general_monitoring" => 0.1,     // Low false positives
};
```

### Memory Optimization
```rust
// For large alphabets, consider preprocessing:
fn reduce_alphabet(sequence: &[String]) -> Vec<String> {
    sequence.iter()
        .map(|s| match s.as_str() {
            "HTTP_GET" | "HTTP_POST" | "HTTP_PUT" => "HTTP_REQUEST".to_string(),
            "TCP_SYN" | "TCP_ACK" | "TCP_FIN" => "TCP_CONTROL".to_string(),
            _ => s.clone()
        })
        .collect()
}
```

### Performance Optimization
```rust
// Use batch processing for multiple sequences
let results = sequences
    .par_iter()  // Parallel processing
    .map(|seq| {
        let mut model = AdvancedTransitionModel::new(2);  // Use lower order
        model.build_context_tree(seq).unwrap();
        model.detect_advanced_anomalies(seq, threshold)
    })
    .collect();
```

## 📚 Documentation

- **[User Manual](USER_MANUAL.md)**: Comprehensive developer guide with examples
- **[API Documentation](https://docs.rs/anomaly-grid)**: Generated from source code


## 🤝 Contributing

```bash
# Development setup
git clone https://github.com/username/anomaly-grid.git
cd anomaly-grid
cargo build --release
cargo test

# Run comprehensive tests (currently limited to prevent infinite loops)
cargo test func_tests -- --nocapture
```

## 📄 License

Licensed under the MIT License. See LICENSE for details.

---

**Note**: This library is currently in active development with built-in safety limits to ensure stability. The spectral analysis has been simplified to prevent infinite loops, and quantum features are experimental placeholders that will be removed in future versions.