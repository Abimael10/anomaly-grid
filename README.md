# anomaly-grid

```
     █████╗ ███╗   ██╗ ██████╗ ███╗   ███╗ █████╗ ██╗  ██╗   ██╗
    ██╔══██╗████╗  ██║██╔═══██╗████╗ ████║██╔══██╗██║  ╚██╗ ██╔╝
    ███████║██╔██╗ ██║██║   ██║██╔████╔██║███████║██║   ╚████╔╝ 
    ██╔══██║██║╚██╗██║██║   ██║██║╚██╔╝██║██╔══██║██║    ╚██╔╝  
    ██║  ██║██║ ╚████║╚██████╔╝██║ ╚═╝ ██║██║  ██║███████╗██║   
    ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝   
    [ANOMALY-GRID v0.1.4] - SEQUENCE ANOMALY DETECTION ENGINE
```

[![Crates.io](https://img.shields.io/crates/v/anomaly-grid)](https://crates.io/crates/anomaly-grid)
[![Documentation](https://docs.rs/anomaly-grid/badge.svg)](https://docs.rs/anomaly-grid/0.1.4)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Sequential pattern analysis through variable-order Markov chains with spectral decomposition and quantum state modeling. Built for detecting deviations in finite-alphabet sequences.**

---


⚠️ Development Status
This library is currently in active development and represents ongoing improve of my knowledge in advanced anomaly detection methodologies. While the core algorithms are mathematically sound and extensively tested, there are areas that require further optimization and refinement.
I acknowledge that complex mathematical implementations can present edge cases and unexpected behaviors. If you encounter any issues, inconsistencies, or have suggestions for improvement, please don't hesitate to reach me out. Your feedback is invaluable for enhancing the library's robustness and reliability.
Known areas for improvement:

Spectral analysis convergence in edge cases
Memory optimization for large state spaces
Performance tuning for specific use cases
Documentation clarity and completeness

Contact: Please file issues on my repository or reach out directly for technical discussions, bug reports, or collaboration opportunities. I am committed to continuous improvement and appreciate your patience as I or we (hopefully) refine this research implementation.


## 🚀 Quick Start

```rust
use anomaly_grid::*;

fn your_function() -> Result<(), Box<dyn std::error::Error>> {
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
- **Variable-Order Markov Models**: Context Tree Weighting with adaptive order selection
- **Spectral Analysis**: Eigenvalue decomposition of transition matrices with robust convergence
- **Information Theory**: Shannon entropy, KL divergence, and surprise quantification
- **Quantum Modeling**: Superposition states with entropy-based phase encoding -- Highly speculative and naive implementations, will be removing in latter versions and experiment somewhere else
- **Topological Features**: Simplified persistent homology and clustering analysis

### Multi-Dimensional Scoring
Each anomaly receives **5 independent scores**:

1. **Likelihood Score**: `prob / sqrt(support)` - Lower = more anomalous
2. **Information Score**: `(surprise + entropy) / length` - Higher = more anomalous  
3. **Spectral Score**: `|observed - stationary|` - Deviation from equilibrium
4. **Quantum Coherence**: `1 - trace/n_states` - Superposition measurement -- Same of what was stated above about these naive implementations.
5. **Topological Signature**: `[components, cycles, clustering]` - Structural complexity

## 🎯 Proven Use Cases

### Network Security
```rust
// Port scan detection
let normal_traffic = vec![
    "TCP_SYN", "TCP_ACK", "HTTP_GET", "HTTP_200", "TCP_FIN"
];
let attack_pattern = vec![
    "TCP_SYN", "TCP_RST", "TCP_SYN", "TCP_RST", "TCP_SYN", "TCP_RST"
];
```

### User Behavior Analysis
```rust
// Privilege escalation detection
let normal_session = vec![
    "LOGIN", "DASHBOARD", "PROFILE", "SETTINGS", "LOGOUT"
];
let suspicious_session = vec![
    "LOGIN", "ADMIN_PANEL", "USER_LIST", "DELETE_USER", "DELETE_USER"
];
```

### Financial Fraud
```rust
// Velocity attack detection
let normal_transactions = vec![
    "AUTH", "PURCHASE", "CONFIRM", "SETTLEMENT"
];
let fraud_pattern = vec![
    "VELOCITY_ALERT", "AUTH", "AUTH", "AUTH", "AUTH"
];
```

### System Monitoring
```rust
// Service crash detection
let normal_logs = vec![
    "BOOT", "SERVICE_START", "AUTH_SUCCESS", "FILE_ACCESS"
];
let anomalous_logs = vec![
    "SERVICE_CRASH", "SERVICE_CRASH", "SERVICE_CRASH", "ROOTKIT_DETECTED"
];
```

### Bioinformatics
```rust
// DNA mutation detection
let normal_gene = vec![
    "ATG", "CGA", "TTC", "AAG", "GCT", "TAA"  // Start -> Stop codon
];
let mutation = vec![
    "XTG", "CGA", "TTC", "AAG", "GCT"  // Invalid nucleotide + missing stop
];
```

## ⚡ Performance Characteristics

### Computational Complexity
```
Training:   O(n × k × order)     where n=sequence_length, k=alphabet_size
Detection:  O(m × k × log(k))    where m=test_length
Memory:     O(k^order)           exponential in context depth
```

### Parallel Processing
```rust
// Batch analysis across multiple sequences
let sequences = vec![
    vec!["GET", "200", "POST", "201"],
    vec!["SELECT", "INSERT", "COMMIT"],
    vec!["SYN", "ACK", "DATA", "FIN"]
];

let results = batch_process_sequences(&sequences, 3, 0.05);
// Processes all sequences in parallel using Rayon
```

## 🛠️ Installation & Dependencies

```toml
[dependencies]
anomaly-grid = "0.1.4"

# Or add manually:
nalgebra = "0.33.2"  # Linear algebra operations
ndarray = "0.16.1"   # N-dimensional arrays
rayon = "1.10.0"     # Parallel processing
```

## 📊 Advanced Usage

### Model Configuration
```rust
// Recommended parameters for different scenarios
let network_detector = AdvancedTransitionModel::new(4);  // Network protocols
let user_detector = AdvancedTransitionModel::new(3);     // User sessions  
let financial_detector = AdvancedTransitionModel::new(4); // Transactions
let bio_detector = AdvancedTransitionModel::new(6);      // DNA sequences
```

### Training Requirements
```rust
// Minimum data requirements for stable analysis
let min_sequence_length = 20 * max_order;  // Statistical significance
let min_examples_per_symbol = 5;           // Reliable probability estimates
let recommended_alphabet_size = 10..=50;   // Memory vs. expressiveness trade-off
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
cargo test test_network_traffic_anomalies     # Network security
cargo test test_user_behavior_patterns        # Behavioral analysis
cargo test test_financial_transaction_patterns # Fraud detection
cargo test test_dna_sequence_analysis         # Bioinformatics
cargo test test_performance_benchmarks        # Scaling analysis
```

### Mathematical Validation
The library automatically validates:
- **Probability Conservation**: All context probabilities sum to 1.0
- **Entropy Bounds**: 0 ≤ entropy ≤ log₂(alphabet_size)
- **Spectral Stability**: Eigenvalue convergence within tolerance
- **Numerical Precision**: No NaN/infinity propagation



## 🚨 Known Limitations

### Memory Scaling
```rust
// Memory usage grows exponentially with context order
let contexts_10_3 = 10_usize.pow(3);      // 1,000 contexts
let contexts_50_3 = 50_usize.pow(3);      // 125,000 contexts  
let contexts_10_5 = 10_usize.pow(5);      // 100,000 contexts

// Recommended limits:
assert!(alphabet_size <= 50);
assert!(max_order <= 5);
assert!(sequence_length >= 20 * max_order);
```

### Spectral Analysis Constraints
- **Matrix Conditioning**: Large/sparse matrices may have unstable eigenvalues
- **Convergence Issues**: Disconnected graphs may not reach stationary distribution
- **Computational Cost**: O(n³) eigenvalue decomposition for n states

### Quantum Features Disclaimer - Speculative implementations
- **Simplified Implementation**: Not full quantum computation, this is highly speculative for many reason in that area, don't crucify me xD
- **Phase Encoding**: Based on classical entropy values only
- **Coherence Measure**: Approximation of true quantum coherence

In later versions this will be avoided completely since I will continue experimenting with it on a sepatate, we can enjoy some naive implementations :)

## 🔧 Configuration Tuning

### Sensitivity vs. False Positives
```rust
let threshold = match use_case {
    "critical_security" => 0.001,    // High sensitivity
    "fraud_detection"   => 0.01,     // Balanced
    "general_monitoring" => 0.1,     // Low false positives
};
```
Example:

```rust
use anomaly_grid::{AdvancedTransitionModel, AnomalyScore};

fn your_fn() {
    println!("Starting anomaly detection example...");

    //Prepare a sequence of states (your data)
    let sequence: Vec<String> = vec![
        "normal_event_A",
        "normal_event_B",
        "normal_event_C",
        "normal_event_A",
        "normal_event_B",
        "normal_event_C",
        "unexpected_event_X",//Anomaly
        "unusual_event_Y",//Anomaly
        "normal_event_A",
        "normal_event_B",
        "normal_event_C",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    //Create a new AdvancedTransitionModel instance
    //max_order determines the context length for the Markov model
    let max_order = 2;
    let mut model = AdvancedTransitionModel::new(max_order);

    //Build the context tree (train the model on your data)
    // this is less than ideal though, you would train on a large set of 'normal' data first.
    // f or this example, I build the model on the sequence itself to demonstrate.
    match model.build_context_tree(&sequence) {
        Ok(_) => println!("Context tree built successfully."),
        Err(e) => {
            eprintln!("Error building context tree: {}", e);
            return;
        }
    }

    //Define a threshold for anomaly detection
    // A lower threshold generally means stricter anomaly detection (fewer false positives, but potentially less sensitivity).
    let anomaly_threshold = 0.05;    //Example threshold

    println!(
        "\nDetecting anomalies with threshold: {:.4}",
        anomaly_threshold
    );

    // Detect advanced anomalies in the sequence
    let anomalies: Vec<AnomalyScore> =
        model.detect_advanced_anomalies(&sequence, anomaly_threshold);

    //   Process and display the detected anomalies
    if anomalies.is_empty() {
        println!("No anomalies detected based on the calculated scores.");
    } else {
        println!("\n--- Detected Anomalies (scores below threshold) ---");
        for anomaly in &anomalies {
            //In your detect_advanced_anomalies function, the threshold isnt used
            //for filtering. Here, we manually filter based on 'likelihood' for demonstration
            if anomaly.likelihood < anomaly_threshold {
                println!("  Anomaly Detected:");
                println!("    Sequence: {:?}", anomaly.state_sequence);
                println!("    Likelihood: {:.6}", anomaly.likelihood);
                println!(
                    "    Information Score: {:.6}",
                    anomaly.information_theoretic_score
                );
                println!("    Spectral Score: {:.6}", anomaly.spectral_anomaly_score);
                println!(
                    "    Quantum Coherence: {:.6}",
                    anomaly.quantum_coherence_measure
                );
                println!(
                    "    Topological Signature: {:?}",
                    anomaly.topological_signature
                );
                println!(
                    "    Confidence Interval: ({:.6}, {:.6})",
                    anomaly.confidence_interval.0, anomaly.confidence_interval.1
                );
                println!();
            } else {
                //You might choose to print events that are "less" anomalous but still scored
                // println!("  Normal Event (score above threshold): {:?}", anomaly.state_sequence);
                // println!("    Likelihood: {:.6}", anomaly.likelihood);
            }
        }
    }
    println!("Anomaly detection example finished.");
}
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

Example:

```rust
//I set up the helper function like this
fn reduce_alphabet(sequence: &[String]) -> Vec<String> {
    sequence
        .iter()
        .map(|s| match s.as_str() {
            "HTTP_GET" | "HTTP_POST" | "HTTP_PUT" => "HTTP_REQUEST".to_string(),
            "TCP_SYN" | "TCP_ACK" | "TCP_FIN" => "TCP_CONTROL".to_string(),
            _ => s.clone(),
        })
        .collect()
}

fn main_fn() {
    // Example of a raw sequence with a potentially large alphabet
    let raw_sequence_data: Vec<String> = vec![
        "HTTP_GET".to_string(),
        "TCP_SYN".to_string(),
        "HTTP_POST".to_string(),
        "FILE_ACCESS".to_string(),
        "TCP_ACK".to_string(),
        "HTTP_GET".to_string(),
        "FTP_LOGIN".to_string(),
        "TCP_FIN".to_string(),
        "SYSTEM_ERROR".to_string(),
    ];

    println!(
        "Original Sequence (size: {}): {:?}",
        raw_sequence_data.len(),
        raw_sequence_data
    );

    // Apply the alphabet reduction for memory optimization
    let processed_sequence_data = reduce_alphabet(&raw_sequence_data);

    println!(
        "Processed Sequence (size: {}): {:?}",
        processed_sequence_data.len(),
        processed_sequence_data
    );
}
```

### Skip if you understood based on the code, I am the worst programmer in the world so I would not be surprised, so for normies like me, here is the memory opt explanation:

The key benefit of `reduce_alphabet` for memory opt
comes when this `processed_sequence_data` is then used to build
data structures that depend on the uniqueness of the elements,
such as a HashMap for contexts in a Markov model.

So if you were to build a `HashMap<Vec<String>, usize>`
to count occurrences of different patterns:
Without `reduce_alphabet`, "HTTP_GET", "HTTP_POST", and "HTTP_PUT"
would be distinct keys. With `reduce_alphabet`, they all become
"HTTP_REQUEST", reducing the number of unique keys and with that
the memory consumed by the HashMap and its associated data.

Example (conceptual, assumes you use our AdvancedTransitionModel or similar):
let mut model = AdvancedTransitionModel::new(3);
model.build_context_tree(&processed_sequence_data).unwrap();
(This step would use less memory than if raw_sequence_data was used)

The 'AdvancedTransitionModel' (from this lib) internally builds
a context tree using a HashMap to store `ContextNode`s. Each `ContextNode`
also contains HashMaps for `counts` and `probabilities`.

By reducing the alphabet, you directly decrease the number of unique
'states' that appear in these HashMaps, leading to:
    1. Fewer entries in the top-level 'contexts' HashMap.
    2. Fewer entries in the 'counts' and 'probabilities' HashMaps within each       'ContextNode'.
This reduces the overall memory footprint of the model, especially for
high-order Markov models and long sequences with many distinct original states.

### Performance Optimization
```rust
// Use batch processing for multiple sequences
let results = sequences
    .par_iter()  // Parallel processing
    .map(|seq| {
        let mut model = AdvancedTransitionModel::new(3);
        model.build_context_tree(seq).unwrap();
        model.detect_advanced_anomalies(seq, threshold)
    })
    .collect();
```

Example:

```rust
use anomaly_grid::{AnomalyScore, batch_process_sequences};

fn your_function() {
    // Define a threshold for anomaly detection
    let anomaly_threshold = 0.05;

    // Define multiple sequences as a vector of vectors of strings
    let sequences_to_analyze: Vec<Vec<String>> = vec![
        vec!["A", "B", "C", "A", "B", "C", "X", "Y", "Z"]
            .into_iter()
            .map(String::from)
            .collect(),
        vec!["P", "Q", "R", "P", "S", "T", "U", "V"]
            .into_iter()
            .map(String::from)
            .collect(),
        vec!["X", "Y", "Z", "X", "Y", "Z", "X", "A", "B", "C"]
            .into_iter()
            .map(String::from)
            .collect(),
    ];

    // Define the maximum order for the Markov model
    let max_markov_order = 3;

    // HERE :) -- Use batch_process_sequences to process all sequences in parallel
    let all_anomaly_results: Vec<Vec<AnomalyScore>> =
        batch_process_sequences(&sequences_to_analyze, max_markov_order, anomaly_threshold);

    // Iterate through the results for each sequence
    for (i, sequence_anomalies) in all_anomaly_results.iter().enumerate() {
        println!("Anomalies for Sequence {}:", i + 1);
        if sequence_anomalies.is_empty() {
            println!("  No anomalies detected.");
        } else {
            for anomaly in sequence_anomalies {
                println!("  Anomaly Detected:");
                println!("    Sequence: {:?}", anomaly.state_sequence);
                println!("    Likelihood: {:.6}", anomaly.likelihood);
                println!(
                    "    Information Score: {:.6}",
                    anomaly.information_theoretic_score
                );
                println!("    Spectral Score: {:.6}", anomaly.spectral_anomaly_score);
                println!(
                    "    Quantum Coherence: {:.6}",
                    anomaly.quantum_coherence_measure
                );
                println!("    Confidence Interval: {:?}", anomaly.confidence_interval);
                println!(
                    "    Topological Signature: {:?}",
                    anomaly.topological_signature
                );
                println!();
            }
        }
        println!("------------------------------------");
    }
}
```

## 📚 Documentation

- **[User Manual](USER_MANUAL.md)**: Comprehensive developer guide with examples
- **[API Documentation](https://docs.rs/anomaly-grid)**: Generated from source code

## 📈 Roadmap

### Version 0.2.0 (Planned)
- [ ] Streaming anomaly detection for real-time systems
- [ ] Advanced topological analysis with true persistent homology
- [ ] GPU acceleration for large-scale datasets
- [ ] Integration with popular ML frameworks (PyTorch, TensorFlow)

### Version 0.3.0 (Future)
- [ ] Distributed processing across multiple machines
- [ ] Advanced quantum algorithms for state analysis
- [ ] Automated hyperparameter optimization
- [ ] Web-based visualization dashboard

## 🤝 Contributing

```bash
# Development setup
git clone https://github.com/username/anomaly-grid.git
cd anomaly-grid
cargo build --release
cargo test

# Run comprehensive benchmarks
cargo test run_all_comprehensive_tests -- --nocapture --ignored
```

## 📄 License

Licensed under the MIT License. See LICENCE for details.

---
