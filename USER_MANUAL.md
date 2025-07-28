# Advanced Markov Anomaly Detection Library
## Developer Manual v2.0

```
╔════════════════════════════════════════════════════╗
║                   anomaly-grid                     ║
║           Sequential Anomaly Detection             ║
║                   Library v0.1.4                   ║
╚════════════════════════════════════════════════════╝
```

## Quick Start

These are included

```toml
[dependencies]
nalgebra = "0.33.2"
ndarray = "0.16.1"
rayon = "1.10.0"
```

```rust
use anomaly_grid::*;

// Basic usage
let mut model = AdvancedTransitionModel::new(3);
let sequence = vec!["A", "B", "C", "A", "B", "D"].iter().map(|s| s.to_string()).collect();
model.build_context_tree(&sequence)?;
let anomalies = model.detect_advanced_anomalies(&sequence, 0.1);
```

## Core Components

### 1. AdvancedTransitionModel

The main analysis engine that builds variable-order Markov models with spectral and quantum extensions (this last one is highly speculative, kind of naive and experimental, i will remove in later versions).

```rust
pub struct AdvancedTransitionModel {
    pub contexts: HashMap<Vec<String>, ContextNode>,     // Learned contexts
    pub max_order: usize,                                // Maximum context length
    pub quantum_representation: Option<QuantumState>,    // Quantum state vector
    pub spectral_decomposition: Option<SpectralAnalysis>,// Eigenvalue analysis
}
```

**Constructor:**
```rust
let model = AdvancedTransitionModel::new(max_order: usize);
```
- `max_order`: Maximum context length (memory scales as `alphabet_size^max_order`)
- Recommended values: 2-5 for most applications
- Higher orders require exponentially more training data

### 2. Context Nodes

Each context stores transition statistics and information-theoretic measures.

```rust
pub struct ContextNode {
    pub counts: HashMap<String, usize>,        // Raw transition counts
    pub probabilities: HashMap<String, f64>,   // Normalized probabilities
    pub information_content: f64,              // -log2(entropy)
    pub entropy: f64,                          // Shannon entropy
    pub kl_divergence: f64,                    // KL divergence from uniform
}
```

### 3. Anomaly Scores

Multi-dimensional anomaly assessment for each sequence window.

```rust
pub struct AnomalyScore {
    pub state_sequence: Vec<String>,           // The analyzed window
    pub likelihood: f64,                       // Probability under model
    pub information_theoretic_score: f64,      // Entropy-based surprise
    pub spectral_anomaly_score: f64,          // Stationary distribution deviation
    pub quantum_coherence_measure: f64,        // Quantum coherence measure
    pub topological_signature: Vec<f64>,       // [components, cycles, clustering]
    pub confidence_interval: (f64, f64),      // Bayesian confidence bounds
}
```

## Training Process: build_context_tree()

The training phase builds the internal model from a sequence of normal patterns.

```rust
pub fn build_context_tree(&mut self, sequence: &[String]) -> Result<(), String>
```

### What Actually Happens:

1. **Context Extraction**: For each window size from 1 to `max_order`:
   ```rust
   // Example: max_order=3, sequence=["A","B","C","A","B"]
   // Creates contexts:
   // ["A"] -> "B" (count: 2)
   // ["B"] -> "C" (count: 1), ["B"] -> end (count: 1)  
   // ["A","B"] -> "C" (count: 1)
   // ["B","C"] -> "A" (count: 1)
   // ["A","B","C"] -> "A" (count: 1)
   ```

2. **Probability Calculation**: For each context-transition pair:
   ```rust
   probability = count / total_transitions_from_context
   ```

3. **Information Measures**:
   ```rust
   entropy = -Σ(p * log2(p))                    // Shannon entropy
   information_content = -log2(entropy)          // Surprise measure
   kl_divergence = Σ(p * log2(p / uniform_p))   // Distance from uniform
   ```

4. **Spectral Analysis**: 
   - Builds transition matrix from first-order contexts only
   - Computes eigenvalues using nalgebra
   - Finds stationary distribution via power iteration (with convergence limits)
   - Calculates spectral gap and mixing time

5. **Quantum Representation**:
   - Creates equal superposition state
   - Applies entropy-based phase encoding
   - Stores as complex amplitude vector

### Training Data Requirements:

- **Minimum sequence length**: `20 * max_order` for stable analysis
- **Alphabet coverage**: Each symbol should appear at least 5-10 times
- **Context diversity**: Need sufficient examples of each context-transition pair

## Detection Process: detect_advanced_anomalies()

Analyzes test sequences using sliding window approach.

```rust
pub fn detect_advanced_anomalies(&self, sequence: &[String], threshold: f64) -> Vec<AnomalyScore>
```

### Window Analysis:

The function slides a window of size `max_order + 1` across the sequence:

```rust
// Example: max_order=2, sequence=["A","B","C","D","E"]
// Windows analyzed:
// ["A","B","C"] -> scores this 3-element window
// ["B","C","D"] -> scores this 3-element window  
// ["C","D","E"] -> scores this 3-element window
```

### Five Scoring Methods:

#### 1. Likelihood Score
```rust
// Finds best probability across all context lengths
for context_len in 1..=max_context_len {
    let context = sequence[i-context_len..i];
    if let Some(prob) = get_transition_probability(context, next_state) {
        let weighted_prob = prob / sqrt(total_support);
        best_prob = max(best_prob, weighted_prob);
    }
}
likelihood *= best_prob;
```
**Lower values = more anomalous**

#### 2. Information Score
```rust
// Combines context entropy with transition surprise
let surprise = -log2(transition_probability);
let score = (total_surprise + context_entropy) / sequence_length;
```
**Higher values = more anomalous**

#### 3. Spectral Score
```rust
// Deviation from expected stationary behavior
let deviation = abs(observed_frequency - expected_stationary_probability);
```

#### 4. Quantum Coherence
```rust
// Simplified coherence measure from density matrix
let coherence = 1.0 - trace(density_matrix) / n_states;
```

#### 5. Topological Signature
- **Connected components**: Number of unique contexts (simplified Betti-0)
- **Cycle detection**: Repeated subsequence patterns
- **Local clustering**: Context interconnectedness measure

## Practical Usage Examples

### Network Security Monitoring

```rust
fn detect_network_anomalies() -> Result<(), String> {
    println!("Starting network traffic anomaly detection example...");

    // Normal traffic patterns (training data)
    let normal_traffic: Vec<String> = vec![ // Corrected type to Vec<String>
        "TCP_SYN",
        "TCP_ACK",
        "HTTP_GET",
        "HTTP_200",
        "TCP_FIN",
        "TCP_SYN",
        "TCP_ACK",
        "HTTPS_POST",
        "HTTP_201",
        "TCP_FIN",
        "UDP_DNS",
        "UDP_RESPONSE",
        "TCP_SYN",
        "TCP_ACK",
        "HTTP_GET",
        "HTTP_200",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    let mut model = AdvancedTransitionModel::new(4);
    //Build the context tree (train the model)
    model.build_context_tree(&normal_traffic)?; // Using '?' operator for error propagation

    println!("Model trained on normal traffic patterns.");

    //Suspicious traffic for anomaly detection
    let suspicious_traffic: Vec<String> = vec![ // Corrected type to Vec<String>
        "TCP_SYN",
        "TCP_RST",
        "TCP_SYN",
        "TCP_RST",
        "TCP_SYN",
        "TCP_RST",// Simulating a port scan
        "HTTP_GET",
        "HTTP_GET",
        "HTTP_GET",
        "HTTP_GET",
        "HTTP_GET",// Simulating a (simplified) DDoS-like pattern
        "UNKNOWN_PROTOCOL",
        "MALFORMED_PACKET",
        "BUFFER_OVERFLOW",// Simulating attack signatures
    ]
    .into_iter()
    .map(String::from)
    .collect();

    // Define a threshold for what constitutes a high-risk anomaly
    // A lower likelihood generally indicates higher anomaly.
    let anomaly_detection_threshold = 1e-6;

    println!("\nAnalyzing suspicious traffic with likelihood threshold: {:.0e}", anomaly_detection_threshold);

    let anomalies: Vec<AnomalyScore> = model.detect_advanced_anomalies(&suspicious_traffic, 0.01); // Note: _threshold parameter in detect_advanced_anomalies is currently unused for filtering, so manual filtering is applied below.

    let mut high_risk_found = false;
    for anomaly in anomalies {
        // Manually filter based on likelihood, as the `_threshold` in `detect_advanced_anomalies`
        // in your `lib.rs` doesn't filter the output.
        if anomaly.likelihood < anomaly_detection_threshold {
            high_risk_found = true;
            println!("\n--- High-risk pattern Detected ---");
            println!("  Sequence: {:?}", anomaly.state_sequence);
            println!("  Risk Score (1 - Likelihood): {:.2e}", 1.0 - anomaly.likelihood);
            println!("  Likelihood: {:.2e}", anomaly.likelihood);
            println!("  Information Score: {:.4}", anomaly.information_theoretic_score);
            println!("  Spectral Score: {:.4}", anomaly.spectral_anomaly_score);
            println!("  Quantum Coherence: {:.4}", anomaly.quantum_coherence_measure);
            println!("  Topological Signature: {:?}", anomaly.topological_signature);
            println!("  Confidence Interval: ({:.2e}, {:.2e})", anomaly.confidence_interval.0, anomaly.confidence_interval.1);
        }
    }

    if !high_risk_found {
        println!("\nNo high-risk patterns detected above the threshold of {:.0e}.", anomaly_detection_threshold);
    }

    println!("\nAnomaly detection example finished.");

    Ok(()) // Indicate successful execution
}
```

### User Behavior Analysis

```rust
fn analyze_user_sessions() -> Result<(), Box<dyn std::error::Error>> {
    // Normal user behavior
    let normal_sessions: Vec<String> = vec![
        // Changed to Vec<String>
        "LOGIN",
        "DASHBOARD",
        "PROFILE",
        "SETTINGS",
        "LOGOUT",
        "LOGIN",
        "SEARCH",
        "VIEW_ITEM",
        "ADD_CART",
        "CHECKOUT",
        "LOGOUT",
        "LOGIN",
        "MESSAGES",
        "COMPOSE",
        "SEND",
        "LOGOUT",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    let mut model = AdvancedTransitionModel::new(3);
    model.build_context_tree(&normal_sessions)?;

    // Detect privilege escalation
    let suspicious_session: Vec<String> = vec![
        // Changed to Vec<String>
        "LOGIN",
        "ADMIN_PANEL",
        "USER_LIST",
        "DELETE_USER",
        "DELETE_USER",
        "BULK_DOWNLOAD",
        "BULK_DOWNLOAD",
        "SENSITIVE_DATA_ACCESS",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    let anomalies = model.detect_advanced_anomalies(&suspicious_session, 0.05);

    // Find behavioral anomalies
    for anomaly in anomalies {
        let risk_score = 1.0 - anomaly.likelihood;
        if risk_score > 0.95 {
            println!("Suspicious behavior: {:?}", anomaly.state_sequence);
            println!("Risk: {:.4}", risk_score);
            println!(
                "Confidence: [{:.2e}, {:.2e}]",
                anomaly.confidence_interval.0, anomaly.confidence_interval.1
            );
        }
    }

    Ok(())
}
```

### Financial Fraud Detection

```rust
fn detect_transaction_fraud() -> Result<(), Box<dyn std::error::Error>> {
    // Normal transaction patterns
    let normal_transactions: Vec<String> = vec![ // Changed to Vec<String>
        "AUTH", "PURCHASE", "CONFIRM", "SETTLEMENT",
        "AUTH", "ATM_WITHDRAWAL", "CONFIRM", "SETTLEMENT",
        "AUTH", "ONLINE_PAYMENT", "CONFIRM", "SETTLEMENT"
    ].into_iter().map(String::from).collect();

    // Repeat patterns for robust training
    let mut training_data = Vec::new();
    for _ in 0..20 {
        training_data.extend(normal_transactions.clone());
    }

    let mut model = AdvancedTransitionModel::new(4);
    model.build_context_tree(&training_data)?;

    // Test suspicious patterns
    let test_transactions: Vec<String> = vec![ // Changed to Vec<String>
        "AUTH", "LARGE_PURCHASE", "FOREIGN_COUNTRY", "CONFIRM",    // Unusual location
        "VELOCITY_ALERT", "AUTH", "AUTH", "AUTH", "AUTH",          // Rapid transactions
        "CARD_NOT_PRESENT", "LARGE_PURCHASE", "DECLINE", "RETRY", "RETRY" // Card testing
    ].into_iter().map(String::from).collect();

    let anomalies = model.detect_advanced_anomalies(&test_transactions, 0.001);

    // Calculate fraud scores
    let mut fraud_alerts: Vec<_> = anomalies.into_iter()
        .map(|a| {
            let fraud_score = (1.0 - a.likelihood) * a.information_theoretic_score;
            (a, fraud_score)
        })
        .filter(|(_, score)| *score > 5.0)
        .collect();

    fraud_alerts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (anomaly, fraud_score) in fraud_alerts.iter().take(5) {
        println!("Fraud alert: {:?}", anomaly.state_sequence);
        println!("Fraud score: {:.4}", fraud_score);
        println!("Quantum coherence: {:.4}", anomaly.quantum_coherence_measure);
    }

    Ok(())
}
```

### System Log Analysis

```rust
fn analyze_system_logs() -> Result<(), Box<dyn std::error::Error>> {
    // Normal system events
    let normal_logs: Vec<String> = vec![ // Changed to Vec<String>
        "BOOT", "SERVICE_START", "AUTH_SUCCESS", "FILE_ACCESS", "NETWORK_CONNECT",
        "CRON_START", "BACKUP_BEGIN", "BACKUP_SUCCESS", "CRON_END",
        "HEALTH_CHECK", "MONITOR_CLEAR", "SERVICE_HEALTHY"
    ].into_iter().map(String::from).collect();

    // Build robust training set
    let mut training_logs = Vec::new();
    for _ in 0..50 {
        training_logs.extend(normal_logs.clone());
    }

    let mut model = AdvancedTransitionModel::new(5);
    model.build_context_tree(&training_logs)?;

    // Test with security incidents
    let mut test_logs = training_logs.clone();
    test_logs.extend(vec![
        "UNAUTHORIZED_ACCESS", "PRIVILEGE_ESCALATION", "FILE_CORRUPTION",
        "SERVICE_CRASH", "SERVICE_CRASH", "SERVICE_CRASH",  //Repeated crashes
        "ROOTKIT_DETECTED", "MALWARE_POSITIVE", "CONFIG_TAMPERED"
    ].into_iter().map(String::from)); // This `vec![]` is fine because `extend` takes an iterator

    let anomalies = model.detect_advanced_anomalies(&test_logs, 0.01);

    // Filter critical anomalies
    let critical: Vec<_> = anomalies.into_iter()
        .filter(|a| a.likelihood < 1e-8)
        .collect();

    for anomaly in critical {
        println!("Critical system event: {:?}", anomaly.state_sequence);
        println!("Severity: {:.2e}", 1.0 / anomaly.likelihood);
        println!("Topological signature: {:?}", anomaly.topological_signature);
    }

    Ok(())
}
```

### DNA Sequence Analysis

```rust
fn analyze_genetic_sequences() -> Result<(), Box<dyn std::error::Error>> {
    // Normal gene patterns (start codon -> coding -> stop codon)
    let normal_genes: Vec<String> = vec![ // Changed to Vec<String>
        "ATG", "CGA", "TTC", "AAG", "GCT", "TAA",  // Gene 1
        "ATG", "CCG", "ATC", "GGC", "TTC", "TAG",  // Gene 2
        "ATG", "GAA", "CTG", "TGC", "CAG", "TGA"   // Gene 3
    ].into_iter().map(String::from).collect();

    // Replicate for statistical significance
    let mut training_dna = Vec::new();
    for _ in 0..100 {
        training_dna.extend(normal_genes.clone());
    }

    let mut model = AdvancedTransitionModel::new(6);  // Longer context for codons
    model.build_context_tree(&training_dna)?;

    // Test with mutations
    let test_dna: Vec<String> = vec![ // Changed to Vec<String>
        "XTG", "CGA", "TTC", "AAG", "GCT", "TAA",  // Invalid nucleotide
        "ATG", "CGA", "TTC", "AAG", "GCT",         // Missing stop codon
        "ATG", "ATG", "ATG", "ATG", "TAA",         // Repeated start codons
        "NNN", "UUU", "QQQ"                        // Invalid sequence
    ].into_iter().map(String::from).collect();

    let mutations = model.detect_advanced_anomalies(&test_dna, 0.01);

    for mutation in mutations {
        let mutation_prob = 1.0 - mutation.likelihood;
        println!("Genetic anomaly: {:?}", mutation.state_sequence);
        println!("Mutation probability: {:.2e}", mutation_prob);

        // Check specific mutation types
        let seq_str = mutation.state_sequence.join("");
        if seq_str.contains("X") || seq_str.contains("N") || seq_str.contains("U") {
            println!("  -> Invalid nucleotide detected");
        }
        if seq_str.starts_with("ATG") && !seq_str.contains("TAA") &&
           !seq_str.contains("TAG") && !seq_str.contains("TGA") {
            println!("  -> Missing stop codon");
        }
    }

    Ok(())
}
```

## Batch Processing

For analyzing multiple sequences in parallel:

```rust
use anomaly_grid::batch_process_sequences;

fn batch_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let sequences = vec![
        // Web server logs
        vec!["GET", "200", "POST", "201", "GET", "404"].iter().map(|s| s.to_string()).collect(),
        // Database operations  
        vec!["CONNECT", "SELECT", "INSERT", "COMMIT"].iter().map(|s| s.to_string()).collect(),
        // Network protocols
        vec!["SYN", "ACK", "DATA", "FIN"].iter().map(|s| s.to_string()).collect(),
    ];
    
    // Process all sequences in parallel
    let results = batch_process_sequences(&sequences, 3, 0.05);
    
    for (i, (original, anomalies)) in sequences.iter().zip(results.iter()).enumerate() {
        println!("Sequence {}: {:?}", i + 1, original);
        println!("Anomalies detected: {}", anomalies.len());
        
        if let Some(most_suspicious) = anomalies.iter()
            .min_by(|a, b| a.likelihood.partial_cmp(&b.likelihood).unwrap()) {
            println!("Most suspicious: {:?} (likelihood: {:.2e})", 
                     most_suspicious.state_sequence, most_suspicious.likelihood);
        }
    }
    
    Ok(())
}
```

## Performance Characteristics

### Computational Complexity:
- **Training**: O(sequence_length × max_order × alphabet_size)
- **Detection**: O(test_length × max_order × log(alphabet_size))
- **Memory**: O(alphabet_size^max_order)

### Recommended Limits:
```rust
//Safe operational parameters, unless you have a NASA mainframe
let max_order = 5;              // Higher orders need exponentially more data
let alphabet_size = 50;         // Keep manageable for memory
let min_sequence_length = 20 * max_order;  // Minimum for stable analysis
let detection_threshold = 0.001; // Balance sensitivity vs false positives
```

### Memory Usage Examples:
```rust
// alphabet_size=10, max_order=3 -> ~1,000 contexts
// alphabet_size=20, max_order=3 -> ~8,000 contexts  
// alphabet_size=50, max_order=3 -> ~125,000 contexts
// alphabet_size=10, max_order=4 -> ~10,000 contexts
```

## Error Handling and Edge Cases

### Robust Training:
```rust
fn safe_training(sequence: &[String], max_order: usize) -> Result<AdvancedTransitionModel, String> {
    if sequence.len() < 20 * max_order {
        return Err("Insufficient training data".to_string());
    }
    
    let unique_states: std::collections::HashSet<_> = sequence.iter().collect();
    if unique_states.len() > 100 {
        return Err("Alphabet too large for efficient processing".to_string());
    }
    
    let mut model = AdvancedTransitionModel::new(max_order);
    
    match model.build_context_tree(sequence) {
        Ok(()) => {
            if model.contexts.is_empty() {
                return Err("No contexts learned from sequence".to_string());
            }
            Ok(model)
        }
        Err(e) => Err(format!("Training failed: {}", e))
    }
}
```

### Spectral Analysis Limitations -- CAUTION :) :
The spectral analysis may fail or provide degraded results when:
- Transition matrix is disconnected (isolated states)
- Very large state spaces (>1000 states)
- Degenerate probability distributions
- Insufficient training data for stable eigenvalue computation

In these cases, the library gracefully degrades by:
- Using uniform distributions as fallbacks
- Skipping spectral analysis while preserving other scores -- working on it
- Providing warning messages instead of crashing -- still working on it

## Testing and Validation

### Unit Tests:
```bash
cargo test test_advanced_anomaly_detection  # Basic functionality
cargo test test_spectral_analysis          # Eigenvalue computation
cargo test test_mathematical_properties    # Probability conservation
```

### Comprehensive Test Suite:
```bash
cargo test -- --nocapture                  # Full test battery with output
```

The test suite includes:
- Network traffic anomaly detection
- User behavior pattern analysis  
- System log analysis
- Financial transaction monitoring
- DNA sequence mutation detection
- Performance benchmarking
- Edge case handling

### Mathematical Validation:
The library validates:
- Probability conservation (all context probabilities sum to 1.0)
- Entropy bounds (0 ≤ entropy ≤ log2(alphabet_size))
- Eigenvalue stability (largest eigenvalue ≤ 1.0)
- Convergence criteria for spectral analysis

## Limitations and Considerations

### Known Limitations:
1. **Exponential memory growth** with context order
2. **Spectral analysis instability** on large/sparse matrices -- huge, I will try to make it better on the go
3. **Requires sufficient training data** (20+ × max_order sequence length) -- Not a problem but there it is
4. **Discrete alphabet assumption** (not optimized for continuous data)
5. **Simplified quantum features** (not full quantum computation) -- Highly speculative as I mentioned, will be removing to continue experiments somewhere else
6. **Basic topological analysis** (not true persistent homology) :)

### Best Practices:
1. Start with `max_order=2` or `3` for initial experiments
2. Ensure training sequences are 5-10x longer than test sequences
3. Monitor memory usage with large alphabets
4. Use batch processing for multiple sequences
5. Validate results with domain knowledge
6. Consider data preprocessing to reduce alphabet size -- Example in the README

### When to Use This Library:
- **Sequential pattern analysis** in discrete event streams
- **Anomaly detection** in time-ordered categorical data
- **Behavioral analysis** where context matters
- **Multi-faceted scoring** combining multiple mathematical approaches
- **Parallel processing** of multiple sequence datasets

### When NOT to Use:
- Continuous numerical data (use other methods for now)
- Very short sequences (<20 elements)  
- Real-time applications requiring microsecond responses
- Datasets with >1000 unique symbols
- Applications requiring explainable AI compliance

---

This manual provides comprehensive coverage of the anomaly-grid library's actual implementation, tested use cases, and practical deployment considerations for experimental systems and not production.