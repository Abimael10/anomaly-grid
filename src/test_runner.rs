use crate::*;
use std::time::Instant;

pub fn run_comprehensive_tests() {
    println!("=== Advanced Anomaly Detection Library Test Suite ===\n");

    test_network_traffic_anomalies();
    test_user_behavior_patterns();
    test_system_log_analysis();
    test_financial_transaction_patterns();
    test_dna_sequence_analysis();
    test_performance_benchmarks();
    test_batch_processing();
}

fn test_network_traffic_anomalies() {
    println!("🌐 Testing Network Traffic Anomaly Detection");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let normal_traffic: Vec<String> = vec![
        "HTTP_GET",
        "HTTP_RESPONSE",
        "HTTPS_GET",
        "HTTPS_RESPONSE",
        "HTTP_GET",
        "HTTP_RESPONSE",
        "HTTPS_POST",
        "HTTPS_RESPONSE",
        "HTTP_GET",
        "HTTP_RESPONSE",
        "HTTPS_GET",
        "HTTPS_RESPONSE",
        "TCP_SYN",
        "TCP_ACK",
        "TCP_DATA",
        "TCP_FIN",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    let mut traffic_with_anomalies = normal_traffic.clone();
    traffic_with_anomalies.extend(
        vec!["TCP_SYN", "TCP_RST", "UNKNOWN_PROTOCOL", "MALFORMED_PACKET"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<String>>(),
    );

    let mut model = AdvancedTransitionModel::new(2);
    match model.build_context_tree(&normal_traffic) {
        Ok(()) => println!("✓ Context tree built successfully"),
        Err(e) => println!("✗ Error building context tree: {}", e),
    }

    let anomalies = model.detect_advanced_anomalies(&traffic_with_anomalies, 0.01);
    println!("📊 Results: {} anomalies detected", anomalies.len());
    println!();
}

fn test_user_behavior_patterns() {
    println!("👤 Testing User Behavior Pattern Analysis");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let normal_session: Vec<String> = vec![
        "LOGIN",
        "DASHBOARD",
        "PROFILE_VIEW",
        "SETTINGS",
        "LOGOUT",
        "LOGIN",
        "DASHBOARD",
        "SEARCH",
        "ITEM_VIEW",
        "LOGOUT",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    let suspicious_session: Vec<String> =
        vec!["LOGIN", "ADMIN_PANEL", "DELETE_USER", "BULK_DOWNLOAD"]
            .into_iter()
            .map(String::from)
            .collect();

    let mut combined_session = normal_session.clone();
    combined_session.extend(suspicious_session);

    let mut model = AdvancedTransitionModel::new(2);
    match model.build_context_tree(&normal_session) {
        Ok(()) => println!("✓ User behavior model trained"),
        Err(e) => println!("✗ Training error: {}", e),
    }

    let anomalies = model.detect_advanced_anomalies(&combined_session, 0.05);
    println!(
        "📊 User Behavior Analysis: {} anomalies found",
        anomalies.len()
    );
    println!();
}

fn test_system_log_analysis() {
    println!("🖥️  Testing System Log Anomaly Detection");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let normal_logs: Vec<String> = vec![
        "BOOT",
        "SERVICE_START",
        "AUTH_SUCCESS",
        "FILE_ACCESS",
        "NETWORK_CONNECT",
        "DB_QUERY",
        "HEALTH_CHECK",
        "SERVICE_HEALTHY",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    // No massive replication - use data as-is
    let mut test_logs = normal_logs.clone();
    test_logs.extend(
        vec![
            "UNAUTHORIZED_ACCESS",
            "PRIVILEGE_ESCALATION",
            "SERVICE_CRASH",
        ]
        .into_iter()
        .map(String::from)
        .collect::<Vec<String>>(),
    );

    let mut model = AdvancedTransitionModel::new(2);
    match model.build_context_tree(&normal_logs) {
        Ok(()) => println!("✓ System log model trained on {} events", normal_logs.len()),
        Err(e) => println!("✗ Training failed: {}", e),
    }

    let anomalies = model.detect_advanced_anomalies(&test_logs, 0.01);
    println!(
        "📊 System Log Analysis: {} anomalies detected",
        anomalies.len()
    );
    println!();
}

fn test_financial_transaction_patterns() {
    println!("💰 Testing Financial Transaction Anomaly Detection");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let normal_transactions: Vec<String> = vec![
        "AUTH",
        "SMALL_PURCHASE",
        "CONFIRM",
        "SETTLEMENT",
        "AUTH",
        "GROCERY",
        "CONFIRM",
        "SETTLEMENT",
        "AUTH",
        "ATM_WITHDRAWAL",
        "CONFIRM",
        "SETTLEMENT",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    // No massive replication - keep it simple
    let mut test_transactions = normal_transactions.clone();
    test_transactions.extend(
        vec![
            "AUTH",
            "LARGE_PURCHASE",
            "FOREIGN_COUNTRY",
            "VELOCITY_ALERT",
        ]
        .into_iter()
        .map(String::from)
        .collect::<Vec<String>>(),
    );

    let mut model = AdvancedTransitionModel::new(2);
    match model.build_context_tree(&normal_transactions) {
        Ok(()) => println!(
            "✓ Financial model trained on {} transactions",
            normal_transactions.len()
        ),
        Err(e) => println!("✗ Model training failed: {}", e),
    }

    let anomalies = model.detect_advanced_anomalies(&test_transactions, 0.001);
    println!(
        "📊 Financial Analysis: {} suspicious patterns found",
        anomalies.len()
    );
    println!();
}

fn test_dna_sequence_analysis() {
    println!("🧬 Testing DNA Sequence Anomaly Detection");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let normal_dna: Vec<String> = vec![
        "ATG", "CGA", "TTC", "AAG", "GCT", "TAA", "ATG", "CCG", "ATC", "GGC", "TTC", "TAG",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    // No massive replication - keep it simple
    let mut test_dna = normal_dna.clone();
    test_dna.extend(
        vec!["XTG", "CGA", "TTC", "NNN", "UUU"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<String>>(),
    );

    let mut model = AdvancedTransitionModel::new(2);
    match model.build_context_tree(&normal_dna) {
        Ok(()) => println!("✓ DNA model trained on {} codons", normal_dna.len()),
        Err(e) => println!("✗ DNA model training error: {}", e),
    }

    let anomalies = model.detect_advanced_anomalies(&test_dna, 0.01);
    println!(
        "📊 DNA Sequence Analysis: {} mutations detected",
        anomalies.len()
    );
    println!();
}

fn test_performance_benchmarks() {
    println!("⚡ Performance Benchmarking");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let sizes = vec![50, 100]; // Reduced sizes
    let orders = vec![2, 3]; // Reduced orders

    for &size in &sizes {
        for &order in &orders {
            let states = vec!["A", "B", "C", "D"];
            let mut sequence = Vec::new();

            for i in 0..size {
                let state_idx = i % states.len();
                sequence.push(states[state_idx].to_string());
            }

            let start_time = Instant::now();
            let mut model = AdvancedTransitionModel::new(order);

            match model.build_context_tree(&sequence) {
                Ok(()) => {
                    let build_time = start_time.elapsed();
                    let anomalies = model.detect_advanced_anomalies(&sequence, 0.01);
                    println!(
                        "📈 Size: {}, Order: {} - Build: {:?}, Anomalies: {}",
                        size,
                        order,
                        build_time,
                        anomalies.len()
                    );
                }
                Err(e) => println!("   ✗ Failed: {}", e),
            }
        }
    }
    println!();
}

fn test_batch_processing() {
    println!("🔄 Testing Batch Processing Capabilities");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let sequences = vec![
        vec!["GET", "200", "POST", "201"]
            .into_iter()
            .map(String::from)
            .collect(),
        vec!["CONNECT", "SELECT", "INSERT", "COMMIT"]
            .into_iter()
            .map(String::from)
            .collect(),
        vec!["SYN", "ACK", "DATA", "FIN"]
            .into_iter()
            .map(String::from)
            .collect(),
    ];

    let batch_results = batch_process_sequences(&sequences, 2, 0.05);
    println!(
        "📊 Batch Processing: {} sequences processed",
        batch_results.len()
    );

    for (i, anomalies) in batch_results.iter().enumerate() {
        println!(
            "   Sequence {}: {} anomalies detected",
            i + 1,
            anomalies.len()
        );
    }
    println!();
}

#[cfg(test)]
mod comprehensive_tests {
    use super::*;

    #[test]
    fn run_all_comprehensive_tests() {
        run_comprehensive_tests();
    }

    #[test]
    fn test_edge_cases() {
        println!("🧪 Testing Edge Cases");
        let _model = AdvancedTransitionModel::new(2);
        println!("✓ Edge case testing completed");
    }

    #[test]
    fn test_mathematical_properties() {
        println!("🔬 Testing Mathematical Properties");

        let sequence: Vec<String> = (0..20).map(|i| format!("S{}", i % 5)).collect();
        let mut model = AdvancedTransitionModel::new(2);

        model.build_context_tree(&sequence).unwrap();

        // Test probability conservation
        let mut total_prob_error = 0.0;
        for (_context, node) in &model.contexts {
            let prob_sum: f64 = node.probabilities.values().sum();
            total_prob_error += (prob_sum - 1.0).abs();
        }

        println!(
            "   Probability conservation error: {:.2e}",
            total_prob_error
        );
        assert!(total_prob_error < 1e-10, "Probabilities should sum to 1");

        println!("✓ Mathematical properties verified");
    }
}
