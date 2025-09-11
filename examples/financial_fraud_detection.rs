//! Financial Fraud Detection Example
//!
//! This example demonstrates real-world financial fraud detection using
//! anomaly-grid for credit card transactions, identifying velocity attacks,
//! card testing, geographic anomalies, and sophisticated fraud patterns with
//! improved accuracy and realistic thresholds.

use anomaly_grid::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("💳 Financial Fraud Detection with Anomaly Grid");
    println!("Detecting credit card fraud, velocity attacks, and suspicious patterns\n");

    // Configure detector for financial transaction patterns
    let config = AnomalyGridConfig::default()
        .with_max_order(5)?                    // Higher order for complex fraud patterns
        .with_smoothing_alpha(0.8)?            // Moderate smoothing for financial data
        .with_weights(0.5, 0.5)?;              // Balance likelihood and information equally

    let mut detector = AnomalyDetector::with_config(config)?;
    println!("✅ Configured fraud detector with order 5 for complex patterns");

    // Generate comprehensive normal transaction data
    let normal_transactions = generate_normal_transactions(30); // 1 month
    println!("📊 Generated {} normal transactions (1 month)", normal_transactions.len());

    // Train on normal transaction patterns
    let train_start = Instant::now();
    detector.train(&normal_transactions)?;
    let train_time = train_start.elapsed();
    
    let metrics = detector.performance_metrics();
    println!("🎯 Training completed in {:?}", train_time);
    println!("   - Transaction patterns learned: {}", metrics.context_count);
    println!("   - Memory usage: {:.1} KB", metrics.estimated_memory_bytes as f64 / 1024.0);

    // Real-time fraud detection simulation with optimized thresholds
    println!("\n🔍 Real-time Fraud Detection Simulation");

    let fraud_scenarios = vec![
        ("Card Testing", generate_card_testing(), 0.01),
        ("Velocity Attack", generate_velocity_attack(), 0.005),
        ("Geographic Anomaly", generate_geographic_anomaly(), 0.02),
        ("Amount Anomaly", generate_amount_anomaly(), 0.01),
        ("Account Takeover", generate_account_takeover(), 0.001),
        ("Synthetic Identity", generate_synthetic_identity(), 0.001),
        ("Merchant Fraud", generate_merchant_fraud(), 0.005),
        ("Money Laundering", generate_money_laundering(), 0.002),
    ];

    let mut total_fraud_detected = 0;
    let mut total_fraud_amount = 0.0;
    let mut critical_fraud_cases = 0;
    let mut total_detection_time = std::time::Duration::new(0, 0);

    for (fraud_type, fraud_sequence, threshold) in fraud_scenarios {
        println!("\n--- Analyzing: {} ---", fraud_type);
        println!("Transaction sequence length: {}", fraud_sequence.len());
        println!("Detection threshold: {}", threshold);

        let detect_start = Instant::now();
        let anomalies = detector.detect_anomalies(&fraud_sequence, threshold)?;
        let detect_time = detect_start.elapsed();
        total_detection_time += detect_time;

        if !anomalies.is_empty() {
            total_fraud_detected += 1;

            let fraud_score = calculate_fraud_score(&anomalies);
            let (risk_level, confidence) = classify_fraud_risk(fraud_score, anomalies.len());
            let estimated_amount = estimate_fraud_amount(&fraud_sequence);
            total_fraud_amount += estimated_amount;

            if risk_level == "CRITICAL" {
                critical_fraud_cases += 1;
            }

            println!("  🚨 FRAUD DETECTED");
            println!("  📊 Anomalies: {}", anomalies.len());
            println!("  🎯 Fraud Score: {:.2}", fraud_score);
            println!("  🔥 Risk Level: {}", risk_level);
            println!("  🎲 Confidence: {:.1}%", confidence);
            println!("  💰 Estimated Amount: ${:.2}", estimated_amount);
            println!("  ⚡ Detection Time: {:?}", detect_time);

            // Generate detailed fraud alert
            generate_fraud_alert(fraud_type, &risk_level, estimated_amount, fraud_score, confidence);

            // Show most suspicious transaction pattern
            if let Some(most_suspicious) = anomalies.iter()
                .max_by(|a, b| a.anomaly_strength.partial_cmp(&b.anomaly_strength).unwrap()) {
                println!("  🔍 Most suspicious pattern: {:?}", most_suspicious.sequence);
                println!("     Strength: {:.3}, Info: {:.3}", 
                        most_suspicious.anomaly_strength, 
                        most_suspicious.information_score);
            }
        } else {
            println!("  ✅ No fraud detected (legitimate transactions)");
        }
    }

    // Advanced fraud analytics
    println!("\n📈 Advanced Fraud Analytics");
    perform_fraud_roc_analysis(&detector)?;

    // High-volume transaction processing simulation
    println!("\n📦 High-Volume Transaction Processing");
    let batch_transactions = generate_mixed_transaction_batch(1000);

    let batch_start = Instant::now();
    let mut fraud_count = 0;
    let mut legitimate_count = 0;
    let mut total_risk_score = 0.0;

    // Process in chunks for memory efficiency
    for chunk in batch_transactions.chunks(100) {
        let anomalies = detector.detect_anomalies(chunk, 0.05)?;
        if !anomalies.is_empty() {
            fraud_count += 1;
            total_risk_score += calculate_fraud_score(&anomalies);
        } else {
            legitimate_count += 1;
        }
    }

    let batch_time = batch_start.elapsed();
    let throughput = batch_transactions.len() as f64 / batch_time.as_secs_f64();

    println!("Processed {} transactions in {:?}", batch_transactions.len(), batch_time);
    println!("Throughput: {:.0} transactions/second", throughput);
    println!("Fraud detected: {} batches ({:.1}%)", 
            fraud_count, 
            (fraud_count as f64 / (fraud_count + legitimate_count) as f64) * 100.0);
    println!("Legitimate: {} batches", legitimate_count);
    println!("Average risk score: {:.2}", total_risk_score / fraud_count.max(1) as f64);

    // Performance and accuracy summary
    println!("\n📊 Fraud Detection Summary");
    println!("═══════════════════════════════════");
    println!("Fraud scenarios detected: {}/8 ({:.1}%)", 
            total_fraud_detected, 
            (total_fraud_detected as f64 / 8.0) * 100.0);
    println!("Critical fraud cases: {}", critical_fraud_cases);
    println!("Total fraud amount prevented: ${:.2}", total_fraud_amount);
    println!("Average detection time: {:?}", 
            total_detection_time / total_fraud_detected.max(1) as u32);

    // Calculate ROI and cost savings
    let investigation_cost = total_fraud_detected as f64 * 500.0; // $500 per investigation
    let net_savings = total_fraud_amount - investigation_cost;
    let roi = if investigation_cost > 0.0 { (net_savings / investigation_cost) * 100.0 } else { 0.0 };

    println!("Investigation costs: ${:.2}", investigation_cost);
    println!("Net savings: ${:.2}", net_savings);
    println!("ROI: {:.1}%", roi);

    // Calculate precision and recall estimates
    let precision = calculate_fraud_precision(&detector)?;
    let recall = calculate_fraud_recall(&detector)?;
    let f1_score = 2.0 * (precision * recall) / (precision + recall);

    println!("Estimated precision: {:.1}%", precision);
    println!("Estimated recall: {:.1}%", recall);
    println!("F1 score: {:.3}", f1_score);

    println!("\n💡 Fraud Detection Insights:");
    println!("   - Card testing shows rapid small-amount probing patterns");
    println!("   - Velocity attacks have high-frequency transaction bursts");
    println!("   - Geographic anomalies show impossible travel patterns");
    println!("   - Account takeovers exhibit sudden behavior changes");
    println!("   - Synthetic identities show artificial credit building");
    println!("   - Real-time detection enables immediate card blocking");

    Ok(())
}

fn generate_normal_transactions(days: usize) -> Vec<String> {
    let mut transactions = Vec::new();
    let transactions_per_day = 800; // Realistic daily volume per account

    let normal_patterns = vec![
        // Regular retail purchases
        vec!["CARD_PRESENT", "PIN_VERIFY", "MERCHANT_GROCERY", "AMOUNT_SMALL", "LOCATION_HOME", "APPROVE", "SETTLE"],
        vec!["CARD_PRESENT", "SIGNATURE", "MERCHANT_RESTAURANT", "AMOUNT_MEDIUM", "LOCATION_HOME", "APPROVE", "SETTLE"],
        vec!["CARD_PRESENT", "PIN_VERIFY", "MERCHANT_GAS", "AMOUNT_SMALL", "LOCATION_HOME", "APPROVE", "SETTLE"],
        
        // Online purchases
        vec!["CARD_NOT_PRESENT", "CVV_VERIFY", "MERCHANT_ECOMMERCE", "AMOUNT_MEDIUM", "LOCATION_HOME", "APPROVE", "SETTLE"],
        vec!["CARD_NOT_PRESENT", "CVV_VERIFY", "MERCHANT_SUBSCRIPTION", "AMOUNT_SMALL", "LOCATION_HOME", "APPROVE", "SETTLE"],
        
        // ATM transactions
        vec!["ATM_CARD_INSERT", "PIN_VERIFY", "WITHDRAWAL", "AMOUNT_SMALL", "LOCATION_HOME", "APPROVE", "DISPENSE"],
        vec!["ATM_CARD_INSERT", "PIN_VERIFY", "BALANCE_INQUIRY", "LOCATION_HOME", "APPROVE", "RECEIPT"],
        
        // Recurring payments
        vec!["AUTO_PAY", "MERCHANT_UTILITY", "AMOUNT_FIXED", "LOCATION_HOME", "APPROVE", "SETTLE"],
        vec!["AUTO_PAY", "MERCHANT_INSURANCE", "AMOUNT_FIXED", "LOCATION_HOME", "APPROVE", "SETTLE"],
        
        // Mobile payments
        vec!["CONTACTLESS", "NFC_TAP", "MERCHANT_COFFEE", "AMOUNT_SMALL", "LOCATION_WORK", "APPROVE", "SETTLE"],
        vec!["MOBILE_WALLET", "BIOMETRIC_AUTH", "MERCHANT_RETAIL", "AMOUNT_MEDIUM", "LOCATION_HOME", "APPROVE", "SETTLE"],
        
        // Travel patterns
        vec!["CARD_PRESENT", "PIN_VERIFY", "MERCHANT_HOTEL", "AMOUNT_LARGE", "LOCATION_TRAVEL", "APPROVE", "SETTLE"],
        vec!["CARD_PRESENT", "SIGNATURE", "MERCHANT_AIRLINE", "AMOUNT_LARGE", "LOCATION_TRAVEL", "APPROVE", "SETTLE"],
    ];

    for _ in 0..days {
        for _ in 0..transactions_per_day {
            let pattern = &normal_patterns[transactions.len() % normal_patterns.len()];
            transactions.extend(pattern.iter().map(|s| s.to_string()));
        }
    }

    transactions
}

fn generate_card_testing() -> Vec<String> {
    vec![
        // Small amount testing with failures
        "CARD_NOT_PRESENT", "NO_CVV", "MERCHANT_UNKNOWN", "AMOUNT_MICRO", "LOCATION_FOREIGN", "DECLINE", "INVALID_CARD",
        "CARD_NOT_PRESENT", "NO_CVV", "MERCHANT_UNKNOWN", "AMOUNT_MICRO", "LOCATION_FOREIGN", "DECLINE", "INVALID_CVV",
        "CARD_NOT_PRESENT", "CVV_FAIL", "MERCHANT_UNKNOWN", "AMOUNT_MICRO", "LOCATION_FOREIGN", "DECLINE", "INVALID_CVV",
        
        // Successful small test
        "CARD_NOT_PRESENT", "CVV_BYPASS", "MERCHANT_UNKNOWN", "AMOUNT_MICRO", "LOCATION_FOREIGN", "APPROVE", "SETTLE",
        
        // Rapid escalation after success
        "CARD_NOT_PRESENT", "CVV_BYPASS", "MERCHANT_HIGH_RISK", "AMOUNT_MEDIUM", "LOCATION_FOREIGN", "APPROVE", "SETTLE",
        "CARD_NOT_PRESENT", "CVV_BYPASS", "MERCHANT_HIGH_RISK", "AMOUNT_LARGE", "LOCATION_FOREIGN", "APPROVE", "SETTLE",
        "CARD_NOT_PRESENT", "CVV_BYPASS", "MERCHANT_CASH_ADVANCE", "AMOUNT_MAX", "LOCATION_FOREIGN", "APPROVE", "SETTLE",
    ].into_iter().map(String::from).collect()
}

fn generate_velocity_attack() -> Vec<String> {
    vec![
        // Rapid-fire transactions across multiple merchants
        "CARD_NOT_PRESENT", "CVV_VERIFY", "MERCHANT_ECOMMERCE", "AMOUNT_MEDIUM", "LOCATION_HOME", "APPROVE", "SETTLE",
        "CARD_NOT_PRESENT", "CVV_VERIFY", "MERCHANT_RETAIL", "AMOUNT_MEDIUM", "LOCATION_HOME", "APPROVE", "SETTLE",
        "CARD_NOT_PRESENT", "CVV_VERIFY", "MERCHANT_ELECTRONICS", "AMOUNT_LARGE", "LOCATION_HOME", "APPROVE", "SETTLE",
        "CARD_NOT_PRESENT", "CVV_VERIFY", "MERCHANT_JEWELRY", "AMOUNT_LARGE", "LOCATION_HOME", "APPROVE", "SETTLE",
        "CARD_NOT_PRESENT", "CVV_VERIFY", "MERCHANT_LUXURY", "AMOUNT_LARGE", "LOCATION_HOME", "APPROVE", "SETTLE",
        
        // Simultaneous transactions (impossible for single user)
        "CARD_PRESENT", "PIN_VERIFY", "MERCHANT_ATM", "AMOUNT_MAX", "LOCATION_FOREIGN", "APPROVE", "DISPENSE",
        "CARD_NOT_PRESENT", "CVV_VERIFY", "MERCHANT_ONLINE", "AMOUNT_MAX", "LOCATION_HOME", "APPROVE", "SETTLE",
        "CARD_PRESENT", "SIGNATURE", "MERCHANT_RETAIL", "AMOUNT_MAX", "LOCATION_TRAVEL", "APPROVE", "SETTLE",
    ].into_iter().map(String::from).collect()
}

fn generate_geographic_anomaly() -> Vec<String> {
    vec![
        // Normal home location
        "CARD_PRESENT", "PIN_VERIFY", "MERCHANT_GROCERY", "AMOUNT_SMALL", "LOCATION_HOME", "APPROVE", "SETTLE",
        
        // Impossible travel (same day, different continents)
        "CARD_PRESENT", "PIN_VERIFY", "MERCHANT_ATM", "AMOUNT_LARGE", "LOCATION_ASIA", "APPROVE", "DISPENSE",
        "CARD_NOT_PRESENT", "CVV_FAIL", "MERCHANT_UNKNOWN", "AMOUNT_LARGE", "LOCATION_EUROPE", "APPROVE", "SETTLE",
        
        // High-risk country transactions
        "CARD_NOT_PRESENT", "NO_CVV", "MERCHANT_HIGH_RISK", "AMOUNT_LARGE", "LOCATION_HIGH_RISK", "APPROVE", "SETTLE",
        "CARD_NOT_PRESENT", "CVV_BYPASS", "MERCHANT_MONEY_TRANSFER", "AMOUNT_MAX", "LOCATION_HIGH_RISK", "APPROVE", "SETTLE",
        
        // Return to normal location (suspicious timing)
        "CARD_PRESENT", "PIN_VERIFY", "MERCHANT_GAS", "AMOUNT_SMALL", "LOCATION_HOME", "APPROVE", "SETTLE",
    ].into_iter().map(String::from).collect()
}

fn generate_amount_anomaly() -> Vec<String> {
    vec![
        // Normal spending pattern
        "CARD_PRESENT", "PIN_VERIFY", "MERCHANT_GROCERY", "AMOUNT_SMALL", "LOCATION_HOME", "APPROVE", "SETTLE",
        "CARD_PRESENT", "SIGNATURE", "MERCHANT_RESTAURANT", "AMOUNT_SMALL", "LOCATION_HOME", "APPROVE", "SETTLE",
        
        // Sudden large purchases (unusual for this account)
        "CARD_NOT_PRESENT", "CVV_VERIFY", "MERCHANT_ELECTRONICS", "AMOUNT_EXTREME", "LOCATION_HOME", "APPROVE", "SETTLE",
        "CARD_NOT_PRESENT", "CVV_VERIFY", "MERCHANT_JEWELRY", "AMOUNT_EXTREME", "LOCATION_HOME", "APPROVE", "SETTLE",
        
        // Round number amounts (suspicious)
        "CARD_NOT_PRESENT", "CVV_BYPASS", "MERCHANT_CASH_ADVANCE", "AMOUNT_ROUND", "LOCATION_HOME", "APPROVE", "SETTLE",
        "CARD_NOT_PRESENT", "CVV_BYPASS", "MERCHANT_MONEY_TRANSFER", "AMOUNT_ROUND", "LOCATION_HOME", "APPROVE", "SETTLE",
        
        // Maximum limit testing
        "CARD_NOT_PRESENT", "CVV_BYPASS", "MERCHANT_CRYPTO", "AMOUNT_MAX", "LOCATION_HOME", "APPROVE", "SETTLE",
    ].into_iter().map(String::from).collect()
}

fn generate_account_takeover() -> Vec<String> {
    vec![
        // Account information changes
        "ONLINE_BANKING", "PASSWORD_CHANGE", "SECURITY_QUESTION_CHANGE", "EMAIL_CHANGE", "PHONE_CHANGE",
        
        // Immediate high-value transactions after takeover
        "CARD_NOT_PRESENT", "CVV_BYPASS", "MERCHANT_MONEY_TRANSFER", "AMOUNT_LARGE", "LOCATION_FOREIGN", "APPROVE", "SETTLE",
        "CARD_NOT_PRESENT", "CVV_BYPASS", "MERCHANT_CRYPTO", "AMOUNT_LARGE", "LOCATION_FOREIGN", "APPROVE", "SETTLE",
        "CARD_NOT_PRESENT", "CVV_BYPASS", "MERCHANT_GIFT_CARD", "AMOUNT_LARGE", "LOCATION_FOREIGN", "APPROVE", "SETTLE",
        
        // Attempt to hide tracks
        "STATEMENT_SUPPRESSION", "ALERT_DISABLE", "CONTACT_INFO_DELETE", "TRANSACTION_HISTORY_DELETE",
        
        // Cash-out attempts
        "CARD_PRESENT", "PIN_BYPASS", "MERCHANT_ATM", "AMOUNT_MAX", "LOCATION_FOREIGN", "APPROVE", "DISPENSE",
        "CARD_NOT_PRESENT", "CVV_BYPASS", "MERCHANT_WIRE_TRANSFER", "AMOUNT_MAX", "LOCATION_FOREIGN", "APPROVE", "SETTLE",
    ].into_iter().map(String::from).collect()
}

fn generate_synthetic_identity() -> Vec<String> {
    vec![
        // New account with suspicious characteristics
        "ACCOUNT_CREATION", "IDENTITY_SYNTHETIC", "CREDIT_CHECK_BYPASS", "INSTANT_APPROVAL",
        
        // Rapid credit building
        "AUTHORIZED_USER_ADD", "CREDIT_LIMIT_INCREASE", "PAYMENT_HISTORY_ARTIFICIAL", "CREDIT_SCORE_MANIPULATION",
        
        // Bust-out pattern (maximize credit then disappear)
        "CARD_NOT_PRESENT", "CVV_BYPASS", "MERCHANT_CASH_ADVANCE", "AMOUNT_MAX", "LOCATION_UNKNOWN", "APPROVE", "SETTLE",
        "CARD_NOT_PRESENT", "CVV_BYPASS", "MERCHANT_MONEY_TRANSFER", "AMOUNT_MAX", "LOCATION_UNKNOWN", "APPROVE", "SETTLE",
        "CARD_NOT_PRESENT", "CVV_BYPASS", "MERCHANT_CRYPTO", "AMOUNT_MAX", "LOCATION_UNKNOWN", "APPROVE", "SETTLE",
        "CARD_NOT_PRESENT", "CVV_BYPASS", "MERCHANT_GIFT_CARD", "AMOUNT_MAX", "LOCATION_UNKNOWN", "APPROVE", "SETTLE",
        
        // Abandonment
        "CONTACT_LOST", "PAYMENT_DEFAULT", "COLLECTION_ATTEMPT", "ACCOUNT_CLOSURE", "IDENTITY_DISAPPEARS",
    ].into_iter().map(String::from).collect()
}

fn generate_merchant_fraud() -> Vec<String> {
    vec![
        // Fake transactions (merchant collusion)
        "CARD_PRESENT", "NO_SIGNATURE", "MERCHANT_SHELL", "AMOUNT_ROUND", "LOCATION_UNKNOWN", "APPROVE", "SETTLE",
        "CARD_PRESENT", "NO_PIN", "MERCHANT_SHELL", "AMOUNT_ROUND", "LOCATION_UNKNOWN", "APPROVE", "SETTLE",
        
        // Transaction laundering
        "CARD_NOT_PRESENT", "CVV_BYPASS", "MERCHANT_FRONT", "AMOUNT_STRUCTURED", "LOCATION_UNKNOWN", "APPROVE", "SETTLE",
        "CARD_NOT_PRESENT", "CVV_BYPASS", "MERCHANT_FRONT", "AMOUNT_STRUCTURED", "LOCATION_UNKNOWN", "APPROVE", "SETTLE",
        
        // Chargeback fraud
        "REFUND_REQUEST", "CHARGEBACK_CLAIM", "DISPUTE_FRAUDULENT", "MERCHANT_RESPONSE_NONE", "CHARGEBACK_WIN",
        
        // Money laundering through transactions
        "CARD_NOT_PRESENT", "CVV_BYPASS", "MERCHANT_MONEY_SERVICE", "AMOUNT_LARGE", "LOCATION_HIGH_RISK", "APPROVE", "SETTLE",
        "CARD_NOT_PRESENT", "CVV_BYPASS", "MERCHANT_CASINO", "AMOUNT_LARGE", "LOCATION_HIGH_RISK", "APPROVE", "SETTLE",
    ].into_iter().map(String::from).collect()
}

fn generate_money_laundering() -> Vec<String> {
    vec![
        // Structuring (keeping amounts below reporting thresholds)
        "CARD_NOT_PRESENT", "CVV_VERIFY", "MERCHANT_MONEY_TRANSFER", "AMOUNT_STRUCTURED", "LOCATION_HOME", "APPROVE", "SETTLE",
        "CARD_NOT_PRESENT", "CVV_VERIFY", "MERCHANT_MONEY_TRANSFER", "AMOUNT_STRUCTURED", "LOCATION_HOME", "APPROVE", "SETTLE",
        "CARD_NOT_PRESENT", "CVV_VERIFY", "MERCHANT_MONEY_TRANSFER", "AMOUNT_STRUCTURED", "LOCATION_HOME", "APPROVE", "SETTLE",
        
        // Rapid movement through multiple accounts
        "CARD_NOT_PRESENT", "CVV_VERIFY", "MERCHANT_BANK_TRANSFER", "AMOUNT_LARGE", "LOCATION_FOREIGN", "APPROVE", "SETTLE",
        "CARD_NOT_PRESENT", "CVV_VERIFY", "MERCHANT_CRYPTO_EXCHANGE", "AMOUNT_LARGE", "LOCATION_FOREIGN", "APPROVE", "SETTLE",
        "CARD_NOT_PRESENT", "CVV_VERIFY", "MERCHANT_CASINO", "AMOUNT_LARGE", "LOCATION_FOREIGN", "APPROVE", "SETTLE",
        
        // Complex layering
        "CARD_NOT_PRESENT", "CVV_VERIFY", "MERCHANT_SHELL_COMPANY", "AMOUNT_LARGE", "LOCATION_OFFSHORE", "APPROVE", "SETTLE",
        "CARD_NOT_PRESENT", "CVV_VERIFY", "MERCHANT_REAL_ESTATE", "AMOUNT_EXTREME", "LOCATION_OFFSHORE", "APPROVE", "SETTLE",
    ].into_iter().map(String::from).collect()
}

fn generate_mixed_transaction_batch(size: usize) -> Vec<String> {
    let mut batch = Vec::new();

    for i in 0..size {
        if i % 100 == 0 {
            // 1% fraud rate
            batch.extend(generate_card_testing());
        } else if i % 200 == 0 {
            // 0.5% velocity attacks
            batch.extend(generate_velocity_attack());
        } else {
            // Normal transaction
            batch.extend(vec![
                "CARD_PRESENT", "PIN_VERIFY", "MERCHANT_RETAIL", "AMOUNT_MEDIUM", 
                "LOCATION_HOME", "APPROVE", "SETTLE"
            ].into_iter().map(String::from));
        }
    }

    batch
}

fn calculate_fraud_score(anomalies: &[AnomalyScore]) -> f64 {
    if anomalies.is_empty() {
        return 0.0;
    }
    
    let avg_strength = anomalies.iter().map(|a| a.anomaly_strength).sum::<f64>() / anomalies.len() as f64;
    let max_information = anomalies.iter().map(|a| a.information_score).fold(0.0f64, f64::max);
    let anomaly_density = anomalies.len() as f64 / 10.0; // Normalize by expected sequence length
    
    (avg_strength * 0.4 + (max_information / 10.0) * 0.4 + anomaly_density * 0.2) * 10.0
}

fn classify_fraud_risk(fraud_score: f64, anomaly_count: usize) -> (String, f64) {
    let base_confidence = (fraud_score * 10.0).min(95.0);
    let count_bonus = (anomaly_count as f64 / 5.0).min(5.0);
    let confidence = (base_confidence + count_bonus).min(99.0);
    
    let risk_level = if fraud_score > 8.0 {
        "CRITICAL"
    } else if fraud_score > 6.0 {
        "HIGH"
    } else if fraud_score > 4.0 {
        "MEDIUM"
    } else {
        "LOW"
    };
    
    (risk_level.to_string(), confidence)
}

fn estimate_fraud_amount(sequence: &[String]) -> f64 {
    let mut amount = 0.0;

    for event in sequence {
        amount += match event.as_str() {
            "AMOUNT_MICRO" => 1.0,
            "AMOUNT_SMALL" => 75.0,
            "AMOUNT_MEDIUM" => 250.0,
            "AMOUNT_LARGE" => 1500.0,
            "AMOUNT_EXTREME" => 7500.0,
            "AMOUNT_MAX" => 15000.0,
            "AMOUNT_ROUND" => 5000.0,
            "AMOUNT_STRUCTURED" => 9500.0, // Just under reporting threshold
            "AMOUNT_FIXED" => 150.0,
            _ => 0.0,
        };
    }

    amount
}

fn generate_fraud_alert(fraud_type: &str, risk_level: &str, amount: f64, score: f64, confidence: f64) {
    println!("  📋 FRAUD ALERT DETAILS");
    println!("    Type: {}", fraud_type);
    println!("    Risk: {}", risk_level);
    println!("    Amount: ${:.2}", amount);
    println!("    Score: {:.2}", score);
    println!("    Confidence: {:.1}%", confidence);
    
    let action = match risk_level {
        "CRITICAL" => "🚨 BLOCK CARD IMMEDIATELY AND CONTACT CUSTOMER",
        "HIGH" => "⚠️ REQUIRE ADDITIONAL VERIFICATION",
        "MEDIUM" => "📞 FLAG FOR MANUAL REVIEW",
        _ => "📝 LOG AND MONITOR CLOSELY",
    };
    
    println!("    Action: {}", action);
    
    if risk_level == "CRITICAL" || risk_level == "HIGH" {
        println!("    🔒 Immediate Steps:");
        println!("      - Freeze card temporarily");
        println!("      - Send SMS/email alert to customer");
        println!("      - Require identity verification");
        println!("      - Review recent transactions");
    }
}

fn perform_fraud_roc_analysis(detector: &AnomalyDetector) -> Result<(), Box<dyn std::error::Error>> {
    println!("Performing fraud detection ROC analysis...");
    
    let test_cases = vec![
        (generate_normal_transactions(1), false),
        (generate_card_testing(), true),
        (generate_velocity_attack(), true),
        (generate_account_takeover(), true),
        (generate_normal_transactions(1), false),
        (generate_synthetic_identity(), true),
    ];
    
    let thresholds = vec![0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2];
    
    println!("Threshold | Precision | Recall | F1-Score | Accuracy");
    println!("----------|-----------|--------|----------|----------");
    
    for threshold in thresholds {
        let mut tp = 0; // True positives
        let mut fp = 0; // False positives
        let mut tn = 0; // True negatives
        let mut fn_count = 0; // False negatives
        
        for (sequence, is_fraud) in &test_cases {
            let anomalies = detector.detect_anomalies(sequence, threshold)?;
            let detected = !anomalies.is_empty();
            
            match (detected, *is_fraud) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, false) => tn += 1,
                (false, true) => fn_count += 1,
            }
        }
        
        let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let recall = if tp + fn_count > 0 { tp as f64 / (tp + fn_count) as f64 } else { 0.0 };
        let f1_score = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
        let accuracy = (tp + tn) as f64 / test_cases.len() as f64;
        
        println!("{:8.3} | {:8.3} | {:6.3} | {:8.3} | {:7.3}",
                threshold, precision, recall, f1_score, accuracy);
    }
    
    Ok(())
}

fn calculate_fraud_precision(detector: &AnomalyDetector) -> Result<f64, Box<dyn std::error::Error>> {
    let fraud_samples = vec![generate_card_testing(), generate_velocity_attack()];
    let normal_samples = vec![generate_normal_transactions(1), generate_normal_transactions(1)];
    
    let mut tp = 0;
    let mut fp = 0;
    let threshold = 0.01;
    
    for sample in fraud_samples {
        let anomalies = detector.detect_anomalies(&sample, threshold)?;
        if !anomalies.is_empty() {
            tp += 1;
        }
    }
    
    for sample in normal_samples {
        let anomalies = detector.detect_anomalies(&sample, threshold)?;
        if !anomalies.is_empty() {
            fp += 1;
        }
    }
    
    if tp + fp > 0 {
        Ok((tp as f64 / (tp + fp) as f64) * 100.0)
    } else {
        Ok(0.0)
    }
}

fn calculate_fraud_recall(detector: &AnomalyDetector) -> Result<f64, Box<dyn std::error::Error>> {
    let fraud_samples = vec![
        generate_card_testing(), 
        generate_velocity_attack(), 
        generate_account_takeover()
    ];
    
    let mut tp = 0;
    let mut fn_count = 0;
    let threshold = 0.01;
    
    for sample in fraud_samples {
        let anomalies = detector.detect_anomalies(&sample, threshold)?;
        if !anomalies.is_empty() {
            tp += 1;
        } else {
            fn_count += 1;
        }
    }
    
    if tp + fn_count > 0 {
        Ok((tp as f64 / (tp + fn_count) as f64) * 100.0)
    } else {
        Ok(0.0)
    }
}