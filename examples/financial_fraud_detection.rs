//! Financial Fraud Detection Example
//!
//! This example demonstrates real-world financial fraud detection using
//! anomaly-grid for credit card transactions, identifying velocity attacks,
//! card testing, geographic anomalies, and sophisticated fraud patterns.

use anomaly_grid::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("💳 Financial Fraud Detection with Anomaly Grid");
    println!("Detecting credit card fraud, velocity attacks, and suspicious patterns\n");

    // Generate 6 months of normal transaction data
    let normal_transactions = generate_normal_transactions(180); // 6 months
    println!(
        "Generated {} normal transactions (6 months)",
        normal_transactions.len()
    );

    // Initialize fraud detection system
    let mut detector = AnomalyDetector::new(5)?; // Higher order for complex financial patterns

    // Train on normal transaction patterns
    detector.train(&normal_transactions)?;

    // Real-time fraud detection simulation
    println!("\n🔍 Real-time Fraud Detection Simulation");

    let fraud_scenarios = vec![
        ("Card Testing", generate_card_testing()),
        ("Velocity Attack", generate_velocity_attack()),
        ("Geographic Anomaly", generate_geographic_anomaly()),
        ("Amount Anomaly", generate_amount_anomaly()),
        ("Account Takeover", generate_account_takeover()),
        ("Synthetic Identity", generate_synthetic_identity()),
        ("Merchant Fraud", generate_merchant_fraud()),
    ];

    let mut total_fraud_detected = 0;
    let mut total_fraud_amount = 0.0;

    for (fraud_type, fraud_sequence) in fraud_scenarios {
        println!("\nTesting: {fraud_type}");

        let detect_start = Instant::now();
        let anomalies = detector.detect_anomalies(&fraud_sequence, 0.01)?;
        let detect_time = detect_start.elapsed();

        if !anomalies.is_empty() {
            total_fraud_detected += 1;

            let fraud_score =
                anomalies.iter().map(|a| a.information_score).sum::<f64>() / anomalies.len() as f64;

            let risk_level = classify_risk_level(fraud_score);
            let estimated_amount = estimate_fraud_amount(&fraud_sequence);
            total_fraud_amount += estimated_amount;

            println!("  ✅ FRAUD DETECTED");
            println!("  📊 Anomalies: {}", anomalies.len());
            println!("  Fraud Score: {fraud_score:.2}");
            println!("  Risk Level: {risk_level}");
            println!("  Estimated Amount: ${estimated_amount:.2}");
            println!("  Detection Time: {detect_time:?}");

            // Generate fraud alert
            generate_fraud_alert(fraud_type, &risk_level, estimated_amount, fraud_score);
        } else {
            println!("  ❌ No fraud detected");
        }
    }

    // Performance summary
    println!("\n📊 Fraud Detection Summary");
    println!("Total fraud scenarios detected: {total_fraud_detected}/7");
    println!("Total fraud amount prevented: ${total_fraud_amount:.2}");
    println!(
        "Detection rate: {:.1}%",
        (total_fraud_detected as f64 / 7.0) * 100.0
    );

    // Batch processing for high-volume scenarios
    println!("\n📦 High-Volume Transaction Processing");
    let batch_transactions = generate_mixed_transaction_batch(10000);

    let batch_start = Instant::now();
    let mut fraud_count = 0;
    let mut legitimate_count = 0;

    // Process in chunks for memory efficiency
    for chunk in batch_transactions.chunks(100) {
        let anomalies = detector.detect_anomalies(chunk, 0.05)?;
        if !anomalies.is_empty() {
            fraud_count += 1;
        } else {
            legitimate_count += 1;
        }
    }

    let batch_time = batch_start.elapsed();
    let throughput = batch_transactions.len() as f64 / batch_time.as_secs_f64();

    println!(
        "Processed {} transactions in {:?}",
        batch_transactions.len(),
        batch_time
    );
    println!("Throughput: {throughput:.0} transactions/second");
    println!("Fraud detected: {fraud_count} batches");
    println!("Legitimate: {legitimate_count} batches");

    Ok(())
}

fn generate_normal_transactions(days: usize) -> Vec<String> {
    let mut transactions = Vec::new();
    let transactions_per_day = 1000; // Typical daily volume per account

    let normal_patterns = vec![
        // Regular purchases
        vec![
            "CARD_PRESENT",
            "PIN_VERIFY",
            "MERCHANT_GROCERY",
            "AMOUNT_SMALL",
            "APPROVE",
            "SETTLE",
        ],
        vec![
            "CARD_PRESENT",
            "PIN_VERIFY",
            "MERCHANT_GAS",
            "AMOUNT_MEDIUM",
            "APPROVE",
            "SETTLE",
        ],
        vec![
            "CARD_PRESENT",
            "SIGNATURE",
            "MERCHANT_RESTAURANT",
            "AMOUNT_SMALL",
            "APPROVE",
            "SETTLE",
        ],
        // Online purchases
        vec![
            "CARD_NOT_PRESENT",
            "CVV_VERIFY",
            "MERCHANT_ECOMMERCE",
            "AMOUNT_MEDIUM",
            "APPROVE",
            "SETTLE",
        ],
        vec![
            "CARD_NOT_PRESENT",
            "CVV_VERIFY",
            "MERCHANT_SUBSCRIPTION",
            "AMOUNT_SMALL",
            "APPROVE",
            "SETTLE",
        ],
        // ATM transactions
        vec![
            "ATM_CARD_INSERT",
            "PIN_VERIFY",
            "WITHDRAWAL",
            "AMOUNT_SMALL",
            "APPROVE",
            "DISPENSE",
        ],
        vec![
            "ATM_CARD_INSERT",
            "PIN_VERIFY",
            "BALANCE_INQUIRY",
            "APPROVE",
            "RECEIPT",
        ],
        // Recurring payments
        vec![
            "AUTO_PAY",
            "MERCHANT_UTILITY",
            "AMOUNT_FIXED",
            "APPROVE",
            "SETTLE",
        ],
        vec![
            "AUTO_PAY",
            "MERCHANT_INSURANCE",
            "AMOUNT_FIXED",
            "APPROVE",
            "SETTLE",
        ],
        // Mobile payments
        vec![
            "CONTACTLESS",
            "NFC_TAP",
            "MERCHANT_COFFEE",
            "AMOUNT_SMALL",
            "APPROVE",
            "SETTLE",
        ],
        vec![
            "MOBILE_WALLET",
            "BIOMETRIC_AUTH",
            "MERCHANT_RETAIL",
            "AMOUNT_MEDIUM",
            "APPROVE",
            "SETTLE",
        ],
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
        // Small amount testing
        "CARD_NOT_PRESENT",
        "NO_CVV",
        "MERCHANT_UNKNOWN",
        "AMOUNT_MICRO",
        "DECLINE",
        "CARD_NOT_PRESENT",
        "NO_CVV",
        "MERCHANT_UNKNOWN",
        "AMOUNT_MICRO",
        "DECLINE",
        "CARD_NOT_PRESENT",
        "NO_CVV",
        "MERCHANT_UNKNOWN",
        "AMOUNT_MICRO",
        "APPROVE",
        // Rapid escalation
        "CARD_NOT_PRESENT",
        "CVV_FAIL",
        "MERCHANT_UNKNOWN",
        "AMOUNT_SMALL",
        "DECLINE",
        "CARD_NOT_PRESENT",
        "CVV_FAIL",
        "MERCHANT_UNKNOWN",
        "AMOUNT_MEDIUM",
        "DECLINE",
        "CARD_NOT_PRESENT",
        "CVV_BYPASS",
        "MERCHANT_HIGH_RISK",
        "AMOUNT_LARGE",
        "APPROVE",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_velocity_attack() -> Vec<String> {
    vec![
        // Rapid-fire transactions
        "CARD_NOT_PRESENT",
        "CVV_VERIFY",
        "MERCHANT_ECOMMERCE",
        "AMOUNT_MEDIUM",
        "APPROVE",
        "CARD_NOT_PRESENT",
        "CVV_VERIFY",
        "MERCHANT_ECOMMERCE",
        "AMOUNT_MEDIUM",
        "APPROVE",
        "CARD_NOT_PRESENT",
        "CVV_VERIFY",
        "MERCHANT_ECOMMERCE",
        "AMOUNT_MEDIUM",
        "APPROVE",
        "CARD_NOT_PRESENT",
        "CVV_VERIFY",
        "MERCHANT_ECOMMERCE",
        "AMOUNT_MEDIUM",
        "APPROVE",
        "CARD_NOT_PRESENT",
        "CVV_VERIFY",
        "MERCHANT_ECOMMERCE",
        "AMOUNT_MEDIUM",
        "APPROVE",
        // Multiple merchants simultaneously
        "CARD_NOT_PRESENT",
        "CVV_VERIFY",
        "MERCHANT_RETAIL",
        "AMOUNT_LARGE",
        "APPROVE",
        "CARD_NOT_PRESENT",
        "CVV_VERIFY",
        "MERCHANT_ELECTRONICS",
        "AMOUNT_LARGE",
        "APPROVE",
        "CARD_NOT_PRESENT",
        "CVV_VERIFY",
        "MERCHANT_JEWELRY",
        "AMOUNT_LARGE",
        "APPROVE",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_geographic_anomaly() -> Vec<String> {
    vec![
        // Normal location
        "CARD_PRESENT",
        "PIN_VERIFY",
        "LOCATION_HOME",
        "MERCHANT_GROCERY",
        "AMOUNT_SMALL",
        "APPROVE",
        // Impossible travel
        "CARD_PRESENT",
        "PIN_VERIFY",
        "LOCATION_FOREIGN",
        "MERCHANT_ATM",
        "AMOUNT_LARGE",
        "APPROVE",
        "CARD_NOT_PRESENT",
        "CVV_FAIL",
        "LOCATION_FOREIGN",
        "MERCHANT_LUXURY",
        "AMOUNT_LARGE",
        "APPROVE",
        // High-risk country
        "CARD_NOT_PRESENT",
        "NO_CVV",
        "LOCATION_HIGH_RISK",
        "MERCHANT_UNKNOWN",
        "AMOUNT_LARGE",
        "APPROVE",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_amount_anomaly() -> Vec<String> {
    vec![
        // Normal spending pattern
        "CARD_PRESENT",
        "PIN_VERIFY",
        "MERCHANT_GROCERY",
        "AMOUNT_SMALL",
        "APPROVE",
        "SETTLE",
        "CARD_PRESENT",
        "PIN_VERIFY",
        "MERCHANT_GAS",
        "AMOUNT_SMALL",
        "APPROVE",
        "SETTLE",
        // Sudden large purchase
        "CARD_NOT_PRESENT",
        "CVV_VERIFY",
        "MERCHANT_ELECTRONICS",
        "AMOUNT_EXTREME",
        "APPROVE",
        "SETTLE",
        "CARD_NOT_PRESENT",
        "CVV_VERIFY",
        "MERCHANT_JEWELRY",
        "AMOUNT_EXTREME",
        "APPROVE",
        "SETTLE",
        // Round number amounts (suspicious)
        "CARD_NOT_PRESENT",
        "CVV_BYPASS",
        "MERCHANT_CASH_ADVANCE",
        "AMOUNT_ROUND",
        "APPROVE",
        "SETTLE",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_account_takeover() -> Vec<String> {
    vec![
        // Account information change
        "ONLINE_BANKING",
        "PASSWORD_CHANGE",
        "EMAIL_CHANGE",
        "PHONE_CHANGE",
        // Immediate high-value transactions
        "CARD_NOT_PRESENT",
        "CVV_BYPASS",
        "MERCHANT_MONEY_TRANSFER",
        "AMOUNT_LARGE",
        "APPROVE",
        "CARD_NOT_PRESENT",
        "CVV_BYPASS",
        "MERCHANT_CRYPTO",
        "AMOUNT_LARGE",
        "APPROVE",
        "CARD_NOT_PRESENT",
        "CVV_BYPASS",
        "MERCHANT_GIFT_CARD",
        "AMOUNT_LARGE",
        "APPROVE",
        // Attempt to hide tracks
        "STATEMENT_SUPPRESSION",
        "ALERT_DISABLE",
        "CONTACT_INFO_DELETE",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_synthetic_identity() -> Vec<String> {
    vec![
        // New account with immediate activity
        "ACCOUNT_CREATION",
        "IDENTITY_SYNTHETIC",
        "CREDIT_CHECK_BYPASS",
        // Rapid credit building
        "AUTHORIZED_USER_ADD",
        "CREDIT_LIMIT_INCREASE",
        "PAYMENT_HISTORY_ARTIFICIAL",
        // Bust-out pattern
        "CARD_NOT_PRESENT",
        "CVV_BYPASS",
        "MERCHANT_CASH_ADVANCE",
        "AMOUNT_MAX",
        "APPROVE",
        "CARD_NOT_PRESENT",
        "CVV_BYPASS",
        "MERCHANT_MONEY_TRANSFER",
        "AMOUNT_MAX",
        "APPROVE",
        "CARD_NOT_PRESENT",
        "CVV_BYPASS",
        "MERCHANT_CRYPTO",
        "AMOUNT_MAX",
        "APPROVE",
        // Abandonment
        "CONTACT_LOST",
        "PAYMENT_DEFAULT",
        "ACCOUNT_CLOSURE",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_merchant_fraud() -> Vec<String> {
    vec![
        // Fake transactions
        "CARD_PRESENT",
        "NO_SIGNATURE",
        "MERCHANT_SHELL",
        "AMOUNT_ROUND",
        "APPROVE",
        "CARD_PRESENT",
        "NO_SIGNATURE",
        "MERCHANT_SHELL",
        "AMOUNT_ROUND",
        "APPROVE",
        // Transaction laundering
        "CARD_NOT_PRESENT",
        "CVV_BYPASS",
        "MERCHANT_FRONT",
        "AMOUNT_STRUCTURED",
        "APPROVE",
        "CARD_NOT_PRESENT",
        "CVV_BYPASS",
        "MERCHANT_FRONT",
        "AMOUNT_STRUCTURED",
        "APPROVE",
        // Chargeback fraud
        "REFUND_REQUEST",
        "CHARGEBACK_CLAIM",
        "DISPUTE_FRAUDULENT",
        "MERCHANT_RESPONSE_NONE",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_mixed_transaction_batch(size: usize) -> Vec<String> {
    let mut batch = Vec::new();

    for i in 0..size {
        if i % 100 == 0 {
            // 1% fraud rate
            batch.extend(generate_card_testing());
        } else {
            // Normal transaction
            batch.extend(
                vec![
                    "CARD_PRESENT",
                    "PIN_VERIFY",
                    "MERCHANT_RETAIL",
                    "AMOUNT_MEDIUM",
                    "APPROVE",
                    "SETTLE",
                ]
                .into_iter()
                .map(String::from),
            );
        }
    }

    batch
}

fn classify_risk_level(fraud_score: f64) -> String {
    if fraud_score > 20.0 {
        "CRITICAL".to_string()
    } else if fraud_score > 10.0 {
        "HIGH".to_string()
    } else if fraud_score > 5.0 {
        "MEDIUM".to_string()
    } else {
        "LOW".to_string()
    }
}

fn estimate_fraud_amount(sequence: &[String]) -> f64 {
    // Estimate based on transaction type
    let mut amount = 0.0;

    for event in sequence {
        amount += match event.as_str() {
            "AMOUNT_MICRO" => 1.0,
            "AMOUNT_SMALL" => 50.0,
            "AMOUNT_MEDIUM" => 200.0,
            "AMOUNT_LARGE" => 1000.0,
            "AMOUNT_EXTREME" => 5000.0,
            "AMOUNT_MAX" => 10000.0,
            _ => 0.0,
        };
    }

    amount
}

fn generate_fraud_alert(fraud_type: &str, risk_level: &str, amount: f64, score: f64) {
    println!("  🚨 FRAUD ALERT GENERATED");
    println!("    Type: {fraud_type}");
    println!("    Risk: {risk_level}");
    println!("    Amount: ${amount:.2}");
    println!("    Score: {score:.2}");
    println!(
        "    Action: {}",
        match risk_level {
            "CRITICAL" => "BLOCK CARD IMMEDIATELY",
            "HIGH" => "REQUIRE ADDITIONAL VERIFICATION",
            "MEDIUM" => "FLAG FOR REVIEW",
            _ => "MONITOR CLOSELY",
        }
    );
}
