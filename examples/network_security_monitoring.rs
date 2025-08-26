//! Network Security Monitoring Example
//!
//! This example demonstrates real-world network security monitoring using
//! anomaly-grid for detecting Advanced Persistent Threats (APTs), DDoS attacks,
//! and other network anomalies in enterprise environments.

use anomaly_grid::*;

use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🛡️ Network Security Monitoring with Anomaly Grid");
    println!("Detecting APTs, DDoS, and network intrusions\n");

    // Simulate 30 days of normal enterprise network traffic
    let normal_traffic = generate_enterprise_traffic(30);
    println!(
        "Generated {} normal network events (30 days)",
        normal_traffic.len()
    );

    // Initialize anomaly detection system
    let mut detector = AnomalyDetector::new(4)?; // Order 4 for complex patterns

    // Train on normal network traffic patterns
    let train_start = Instant::now();
    detector.train(&normal_traffic)?;
    println!("Training completed in {:?}", train_start.elapsed());

    // Real-time monitoring simulation
    println!("\n🔍 Starting real-time network monitoring...");

    // Test various attack scenarios
    let attack_scenarios = vec![
        ("Advanced Persistent Threat", generate_apt_campaign()),
        ("DDoS Attack", generate_ddos_attack()),
        ("Port Scan", generate_port_scan()),
        ("SQL Injection", generate_sql_injection()),
        ("Lateral Movement", generate_lateral_movement()),
    ];

    for (attack_name, attack_sequence) in attack_scenarios {
        println!("\nTesting: {attack_name}");

        let detect_start = Instant::now();
        let anomalies = detector.detect_anomalies(&attack_sequence, 0.001)?;
        let detect_time = detect_start.elapsed();

        if !anomalies.is_empty() {
            let max_strength = anomalies
                .iter()
                .map(|a| a.anomaly_strength)
                .fold(0.0, f64::max);

            let min_likelihood = anomalies
                .iter()
                .map(|a| a.likelihood)
                .fold(f64::INFINITY, f64::min);

            println!("  ✅ THREAT DETECTED");
            println!("  📊 Anomalies found: {}", anomalies.len());
            println!("  Max anomaly strength: {max_strength:.3}");
            println!("  Min likelihood: {min_likelihood:.2e}");
            println!("  Detection time: {detect_time:?}");

            // Alert classification
            let threat_level = if max_strength > 0.9 {
                "CRITICAL"
            } else if max_strength > 0.7 {
                "HIGH"
            } else if max_strength > 0.5 {
                "MEDIUM"
            } else {
                "LOW"
            };

            println!("  Threat Level: {threat_level}");
        } else {
            println!("  ❌ No anomalies detected");
        }
    }

    // Batch processing demonstration
    println!("\n📦 Batch Processing Demonstration");
    let batch_sequences = vec![
        generate_normal_session(),
        generate_malware_communication(),
        generate_data_exfiltration(),
    ];

    let batch_start = Instant::now();
    let config = AnomalyGridConfig::default().with_max_order(5)?;
    let batch_results = batch_process_sequences(&batch_sequences, &config, 0.01)?;
    let batch_time = batch_start.elapsed();

    println!(
        "Processed {} sequences in {:?}",
        batch_sequences.len(),
        batch_time
    );
    for (i, results) in batch_results.iter().enumerate() {
        println!("  Sequence {}: {} anomalies detected", i + 1, results.len());
    }

    Ok(())
}

fn generate_enterprise_traffic(days: usize) -> Vec<String> {
    let mut traffic = Vec::new();
    let events_per_day = 10000;

    let normal_patterns = vec![
        // HTTP/HTTPS traffic
        vec!["TCP_SYN", "TCP_ACK", "HTTP_GET", "HTTP_200", "TCP_FIN"],
        vec![
            "TCP_SYN",
            "TCP_ACK",
            "TLS_HANDSHAKE",
            "HTTPS_POST",
            "HTTP_201",
            "TCP_FIN",
        ],
        // Email traffic
        vec![
            "TCP_SYN",
            "TCP_ACK",
            "SMTP_HELO",
            "SMTP_AUTH",
            "SMTP_DATA",
            "SMTP_QUIT",
        ],
        vec![
            "TCP_SYN",
            "TCP_ACK",
            "IMAP_LOGIN",
            "IMAP_SELECT",
            "IMAP_FETCH",
            "IMAP_LOGOUT",
        ],
        // DNS queries
        vec!["UDP_DNS_QUERY", "DNS_A_RECORD", "UDP_DNS_RESPONSE"],
        vec!["UDP_DNS_QUERY", "DNS_MX_RECORD", "UDP_DNS_RESPONSE"],
        // Internal services
        vec![
            "TCP_SYN",
            "TCP_ACK",
            "LDAP_BIND",
            "LDAP_SEARCH",
            "LDAP_UNBIND",
        ],
        vec![
            "TCP_SYN",
            "TCP_ACK",
            "SMB_NEGOTIATE",
            "SMB_SESSION",
            "SMB_TREE_CONNECT",
        ],
        // VPN connections
        vec![
            "IPSEC_INIT",
            "IPSEC_AUTH",
            "IPSEC_ESTABLISHED",
            "DATA_TRANSFER",
            "IPSEC_CLOSE",
        ],
    ];

    for _ in 0..days {
        for _ in 0..events_per_day {
            let pattern = &normal_patterns[traffic.len() % normal_patterns.len()];
            traffic.extend(pattern.iter().map(|s| s.to_string()));
        }
    }

    traffic
}

fn generate_apt_campaign() -> Vec<String> {
    vec![
        // Initial compromise
        "SPEAR_PHISHING_EMAIL",
        "MACRO_EXECUTION",
        "PAYLOAD_DOWNLOAD",
        "PERSISTENCE_REGISTRY",
        "SCHEDULED_TASK_CREATE",
        // Reconnaissance
        "NETWORK_DISCOVERY",
        "SERVICE_ENUMERATION",
        "USER_ENUMERATION",
        "DOMAIN_TRUST_DISCOVERY",
        "REMOTE_SYSTEM_DISCOVERY",
        // Lateral movement
        "CREDENTIAL_DUMPING",
        "PASS_THE_HASH",
        "REMOTE_DESKTOP",
        "ADMIN_SHARE_ACCESS",
        "SERVICE_EXECUTION",
        // Data collection
        "FILE_SYSTEM_SEARCH",
        "DATA_STAGING",
        "ARCHIVE_CREATION",
        "SENSITIVE_DATA_ACCESS",
        "DATABASE_QUERY",
        // Exfiltration
        "C2_COMMUNICATION",
        "DATA_ENCRYPTION",
        "EXTERNAL_TRANSFER",
        "DNS_TUNNELING",
        "STEGANOGRAPHY",
        // Cover tracks
        "LOG_DELETION",
        "ARTIFACT_REMOVAL",
        "TIMESTAMP_MODIFICATION",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_ddos_attack() -> Vec<String> {
    vec![
        // Volumetric attack
        "UDP_FLOOD",
        "UDP_FLOOD",
        "UDP_FLOOD",
        "UDP_FLOOD",
        "UDP_FLOOD",
        "ICMP_FLOOD",
        "ICMP_FLOOD",
        "ICMP_FLOOD",
        "ICMP_FLOOD",
        // Protocol attack
        "TCP_SYN_FLOOD",
        "TCP_SYN_FLOOD",
        "TCP_SYN_FLOOD",
        "TCP_SYN_FLOOD",
        "TCP_ACK_FLOOD",
        "TCP_ACK_FLOOD",
        "TCP_ACK_FLOOD",
        // Application layer attack
        "HTTP_GET_FLOOD",
        "HTTP_GET_FLOOD",
        "HTTP_GET_FLOOD",
        "HTTP_GET_FLOOD",
        "HTTP_POST_FLOOD",
        "HTTP_POST_FLOOD",
        "HTTP_POST_FLOOD",
        "SLOWLORIS_ATTACK",
        "SLOWLORIS_ATTACK",
        // Amplification attack
        "DNS_AMPLIFICATION",
        "NTP_AMPLIFICATION",
        "MEMCACHED_AMPLIFICATION",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_port_scan() -> Vec<String> {
    vec![
        "TCP_SYN",
        "TCP_RST",
        "TCP_SYN",
        "TCP_RST",
        "TCP_SYN",
        "TCP_RST",
        "TCP_SYN",
        "TCP_RST",
        "TCP_SYN",
        "TCP_RST",
        "TCP_SYN",
        "TCP_RST",
        "UDP_PROBE",
        "ICMP_UNREACHABLE",
        "UDP_PROBE",
        "ICMP_UNREACHABLE",
        "STEALTH_SCAN",
        "FIN_SCAN",
        "NULL_SCAN",
        "XMAS_SCAN",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_sql_injection() -> Vec<String> {
    vec![
        "TCP_SYN",
        "TCP_ACK",
        "HTTP_POST",
        "SQL_INJECTION_ATTEMPT",
        "ERROR_BASED_SQLI",
        "UNION_BASED_SQLI",
        "BLIND_SQLI",
        "TIME_BASED_SQLI",
        "DATABASE_ERROR",
        "SENSITIVE_DATA_LEAK",
        "HTTP_500",
        "TCP_RST",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_lateral_movement() -> Vec<String> {
    vec![
        "CREDENTIAL_THEFT",
        "PASS_THE_TICKET",
        "GOLDEN_TICKET",
        "SILVER_TICKET",
        "DCSYNC_ATTACK",
        "KERBEROASTING",
        "SMB_RELAY",
        "NTLM_RELAY",
        "REMOTE_EXECUTION",
        "PSEXEC",
        "WMIEXEC",
        "SCHTASKS_ABUSE",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_normal_session() -> Vec<String> {
    vec![
        "TCP_SYN",
        "TCP_ACK",
        "HTTP_GET",
        "HTTP_200",
        "HTTP_GET",
        "HTTP_200",
        "HTTP_POST",
        "HTTP_201",
        "TCP_FIN",
        "TCP_ACK",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_malware_communication() -> Vec<String> {
    vec![
        "DNS_QUERY_SUSPICIOUS",
        "C2_DOMAIN_RESOLUTION",
        "TCP_SYN",
        "TCP_ACK",
        "ENCRYPTED_C2_TRAFFIC",
        "BEACON_HEARTBEAT",
        "COMMAND_DOWNLOAD",
        "PAYLOAD_EXECUTION",
        "DATA_UPLOAD",
        "TCP_FIN",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_data_exfiltration() -> Vec<String> {
    vec![
        "FILE_SYSTEM_ACCESS",
        "SENSITIVE_FILE_READ",
        "DATA_COMPRESSION",
        "ENCRYPTION_PROCESS",
        "EXTERNAL_CONNECTION",
        "LARGE_DATA_TRANSFER",
        "STEGANOGRAPHY_ENCODING",
        "COVERT_CHANNEL",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}
