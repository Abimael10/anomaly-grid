//! Network Security Monitoring Example
//!
//! This example demonstrates real-world network security monitoring using
//! anomaly-grid for detecting Advanced Persistent Threats (APTs), DDoS attacks,
//! and other network anomalies in enterprise environments with improved accuracy.

use anomaly_grid::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🛡️ Network Security Monitoring with Anomaly Grid");
    println!("Detecting APTs, DDoS, and network intrusions with high accuracy\n");

    // Configure detector for network security patterns
    let config = AnomalyGridConfig::default()
        .with_max_order(4)?                    // Higher order for complex attack patterns
        .with_smoothing_alpha(0.5)?            // Lower smoothing for better discrimination
        .with_weights(0.6, 0.4)?;              // Balance likelihood and information

    let mut detector = AnomalyDetector::with_config(config)?;
    println!("✅ Configured security detector with order 4 for complex patterns");

    // Generate comprehensive normal network traffic
    let normal_events = generate_enterprise_traffic(5);
    println!("📊 Generated {} normal network events (5 days)", normal_events.len());

    // Train on normal network patterns
    let train_start = Instant::now();
    detector.train(&normal_events)?;
    let train_time = train_start.elapsed();
    
    let metrics = detector.performance_metrics();
    println!("🎯 Training completed in {:?}", train_time);
    println!("   - Security contexts learned: {}", metrics.context_count);
    println!("   - Memory usage: {:.1} KB", metrics.estimated_memory_bytes as f64 / 1024.0);

    // Real-time monitoring simulation with tuned thresholds
    println!("\n🔍 Real-time Network Security Monitoring");

    let attack_scenarios = vec![
        ("Advanced Persistent Threat", generate_apt_campaign(), 0.001),
        ("DDoS Attack", generate_ddos_attack(), 0.005),
        ("Port Scan", generate_port_scan(), 0.01),
        ("SQL Injection", generate_sql_injection(), 0.005),
        ("Lateral Movement", generate_lateral_movement(), 0.001),
        ("Data Exfiltration", generate_data_exfiltration(), 0.001),
        ("Malware C2 Communication", generate_malware_communication(), 0.002),
    ];

    let mut total_threats_detected = 0;
    let mut critical_threats = 0;
    let mut total_detection_time = std::time::Duration::new(0, 0);

    for (attack_name, attack_sequence, threshold) in attack_scenarios {
        println!("\n--- Analyzing: {} ---", attack_name);
        println!("Sequence length: {} events", attack_sequence.len());
        println!("Detection threshold: {}", threshold);

        let detect_start = Instant::now();
        let anomalies = detector.detect_anomalies(&attack_sequence, threshold)?;
        let detect_time = detect_start.elapsed();
        total_detection_time += detect_time;

        if !anomalies.is_empty() {
            total_threats_detected += 1;

            let max_strength = anomalies.iter()
                .map(|a| a.anomaly_strength)
                .fold(0.0, f64::max);

            let avg_information = anomalies.iter()
                .map(|a| a.information_score)
                .sum::<f64>() / anomalies.len() as f64;

            let min_likelihood = anomalies.iter()
                .map(|a| a.likelihood)
                .fold(f64::INFINITY, f64::min);

            // Enhanced threat classification
            let (threat_level, confidence) = classify_threat(max_strength, avg_information, anomalies.len());
            
            if threat_level == "CRITICAL" {
                critical_threats += 1;
            }

            println!("  🚨 THREAT DETECTED");
            println!("  📊 Anomalies found: {}", anomalies.len());
            println!("  🎯 Max anomaly strength: {:.3}", max_strength);
            println!("  📈 Avg information score: {:.3}", avg_information);
            println!("  📉 Min likelihood: {:.2e}", min_likelihood);
            println!("  ⚡ Detection time: {:?}", detect_time);
            println!("  🔥 Threat Level: {}", threat_level);
            println!("  🎲 Confidence: {:.1}%", confidence);

            // Generate detailed security alert
            generate_security_alert(attack_name, &threat_level, confidence, &anomalies);

            // Show most suspicious patterns
            if anomalies.len() > 0 {
                let most_suspicious = anomalies.iter()
                    .max_by(|a, b| a.anomaly_strength.partial_cmp(&b.anomaly_strength).unwrap())
                    .unwrap();
                println!("  🔍 Most suspicious pattern: {:?}", most_suspicious.sequence);
                println!("     Strength: {:.3}, Info: {:.3}", 
                        most_suspicious.anomaly_strength, 
                        most_suspicious.information_score);
            }
        } else {
            println!("  ✅ No threats detected (normal traffic)");
        }
    }

    // Advanced analytics and ROC analysis
    println!("\n📈 Advanced Security Analytics");
    perform_roc_analysis(&detector)?;

    // Batch processing demonstration with performance metrics
    println!("\n📦 High-Volume Batch Processing");
    let batch_sequences = vec![
        generate_normal_session(),
        generate_malware_communication(),
        generate_data_exfiltration(),
        generate_insider_threat(),
        generate_zero_day_exploit(),
    ];

    let batch_start = Instant::now();
    let config = AnomalyGridConfig::default().with_max_order(5)?;
    let batch_results = batch_process_sequences(&batch_sequences, &config, 0.01)?;
    let batch_time = batch_start.elapsed();

    println!("Processed {} sequences in {:?}", batch_sequences.len(), batch_time);
    let throughput = batch_sequences.len() as f64 / batch_time.as_secs_f64();
    println!("Throughput: {:.1} sequences/second", throughput);
    
    for (i, results) in batch_results.iter().enumerate() {
        let sequence_types = ["Normal Session", "Malware C2", "Data Exfiltration", "Insider Threat", "Zero-day Exploit"];
        let risk_score = calculate_risk_score(results);
        println!("  {}: {} anomalies, risk score: {:.2}", 
                sequence_types[i], results.len(), risk_score);
    }

    // Performance and accuracy summary
    println!("\n📊 Security Monitoring Summary");
    println!("═══════════════════════════════════");
    println!("Threats detected: {}/7 ({:.1}%)", 
            total_threats_detected, 
            (total_threats_detected as f64 / 7.0) * 100.0);
    println!("Critical threats: {}", critical_threats);
    println!("Average detection time: {:?}", 
            total_detection_time / total_threats_detected.max(1) as u32);
    
    let false_positive_rate = calculate_false_positive_rate(&detector)?;
    println!("Estimated false positive rate: {:.2}%", false_positive_rate);
    
    // Calculate estimated cost savings
    let cost_savings = calculate_security_cost_savings(critical_threats, total_threats_detected);
    println!("Estimated cost savings: ${:.0}", cost_savings);

    println!("\n💡 Security Insights:");
    println!("   - APT campaigns show complex multi-stage patterns");
    println!("   - DDoS attacks have high-volume repetitive signatures");
    println!("   - Lateral movement creates subtle context anomalies");
    println!("   - Data exfiltration shows unusual data flow patterns");
    println!("   - Real-time detection enables rapid incident response");

    Ok(())
}

fn generate_enterprise_traffic(days: usize) -> Vec<String> {
    let mut traffic = Vec::new();
    let events_per_day = 8000; // Reduced but still realistic

    let normal_patterns = vec![
        // Standard HTTP/HTTPS workflows
        vec!["TCP_SYN", "TCP_ACK", "TLS_HANDSHAKE", "HTTP_GET", "HTTP_200", "DATA_TRANSFER", "TCP_FIN"],
        vec!["TCP_SYN", "TCP_ACK", "HTTP_POST", "AUTH_SUCCESS", "HTTP_201", "TCP_FIN"],
        
        // Email and collaboration
        vec!["SMTP_CONNECT", "SMTP_AUTH", "SMTP_DATA", "EMAIL_SENT", "SMTP_QUIT"],
        vec!["IMAP_CONNECT", "IMAP_AUTH", "IMAP_SELECT", "EMAIL_FETCH", "IMAP_LOGOUT"],
        
        // DNS and network services
        vec!["DNS_QUERY", "DNS_A_RECORD", "DNS_RESPONSE"],
        vec!["DNS_QUERY", "DNS_MX_RECORD", "DNS_RESPONSE"],
        vec!["DHCP_DISCOVER", "DHCP_OFFER", "DHCP_REQUEST", "DHCP_ACK"],
        
        // Enterprise services
        vec!["LDAP_BIND", "LDAP_SEARCH", "LDAP_RESULT", "LDAP_UNBIND"],
        vec!["SMB_NEGOTIATE", "SMB_SESSION", "SMB_TREE_CONNECT", "FILE_ACCESS", "SMB_CLOSE"],
        vec!["RDP_CONNECT", "RDP_AUTH", "RDP_SESSION", "RDP_DISCONNECT"],
        
        // Security and monitoring
        vec!["FIREWALL_ALLOW", "TRAFFIC_LOG", "CONN_ESTABLISHED"],
        vec!["ANTIVIRUS_SCAN", "FILE_CLEAN", "SCAN_COMPLETE"],
        vec!["IDS_MONITOR", "TRAFFIC_NORMAL", "NO_ALERTS"],
        
        // VPN and remote access
        vec!["VPN_CONNECT", "IPSEC_TUNNEL", "AUTH_SUCCESS", "TUNNEL_ESTABLISHED"],
        vec!["SSL_VPN", "USER_AUTH", "POLICY_APPLIED", "SESSION_ACTIVE"],
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
        // Initial compromise (spear phishing)
        "SPEAR_PHISHING_EMAIL", "MACRO_EXECUTION", "PAYLOAD_DOWNLOAD", 
        "PERSISTENCE_REGISTRY", "SCHEDULED_TASK_CREATE",
        
        // Reconnaissance and discovery
        "NETWORK_DISCOVERY", "SERVICE_ENUMERATION", "USER_ENUMERATION",
        "DOMAIN_TRUST_DISCOVERY", "REMOTE_SYSTEM_DISCOVERY", "AD_ENUMERATION",
        
        // Credential harvesting
        "CREDENTIAL_DUMPING", "LSASS_ACCESS", "SAM_DATABASE_ACCESS",
        "KERBEROS_TICKET_EXTRACTION", "NTLM_HASH_EXTRACTION",
        
        // Lateral movement
        "PASS_THE_HASH", "PASS_THE_TICKET", "REMOTE_DESKTOP", 
        "ADMIN_SHARE_ACCESS", "SERVICE_EXECUTION", "WMI_EXECUTION",
        
        // Privilege escalation
        "EXPLOIT_VULNERABILITY", "TOKEN_IMPERSONATION", "PROCESS_INJECTION",
        "DLL_HIJACKING", "SERVICE_PRIVILEGE_ESCALATION",
        
        // Data collection and staging
        "FILE_SYSTEM_SEARCH", "SENSITIVE_DATA_DISCOVERY", "DATA_STAGING",
        "ARCHIVE_CREATION", "COMPRESSION", "ENCRYPTION",
        
        // Command and control
        "C2_COMMUNICATION", "BEACON_HEARTBEAT", "COMMAND_DOWNLOAD",
        "ENCRYPTED_CHANNEL", "DNS_TUNNELING", "STEGANOGRAPHY",
        
        // Data exfiltration
        "EXTERNAL_TRANSFER", "CLOUD_UPLOAD", "EMAIL_EXFILTRATION",
        "FTP_TRANSFER", "COVERT_CHANNEL",
        
        // Anti-forensics
        "LOG_DELETION", "ARTIFACT_REMOVAL", "TIMESTAMP_MODIFICATION",
        "REGISTRY_CLEANUP", "MEMORY_CLEANUP",
    ].into_iter().map(String::from).collect()
}

fn generate_ddos_attack() -> Vec<String> {
    vec![
        // Volumetric flood
        "UDP_FLOOD", "UDP_FLOOD", "UDP_FLOOD", "UDP_FLOOD", "UDP_FLOOD",
        "ICMP_FLOOD", "ICMP_FLOOD", "ICMP_FLOOD", "ICMP_FLOOD",
        
        // Protocol attacks
        "TCP_SYN_FLOOD", "TCP_SYN_FLOOD", "TCP_SYN_FLOOD", "TCP_SYN_FLOOD",
        "TCP_ACK_FLOOD", "TCP_ACK_FLOOD", "TCP_RST_FLOOD",
        
        // Application layer attacks
        "HTTP_GET_FLOOD", "HTTP_GET_FLOOD", "HTTP_GET_FLOOD",
        "HTTP_POST_FLOOD", "HTTP_POST_FLOOD", "SLOWLORIS_ATTACK",
        "SLOW_POST_ATTACK", "HTTP_HEADER_FLOOD",
        
        // Amplification attacks
        "DNS_AMPLIFICATION", "NTP_AMPLIFICATION", "MEMCACHED_AMPLIFICATION",
        "SSDP_AMPLIFICATION", "CHARGEN_AMPLIFICATION",
        
        // Advanced DDoS techniques
        "BOTNET_COORDINATION", "DISTRIBUTED_ATTACK", "MULTI_VECTOR_ATTACK",
        "ADAPTIVE_ATTACK", "EVASION_TECHNIQUE",
    ].into_iter().map(String::from).collect()
}

fn generate_port_scan() -> Vec<String> {
    vec![
        // TCP connect scan
        "TCP_SYN", "TCP_RST", "TCP_SYN", "TCP_RST", "TCP_SYN", "TCP_RST",
        
        // Stealth scans
        "STEALTH_SCAN", "FIN_SCAN", "NULL_SCAN", "XMAS_SCAN",
        "ACK_SCAN", "WINDOW_SCAN", "MAIMON_SCAN",
        
        // UDP scan
        "UDP_PROBE", "ICMP_UNREACHABLE", "UDP_PROBE", "ICMP_UNREACHABLE",
        
        // Service detection
        "SERVICE_PROBE", "VERSION_DETECTION", "OS_FINGERPRINTING",
        "BANNER_GRABBING", "SERVICE_ENUMERATION",
        
        // Evasion techniques
        "DECOY_SCAN", "FRAGMENTED_PACKETS", "TIMING_EVASION",
        "SOURCE_PORT_MANIPULATION", "IP_SPOOFING",
    ].into_iter().map(String::from).collect()
}

fn generate_sql_injection() -> Vec<String> {
    vec![
        // Initial probing
        "HTTP_POST", "SQL_INJECTION_PROBE", "ERROR_MESSAGE",
        "DATABASE_ERROR", "SYNTAX_ERROR",
        
        // Injection techniques
        "UNION_BASED_SQLI", "ERROR_BASED_SQLI", "BOOLEAN_BASED_SQLI",
        "TIME_BASED_SQLI", "BLIND_SQLI",
        
        // Data extraction
        "SCHEMA_ENUMERATION", "TABLE_ENUMERATION", "COLUMN_ENUMERATION",
        "DATA_EXTRACTION", "SENSITIVE_DATA_LEAK",
        
        // Privilege escalation
        "ADMIN_ACCOUNT_ACCESS", "SYSTEM_COMMAND_EXECUTION",
        "FILE_SYSTEM_ACCESS", "NETWORK_ACCESS",
        
        // Persistence and cleanup
        "BACKDOOR_CREATION", "LOG_MANIPULATION", "EVIDENCE_CLEANUP",
    ].into_iter().map(String::from).collect()
}

fn generate_lateral_movement() -> Vec<String> {
    vec![
        // Credential theft
        "CREDENTIAL_THEFT", "PASSWORD_SPRAYING", "KERBEROASTING",
        "AS_REP_ROASTING", "DCSYNC_ATTACK",
        
        // Authentication attacks
        "PASS_THE_TICKET", "GOLDEN_TICKET", "SILVER_TICKET",
        "OVERPASS_THE_HASH", "PASS_THE_CERTIFICATE",
        
        // Remote execution
        "PSEXEC", "WMIEXEC", "SCHTASKS_ABUSE", "DCOM_EXECUTION",
        "WINRM_EXECUTION", "SSH_LATERAL_MOVEMENT",
        
        // Network protocols abuse
        "SMB_RELAY", "NTLM_RELAY", "LLMNR_POISONING",
        "NBT_NS_POISONING", "RESPONDER_ATTACK",
        
        // Persistence mechanisms
        "SERVICE_CREATION", "REGISTRY_PERSISTENCE", "WMI_PERSISTENCE",
        "STARTUP_PERSISTENCE", "LOGON_SCRIPT_PERSISTENCE",
    ].into_iter().map(String::from).collect()
}

fn generate_data_exfiltration() -> Vec<String> {
    vec![
        // Data discovery
        "FILE_SYSTEM_SEARCH", "DATABASE_ENUMERATION", "NETWORK_SHARE_DISCOVERY",
        "EMAIL_SEARCH", "DOCUMENT_SEARCH", "SENSITIVE_DATA_IDENTIFICATION",
        
        // Data collection
        "FILE_COPY", "DATABASE_DUMP", "EMAIL_EXPORT", "SCREENSHOT_CAPTURE",
        "KEYLOGGER_DATA", "CLIPBOARD_CAPTURE",
        
        // Data preparation
        "DATA_STAGING", "ARCHIVE_CREATION", "COMPRESSION", "ENCRYPTION",
        "DATA_OBFUSCATION", "STEGANOGRAPHY_ENCODING",
        
        // Exfiltration channels
        "HTTP_EXFILTRATION", "HTTPS_EXFILTRATION", "DNS_EXFILTRATION",
        "EMAIL_EXFILTRATION", "FTP_EXFILTRATION", "CLOUD_EXFILTRATION",
        
        // Covert channels
        "ICMP_TUNNELING", "TCP_TUNNELING", "SOCIAL_MEDIA_EXFILTRATION",
        "PHYSICAL_MEDIA_COPY", "PRINTER_EXFILTRATION",
    ].into_iter().map(String::from).collect()
}

fn generate_normal_session() -> Vec<String> {
    vec![
        "USER_LOGIN", "AUTH_SUCCESS", "SESSION_START",
        "APPLICATION_ACCESS", "FILE_READ", "EMAIL_CHECK",
        "WEB_BROWSING", "DOCUMENT_EDIT", "EMAIL_SEND",
        "APPLICATION_CLOSE", "SESSION_END", "USER_LOGOUT",
    ].into_iter().map(String::from).collect()
}

fn generate_malware_communication() -> Vec<String> {
    vec![
        "DNS_QUERY_SUSPICIOUS", "C2_DOMAIN_RESOLUTION", "TCP_CONNECT",
        "TLS_HANDSHAKE_ANOMALY", "ENCRYPTED_C2_TRAFFIC", "BEACON_HEARTBEAT",
        "COMMAND_DOWNLOAD", "PAYLOAD_EXECUTION", "DATA_UPLOAD",
        "PERSISTENCE_CHECK", "ANTI_ANALYSIS", "EVASION_TECHNIQUE",
    ].into_iter().map(String::from).collect()
}

fn generate_insider_threat() -> Vec<String> {
    vec![
        "OFF_HOURS_ACCESS", "UNUSUAL_LOGIN_LOCATION", "PRIVILEGE_ABUSE",
        "UNAUTHORIZED_DATA_ACCESS", "BULK_DATA_DOWNLOAD", "USB_DEVICE_INSERT",
        "PERSONAL_EMAIL_FORWARD", "CLOUD_UPLOAD_PERSONAL", "POLICY_VIOLATION",
        "AUDIT_LOG_ACCESS", "SECURITY_TOOL_DISABLE", "EVIDENCE_DELETION",
    ].into_iter().map(String::from).collect()
}

fn generate_zero_day_exploit() -> Vec<String> {
    vec![
        "UNKNOWN_VULNERABILITY", "EXPLOIT_ATTEMPT", "BUFFER_OVERFLOW",
        "SHELLCODE_EXECUTION", "PRIVILEGE_ESCALATION", "MEMORY_CORRUPTION",
        "CODE_INJECTION", "RETURN_ORIENTED_PROGRAMMING", "HEAP_SPRAY",
        "ANTI_DEP_BYPASS", "ASLR_BYPASS", "SANDBOX_ESCAPE",
    ].into_iter().map(String::from).collect()
}

fn classify_threat(max_strength: f64, avg_information: f64, anomaly_count: usize) -> (String, f64) {
    let base_score = max_strength * 0.4 + (avg_information / 10.0) * 0.3 + (anomaly_count as f64 / 20.0) * 0.3;
    let confidence = (base_score * 100.0).min(99.0);
    
    let threat_level = if base_score > 0.8 {
        "CRITICAL"
    } else if base_score > 0.6 {
        "HIGH"
    } else if base_score > 0.4 {
        "MEDIUM"
    } else {
        "LOW"
    };
    
    (threat_level.to_string(), confidence)
}

fn generate_security_alert(attack_name: &str, threat_level: &str, confidence: f64, anomalies: &[AnomalyScore]) {
    println!("  📋 SECURITY ALERT DETAILS");
    println!("    Attack Type: {}", attack_name);
    println!("    Threat Level: {}", threat_level);
    println!("    Confidence: {:.1}%", confidence);
    println!("    Anomalies: {}", anomalies.len());
    
    let response_action = match threat_level {
        "CRITICAL" => "🚨 IMMEDIATE ISOLATION AND INCIDENT RESPONSE",
        "HIGH" => "⚠️ ESCALATE TO SOC TEAM IMMEDIATELY",
        "MEDIUM" => "📞 NOTIFY SECURITY TEAM WITHIN 1 HOUR",
        _ => "📝 LOG AND CONTINUE MONITORING",
    };
    
    println!("    Recommended Action: {}", response_action);
    
    if threat_level == "CRITICAL" || threat_level == "HIGH" {
        println!("    🔒 Suggested Containment:");
        println!("      - Isolate affected systems");
        println!("      - Preserve forensic evidence");
        println!("      - Activate incident response plan");
        println!("      - Notify stakeholders");
    }
}

fn perform_roc_analysis(detector: &AnomalyDetector) -> Result<(), Box<dyn std::error::Error>> {
    println!("Performing ROC analysis with multiple thresholds...");
    
    let test_cases = vec![
        (generate_normal_session(), false),
        (generate_apt_campaign(), true),
        (generate_ddos_attack(), true),
        (generate_malware_communication(), true),
        (generate_normal_session(), false),
    ];
    
    let thresholds = vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5];
    
    println!("Threshold | True Pos | False Pos | True Neg | False Neg | Accuracy");
    println!("----------|----------|-----------|----------|-----------|----------");
    
    for threshold in thresholds {
        let mut tp = 0; // True positives
        let mut fp = 0; // False positives
        let mut tn = 0; // True negatives
        let mut fn_count = 0; // False negatives
        
        for (sequence, is_malicious) in &test_cases {
            let anomalies = detector.detect_anomalies(sequence, threshold)?;
            let detected = !anomalies.is_empty();
            
            match (detected, *is_malicious) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, false) => tn += 1,
                (false, true) => fn_count += 1,
            }
        }
        
        let accuracy = (tp + tn) as f64 / test_cases.len() as f64 * 100.0;
        
        println!("{:8.3} | {:8} | {:9} | {:8} | {:9} | {:7.1}%",
                threshold, tp, fp, tn, fn_count, accuracy);
    }
    
    Ok(())
}

fn calculate_risk_score(anomalies: &[AnomalyScore]) -> f64 {
    if anomalies.is_empty() {
        return 0.0;
    }
    
    let avg_strength = anomalies.iter().map(|a| a.anomaly_strength).sum::<f64>() / anomalies.len() as f64;
    let max_information = anomalies.iter().map(|a| a.information_score).fold(0.0f64, f64::max);
    
    (avg_strength * 0.6 + (max_information / 10.0) * 0.4) * 10.0
}

fn calculate_false_positive_rate(detector: &AnomalyDetector) -> Result<f64, Box<dyn std::error::Error>> {
    let normal_samples = vec![
        generate_normal_session(),
        generate_normal_session(),
        generate_normal_session(),
    ];
    
    let mut false_positives = 0;
    let threshold = 0.05; // Standard threshold
    
    for sample in normal_samples {
        let anomalies = detector.detect_anomalies(&sample, threshold)?;
        if !anomalies.is_empty() {
            false_positives += 1;
        }
    }
    
    Ok((false_positives as f64 / 3.0) * 100.0)
}

fn calculate_security_cost_savings(critical_threats: usize, total_threats: usize) -> f64 {
    let critical_incident_cost = 500000.0; // Average cost of critical security incident
    let regular_incident_cost = 50000.0;   // Average cost of regular security incident
    
    let critical_savings = critical_threats as f64 * critical_incident_cost;
    let regular_savings = (total_threats - critical_threats) as f64 * regular_incident_cost;
    
    critical_savings + regular_savings
}