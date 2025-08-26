//! System Log Analysis Example
//!
//! This example demonstrates comprehensive system log analysis using
//! anomaly-grid for detecting security incidents, system failures,
//! and operational anomalies in enterprise IT environments.

use anomaly_grid::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🖥️ System Log Analysis with Anomaly Grid");
    println!("Detecting security incidents and system anomalies\n");

    // Generate 90 days of normal system logs
    let normal_logs = generate_normal_system_logs(90);
    println!(
        "Generated {} normal log entries (90 days)",
        normal_logs.len()
    );

    // Initialize anomaly detection system
    let mut detector = AnomalyDetector::new(4)?; // Order 4 for complex log patterns

    // Train on normal system behavior
    detector.train(&normal_logs)?;

    // Security incident detection
    println!("\n🔒 Security Incident Detection");

    let security_scenarios = vec![
        ("Malware Infection", generate_malware_infection()),
        ("Privilege Escalation", generate_privilege_escalation()),
        ("Data Exfiltration", generate_data_exfiltration()),
        ("Brute Force Attack", generate_brute_force_attack()),
        ("Insider Threat", generate_insider_threat()),
        ("DDoS Attack", generate_ddos_logs()),
        ("SQL Injection", generate_sql_injection_logs()),
    ];

    let mut incidents_detected = 0;
    let mut total_threat_score = 0.0;

    for (incident_type, log_sequence) in security_scenarios {
        println!("\nAnalyzing: {incident_type}");

        let detect_start = Instant::now();
        let anomalies = detector.detect_anomalies(&log_sequence, 0.001)?;
        let detect_time = detect_start.elapsed();

        if !anomalies.is_empty() {
            incidents_detected += 1;

            let threat_score = calculate_threat_score(&anomalies);
            total_threat_score += threat_score;

            let severity = classify_security_severity(threat_score);
            let confidence = calculate_confidence(&anomalies);

            println!("  ⚠️ SECURITY INCIDENT DETECTED");
            println!("  📊 Anomalies: {}", anomalies.len());
            println!("  Threat Score: {threat_score:.2}");
            println!("  Severity: {severity}");
            println!("  Confidence: {confidence:.1}%");
            println!("  Detection Time: {detect_time:?}");

            generate_security_alert(incident_type, &severity, threat_score, confidence);
        } else {
            println!("  ✅ No security incidents detected");
        }
    }

    // System health monitoring
    println!("\n🏥 System Health Monitoring");

    let system_scenarios = vec![
        ("Memory Leak", generate_memory_leak()),
        ("Disk Space Issue", generate_disk_space_issue()),
        ("Network Congestion", generate_network_congestion()),
        ("Service Failure", generate_service_failure()),
        (
            "Performance Degradation",
            generate_performance_degradation(),
        ),
    ];

    let mut health_issues = 0;

    for (issue_type, log_sequence) in system_scenarios {
        println!("\nMonitoring: {issue_type}");

        let detect_start = Instant::now();
        let anomalies = detector.detect_anomalies(&log_sequence, 0.01)?;
        let detect_time = detect_start.elapsed();

        if !anomalies.is_empty() {
            health_issues += 1;

            let impact_score = calculate_impact_score(&anomalies);
            let urgency = classify_urgency(impact_score);

            println!("  ⚠️ SYSTEM ISSUE DETECTED");
            println!("  Impact Score: {impact_score:.2}");
            println!("  Urgency: {urgency}");
            println!("  Detection Time: {detect_time:?}");

            generate_system_alert(issue_type, &urgency, impact_score);
        } else {
            println!("  ✅ System operating normally");
        }
    }

    // Compliance monitoring
    println!("\n📋 Compliance Monitoring");
    let compliance_logs = generate_compliance_scenario();

    let compliance_start = Instant::now();
    let compliance_anomalies = detector.detect_anomalies(&compliance_logs, 0.02)?;
    let compliance_time = compliance_start.elapsed();

    if !compliance_anomalies.is_empty() {
        let compliance_risk = calculate_compliance_risk(&compliance_anomalies);
        println!(
            "Compliance violations detected: {}",
            compliance_anomalies.len()
        );
        println!("Compliance risk score: {compliance_risk:.2}");
        println!("Detection time: {compliance_time:?}");

        if compliance_risk > 5.0 {
            println!("🚨 HIGH COMPLIANCE RISK - IMMEDIATE ATTENTION REQUIRED");
        }
    } else {
        println!("✅ No compliance violations detected");
    }

    // Real-time log streaming simulation
    println!("\n📡 Real-time Log Streaming Simulation");
    let stream_logs = generate_log_stream(1000);

    let stream_start = Instant::now();
    let mut stream_anomalies = 0;

    // Process logs in real-time chunks
    for chunk in stream_logs.chunks(50) {
        let anomalies = detector.detect_anomalies(chunk, 0.05)?;
        if !anomalies.is_empty() {
            stream_anomalies += 1;
        }
    }

    let stream_time = stream_start.elapsed();
    let throughput = stream_logs.len() as f64 / stream_time.as_secs_f64();

    println!(
        "Processed {} log entries in {:?}",
        stream_logs.len(),
        stream_time
    );
    println!("Throughput: {throughput:.0} logs/second");
    println!("Anomalous chunks: {stream_anomalies}");

    // Summary and metrics
    println!("\n📊 Log Analysis Summary");
    println!("Security incidents detected: {incidents_detected}/7");
    println!("System health issues: {health_issues}/5");
    println!(
        "Average threat score: {:.2}",
        total_threat_score / incidents_detected.max(1) as f64
    );

    let detection_rate = (incidents_detected + health_issues) as f64 / 12.0 * 100.0;
    println!("Overall detection rate: {detection_rate:.1}%");

    Ok(())
}

fn generate_normal_system_logs(days: usize) -> Vec<String> {
    let mut logs = Vec::new();
    let logs_per_day = 10000; // Typical enterprise volume

    let normal_patterns = vec![
        // Authentication logs
        vec![
            "USER_LOGIN",
            "AUTH_SUCCESS",
            "SESSION_START",
            "USER_ACTIVITY",
            "SESSION_END",
            "USER_LOGOUT",
        ],
        vec![
            "SERVICE_AUTH",
            "KERBEROS_SUCCESS",
            "TOKEN_ISSUED",
            "ACCESS_GRANTED",
        ],
        // System operations
        vec![
            "SERVICE_START",
            "HEALTH_CHECK",
            "STATUS_OK",
            "SERVICE_RUNNING",
        ],
        vec![
            "BACKUP_START",
            "DATA_BACKUP",
            "BACKUP_SUCCESS",
            "BACKUP_COMPLETE",
        ],
        vec!["LOG_ROTATION", "ARCHIVE_LOGS", "CLEANUP_COMPLETE"],
        // Network activity
        vec![
            "CONNECTION_ESTABLISHED",
            "DATA_TRANSFER",
            "CONNECTION_CLOSED",
        ],
        vec!["DNS_QUERY", "DNS_RESPONSE", "RESOLUTION_SUCCESS"],
        // Application logs
        vec![
            "APP_START",
            "CONFIG_LOADED",
            "DATABASE_CONNECT",
            "APP_READY",
        ],
        vec![
            "REQUEST_RECEIVED",
            "PROCESSING",
            "RESPONSE_SENT",
            "REQUEST_COMPLETE",
        ],
        // Security logs
        vec!["FIREWALL_ALLOW", "TRAFFIC_PERMITTED", "CONNECTION_LOGGED"],
        vec!["ANTIVIRUS_SCAN", "NO_THREATS", "SCAN_COMPLETE"],
        // Maintenance logs
        vec![
            "PATCH_CHECK",
            "UPDATES_AVAILABLE",
            "PATCH_INSTALL",
            "REBOOT_REQUIRED",
        ],
        vec!["DISK_CHECK", "FILESYSTEM_OK", "SPACE_AVAILABLE"],
    ];

    for _ in 0..days {
        for _ in 0..logs_per_day {
            let pattern = &normal_patterns[logs.len() % normal_patterns.len()];
            logs.extend(pattern.iter().map(|s| s.to_string()));
        }
    }

    logs
}

fn generate_malware_infection() -> Vec<String> {
    vec![
        // Initial infection
        "EMAIL_RECEIVED",
        "ATTACHMENT_OPENED",
        "MACRO_ENABLED",
        "SUSPICIOUS_PROCESS",
        "UNKNOWN_EXECUTABLE",
        "PROCESS_INJECTION",
        // Persistence
        "REGISTRY_MODIFICATION",
        "STARTUP_ENTRY",
        "SCHEDULED_TASK",
        "SERVICE_CREATION",
        "AUTORUN_ENTRY",
        // Communication
        "OUTBOUND_CONNECTION",
        "SUSPICIOUS_DOMAIN",
        "C2_COMMUNICATION",
        "DATA_EXFILTRATION",
        "ENCRYPTED_TRAFFIC",
        // Lateral movement
        "NETWORK_SCAN",
        "CREDENTIAL_ACCESS",
        "REMOTE_EXECUTION",
        "PRIVILEGE_ESCALATION",
        "ADMIN_ACCESS",
        // Impact
        "FILE_ENCRYPTION",
        "RANSOM_NOTE",
        "SYSTEM_LOCKDOWN",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_privilege_escalation() -> Vec<String> {
    vec![
        // Normal user activity
        "USER_LOGIN",
        "STANDARD_ACCESS",
        "FILE_READ",
        // Escalation attempt
        "SUDO_ATTEMPT",
        "PERMISSION_DENIED",
        "RETRY_SUDO",
        "EXPLOIT_ATTEMPT",
        "BUFFER_OVERFLOW",
        "SHELLCODE_EXECUTION",
        // Success
        "ROOT_ACCESS",
        "ADMIN_PRIVILEGES",
        "SYSTEM_CONTROL",
        "SENSITIVE_FILE_ACCESS",
        "CONFIGURATION_CHANGE",
        // Cover tracks
        "LOG_MODIFICATION",
        "AUDIT_DISABLE",
        "HISTORY_CLEAR",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_data_exfiltration() -> Vec<String> {
    vec![
        // Data discovery
        "FILE_SEARCH",
        "DATABASE_QUERY",
        "SENSITIVE_DATA_ACCESS",
        "DOCUMENT_ENUMERATION",
        "EMAIL_ACCESS",
        // Data collection
        "FILE_COPY",
        "DATA_STAGING",
        "ARCHIVE_CREATION",
        "COMPRESSION",
        "ENCRYPTION",
        // Exfiltration
        "EXTERNAL_CONNECTION",
        "LARGE_UPLOAD",
        "DATA_TRANSFER",
        "CLOUD_STORAGE",
        "EMAIL_ATTACHMENT",
        "FTP_UPLOAD",
        // Cleanup
        "TEMP_FILE_DELETE",
        "CACHE_CLEAR",
        "EVIDENCE_REMOVAL",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_brute_force_attack() -> Vec<String> {
    vec![
        // Failed attempts
        "LOGIN_ATTEMPT",
        "AUTH_FAILURE",
        "INVALID_PASSWORD",
        "LOGIN_ATTEMPT",
        "AUTH_FAILURE",
        "INVALID_PASSWORD",
        "LOGIN_ATTEMPT",
        "AUTH_FAILURE",
        "INVALID_PASSWORD",
        "LOGIN_ATTEMPT",
        "AUTH_FAILURE",
        "INVALID_PASSWORD",
        "LOGIN_ATTEMPT",
        "AUTH_FAILURE",
        "INVALID_PASSWORD",
        // Account lockout
        "ACCOUNT_LOCKOUT",
        "SECURITY_ALERT",
        "ADMIN_NOTIFICATION",
        // Continued attempts
        "LOGIN_ATTEMPT",
        "AUTH_FAILURE",
        "ACCOUNT_LOCKED",
        "LOGIN_ATTEMPT",
        "AUTH_FAILURE",
        "ACCOUNT_LOCKED",
        // Success
        "LOGIN_ATTEMPT",
        "AUTH_SUCCESS",
        "SUSPICIOUS_LOGIN",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_insider_threat() -> Vec<String> {
    vec![
        // Normal activity
        "USER_LOGIN",
        "NORMAL_ACCESS",
        "ROUTINE_WORK",
        // Suspicious behavior
        "OFF_HOURS_ACCESS",
        "UNUSUAL_FILE_ACCESS",
        "BULK_DOWNLOAD",
        "SENSITIVE_DATA_QUERY",
        "UNAUTHORIZED_AREA",
        "POLICY_VIOLATION",
        // Data theft
        "PERSONAL_DEVICE",
        "USB_INSERT",
        "FILE_COPY",
        "EMAIL_FORWARD",
        "CLOUD_UPLOAD",
        "DATA_EXFILTRATION",
        // Cover up
        "ACCESS_LOG_VIEW",
        "AUDIT_TRAIL_CHECK",
        "SUSPICIOUS_DELETION",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_ddos_logs() -> Vec<String> {
    vec![
        // Normal traffic
        "CONNECTION_REQUEST",
        "RESPONSE_SENT",
        "CONNECTION_CLOSED",
        // Attack begins
        "HIGH_TRAFFIC",
        "CONNECTION_FLOOD",
        "RESOURCE_EXHAUSTION",
        "SERVER_OVERLOAD",
        "RESPONSE_TIMEOUT",
        "CONNECTION_DROPPED",
        // Mitigation
        "DDOS_DETECTED",
        "RATE_LIMITING",
        "TRAFFIC_FILTERING",
        "BLACKLIST_UPDATE",
        "MITIGATION_ACTIVE",
        // Recovery
        "TRAFFIC_NORMALIZED",
        "SERVICE_RESTORED",
        "MONITORING_ACTIVE",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_sql_injection_logs() -> Vec<String> {
    vec![
        // Normal requests
        "HTTP_REQUEST",
        "DATABASE_QUERY",
        "RESULT_RETURNED",
        "RESPONSE_SENT",
        // Attack attempts
        "SUSPICIOUS_INPUT",
        "SQL_INJECTION_ATTEMPT",
        "MALFORMED_QUERY",
        "ERROR_MESSAGE",
        "DATABASE_ERROR",
        "INJECTION_DETECTED",
        // Successful injection
        "UNAUTHORIZED_QUERY",
        "DATA_EXTRACTION",
        "PRIVILEGE_BYPASS",
        "SENSITIVE_DATA_ACCESS",
        "ADMIN_TABLE_ACCESS",
        // Detection and response
        "WAF_ALERT",
        "ATTACK_BLOCKED",
        "IP_BLACKLISTED",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_memory_leak() -> Vec<String> {
    vec![
        "PROCESS_START",
        "MEMORY_NORMAL",
        "OPERATION_NORMAL",
        "MEMORY_INCREASE",
        "ALLOCATION_HIGH",
        "MEMORY_PRESSURE",
        "MEMORY_CRITICAL",
        "SWAP_USAGE",
        "PERFORMANCE_DEGRADED",
        "OUT_OF_MEMORY",
        "PROCESS_KILLED",
        "SYSTEM_UNSTABLE",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_disk_space_issue() -> Vec<String> {
    vec![
        "DISK_NORMAL",
        "SPACE_AVAILABLE",
        "WRITE_SUCCESS",
        "DISK_USAGE_HIGH",
        "SPACE_WARNING",
        "CLEANUP_NEEDED",
        "DISK_FULL",
        "WRITE_FAILURE",
        "APPLICATION_ERROR",
        "CRITICAL_SPACE",
        "SYSTEM_UNSTABLE",
        "EMERGENCY_CLEANUP",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_network_congestion() -> Vec<String> {
    vec![
        "NETWORK_NORMAL",
        "BANDWIDTH_OK",
        "LATENCY_LOW",
        "TRAFFIC_INCREASE",
        "BANDWIDTH_HIGH",
        "LATENCY_RISING",
        "NETWORK_CONGESTION",
        "PACKET_LOSS",
        "TIMEOUT_ERRORS",
        "NETWORK_CRITICAL",
        "SERVICE_DEGRADED",
        "CONNECTION_FAILURES",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_service_failure() -> Vec<String> {
    vec![
        "SERVICE_RUNNING",
        "HEALTH_CHECK_OK",
        "RESPONSE_NORMAL",
        "SERVICE_SLOW",
        "RESPONSE_DELAYED",
        "ERROR_INCREASE",
        "SERVICE_UNSTABLE",
        "FREQUENT_ERRORS",
        "RESTART_ATTEMPT",
        "SERVICE_FAILURE",
        "CRASH_DETECTED",
        "RESTART_FAILED",
        "SERVICE_DOWN",
        "OUTAGE_DECLARED",
        "INCIDENT_CREATED",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_performance_degradation() -> Vec<String> {
    vec![
        "PERFORMANCE_NORMAL",
        "RESPONSE_FAST",
        "CPU_NORMAL",
        "PERFORMANCE_SLOW",
        "RESPONSE_DELAYED",
        "CPU_HIGH",
        "PERFORMANCE_POOR",
        "TIMEOUT_ERRORS",
        "CPU_CRITICAL",
        "PERFORMANCE_CRITICAL",
        "SYSTEM_UNRESPONSIVE",
        "INTERVENTION_NEEDED",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_compliance_scenario() -> Vec<String> {
    vec![
        // Normal compliance
        "ACCESS_LOGGED",
        "AUDIT_TRAIL",
        "POLICY_COMPLIANT",
        // Violations
        "UNAUTHORIZED_ACCESS",
        "POLICY_VIOLATION",
        "AUDIT_FAILURE",
        "DATA_BREACH",
        "PRIVACY_VIOLATION",
        "COMPLIANCE_ALERT",
        "REGULATORY_VIOLATION",
        "AUDIT_EXCEPTION",
        "INVESTIGATION_REQUIRED",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn generate_log_stream(size: usize) -> Vec<String> {
    let mut stream = Vec::new();

    for i in 0..size {
        if i % 100 == 0 {
            // 1% anomalous logs
            stream.extend(vec![
                "SUSPICIOUS_ACTIVITY".to_string(),
                "SECURITY_ALERT".to_string(),
                "INVESTIGATION_NEEDED".to_string(),
            ]);
        } else {
            // Normal logs
            stream.extend(vec![
                "NORMAL_OPERATION".to_string(),
                "STATUS_OK".to_string(),
                "ROUTINE_LOG".to_string(),
            ]);
        }
    }

    stream
}

fn calculate_threat_score(anomalies: &[AnomalyScore]) -> f64 {
    anomalies
        .iter()
        .map(|a| a.information_score * a.anomaly_strength)
        .sum::<f64>()
        / anomalies.len() as f64
}

fn classify_security_severity(threat_score: f64) -> String {
    if threat_score > 15.0 {
        "CRITICAL".to_string()
    } else if threat_score > 10.0 {
        "HIGH".to_string()
    } else if threat_score > 5.0 {
        "MEDIUM".to_string()
    } else {
        "LOW".to_string()
    }
}

fn calculate_confidence(anomalies: &[AnomalyScore]) -> f64 {
    let avg_strength =
        anomalies.iter().map(|a| a.anomaly_strength).sum::<f64>() / anomalies.len() as f64;

    avg_strength * 100.0
}

fn calculate_impact_score(anomalies: &[AnomalyScore]) -> f64 {
    anomalies.iter().map(|a| a.anomaly_strength).sum::<f64>()
}

fn classify_urgency(impact_score: f64) -> String {
    if impact_score > 3.0 {
        "URGENT".to_string()
    } else if impact_score > 2.0 {
        "HIGH".to_string()
    } else if impact_score > 1.0 {
        "MEDIUM".to_string()
    } else {
        "LOW".to_string()
    }
}

fn calculate_compliance_risk(anomalies: &[AnomalyScore]) -> f64 {
    anomalies.iter().map(|a| a.information_score).sum::<f64>() / anomalies.len() as f64
}

fn generate_security_alert(
    incident_type: &str,
    severity: &str,
    threat_score: f64,
    confidence: f64,
) {
    println!("  🚨 SECURITY ALERT GENERATED");
    println!("    Incident: {incident_type}");
    println!("    Severity: {severity}");
    println!("    Threat Score: {threat_score:.2}");
    println!("    Confidence: {confidence:.1}%");

    let response = match severity {
        "CRITICAL" => "IMMEDIATE INCIDENT RESPONSE REQUIRED",
        "HIGH" => "ESCALATE TO SECURITY TEAM",
        "MEDIUM" => "INVESTIGATE WITHIN 4 HOURS",
        _ => "LOG AND MONITOR",
    };

    println!("    Response: {response}");
}

fn generate_system_alert(issue_type: &str, urgency: &str, impact_score: f64) {
    println!("  ⚠️ SYSTEM ALERT GENERATED");
    println!("    Issue: {issue_type}");
    println!("    Urgency: {urgency}");
    println!("    Impact Score: {impact_score:.2}");

    let action = match urgency {
        "URGENT" => "IMMEDIATE SYSTEM INTERVENTION",
        "HIGH" => "ESCALATE TO OPERATIONS TEAM",
        "MEDIUM" => "SCHEDULE MAINTENANCE",
        _ => "MONITOR AND LOG",
    };

    println!("    Action: {action}");
}
