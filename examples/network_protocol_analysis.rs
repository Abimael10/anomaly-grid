//! Network Protocol State Analysis
//!
//! This example demonstrates using anomaly-grid for analyzing network protocol state sequences
//! to detect protocol violations and optimize network behavior.
//!
//! ## Use Case:
//! - Protocol states/messages are naturally categorical (finite alphabet)
//! - State transition patterns ARE the protocol behavior signal
//! - No missing fundamental features - protocol sequences contain all needed information
//! - Mathematical approach (Markov chains) is perfectly suited for protocol analysis
//! - Limitations are purely computational (scale/performance), not missing features
//!
//! ## Value Provided:
//! - Protocol compliance verification
//! - Network behavior optimization
//! - Anomaly detection in protocol flows
//! - Performance bottleneck identification
//!
//! ## Computational Limitations (Not Feature Gaps):
//! - Scale: Processing high-frequency network traffic
//! - Performance: Real-time analysis of protocol streams
//! - Memory: Large protocol vocabularies and flow patterns
//! - Distribution: Analysis across network infrastructure

#![allow(clippy::uninlined_format_args)]

use anomaly_grid::*;
use std::collections::HashMap;
use std::time::Instant;

/// Network protocol states and messages
#[derive(Debug, Clone, PartialEq)]
pub enum ProtocolState {
    // TCP States
    TcpSyn,
    TcpSynAck,
    TcpAck,
    TcpEstablished,
    TcpFin,
    TcpFinAck,
    TcpRst,
    TcpClosed,

    // HTTP States
    HttpRequest,
    HttpResponse,
    HttpRedirect,
    HttpError,
    HttpKeepAlive,
    HttpClose,

    // TLS/SSL States
    TlsClientHello,
    TlsServerHello,
    TlsCertificate,
    TlsKeyExchange,
    TlsFinished,
    TlsApplicationData,
    TlsAlert,

    // DNS States
    DnsQuery,
    DnsResponse,
    DnsError,
    DnsTimeout,

    // Application States
    AppConnect,
    AppAuthenticate,
    AppDataTransfer,
    AppHeartbeat,
    AppDisconnect,
    AppError,
}

impl std::fmt::Display for ProtocolState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ProtocolState::TcpSyn => "TCP_SYN",
            ProtocolState::TcpSynAck => "TCP_SYN_ACK",
            ProtocolState::TcpAck => "TCP_ACK",
            ProtocolState::TcpEstablished => "TCP_ESTABLISHED",
            ProtocolState::TcpFin => "TCP_FIN",
            ProtocolState::TcpFinAck => "TCP_FIN_ACK",
            ProtocolState::TcpRst => "TCP_RST",
            ProtocolState::TcpClosed => "TCP_CLOSED",
            ProtocolState::HttpRequest => "HTTP_REQUEST",
            ProtocolState::HttpResponse => "HTTP_RESPONSE",
            ProtocolState::HttpRedirect => "HTTP_REDIRECT",
            ProtocolState::HttpError => "HTTP_ERROR",
            ProtocolState::HttpKeepAlive => "HTTP_KEEP_ALIVE",
            ProtocolState::HttpClose => "HTTP_CLOSE",
            ProtocolState::TlsClientHello => "TLS_CLIENT_HELLO",
            ProtocolState::TlsServerHello => "TLS_SERVER_HELLO",
            ProtocolState::TlsCertificate => "TLS_CERTIFICATE",
            ProtocolState::TlsKeyExchange => "TLS_KEY_EXCHANGE",
            ProtocolState::TlsFinished => "TLS_FINISHED",
            ProtocolState::TlsApplicationData => "TLS_APPLICATION_DATA",
            ProtocolState::TlsAlert => "TLS_ALERT",
            ProtocolState::DnsQuery => "DNS_QUERY",
            ProtocolState::DnsResponse => "DNS_RESPONSE",
            ProtocolState::DnsError => "DNS_ERROR",
            ProtocolState::DnsTimeout => "DNS_TIMEOUT",
            ProtocolState::AppConnect => "APP_CONNECT",
            ProtocolState::AppAuthenticate => "APP_AUTHENTICATE",
            ProtocolState::AppDataTransfer => "APP_DATA_TRANSFER",
            ProtocolState::AppHeartbeat => "APP_HEARTBEAT",
            ProtocolState::AppDisconnect => "APP_DISCONNECT",
            ProtocolState::AppError => "APP_ERROR",
        };
        write!(f, "{}", s)
    }
}

/// Analysis result for a protocol flow
#[derive(Debug, Clone)]
pub struct ProtocolFlowAnalysis {
    pub flow_id: String,
    pub protocol_sequence: Vec<String>,
    pub compliance_score: f64,
    pub protocol_violations: Vec<String>,
    pub optimization_insights: Vec<String>,
    pub compliance_level: ComplianceLevel,
    pub explanation: String,
    pub similar_flows_found: usize,
}

/// Protocol compliance levels
#[derive(Debug, Clone)]
pub enum ComplianceLevel {
    Compliant,
    MinorViolations,
    MajorViolations,
    NonCompliant,
}

impl ComplianceLevel {
    fn from_score(score: f64) -> Self {
        if score >= 0.8 {
            ComplianceLevel::NonCompliant
        } else if score >= 0.6 {
            ComplianceLevel::MajorViolations
        } else if score >= 0.3 {
            ComplianceLevel::MinorViolations
        } else {
            ComplianceLevel::Compliant
        }
    }

    fn to_string(&self) -> &str {
        match self {
            ComplianceLevel::Compliant => "COMPLIANT",
            ComplianceLevel::MinorViolations => "MINOR_VIOLATIONS",
            ComplianceLevel::MajorViolations => "MAJOR_VIOLATIONS",
            ComplianceLevel::NonCompliant => "NON_COMPLIANT",
        }
    }
}

/// Network protocol analyzer
pub struct NetworkProtocolAnalyzer {
    detector: AnomalyDetector,
    protocol_flows: HashMap<String, Vec<String>>,
    analysis_results: Vec<ProtocolFlowAnalysis>,
    performance_metrics: HashMap<String, f64>,
}

impl NetworkProtocolAnalyzer {
    /// Create new network protocol analyzer
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let detector = AnomalyDetector::new(5)?; // 5th order for complex protocol patterns

        Ok(Self {
            detector,
            protocol_flows: HashMap::new(),
            analysis_results: Vec::new(),
            performance_metrics: HashMap::new(),
        })
    }

    /// Train on compliant protocol patterns
    pub fn train_on_compliant_protocols(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔍 Training protocol analyzer on compliant network patterns...");

        let compliant_patterns = self.generate_compliant_protocol_patterns();
        let start_time = Instant::now();

        for pattern in &compliant_patterns {
            self.detector.train(pattern)?;
        }

        let training_time = start_time.elapsed();
        self.performance_metrics.insert(
            "training_time_ms".to_string(),
            training_time.as_millis() as f64,
        );

        println!("✅ Training completed in {:?}", training_time);
        println!(
            "📊 Trained on {} compliant protocol patterns",
            compliant_patterns.len()
        );

        Ok(())
    }

    /// Analyze a protocol flow
    pub fn analyze_protocol_flow(
        &mut self,
        states: &[ProtocolState],
        flow_id: &str,
        threshold: f64,
    ) -> Result<Option<ProtocolFlowAnalysis>, Box<dyn std::error::Error>> {
        let protocol_sequence: Vec<String> = states.iter().map(|s| s.to_string()).collect();

        // Store protocol flow
        self.protocol_flows
            .insert(flow_id.to_string(), protocol_sequence.clone());

        let detection_start = Instant::now();
        let anomalies = self
            .detector
            .detect_anomalies(&protocol_sequence, threshold)?;
        let detection_time = detection_start.elapsed();

        self.performance_metrics.insert(
            "last_detection_time_ms".to_string(),
            detection_time.as_millis() as f64,
        );

        if anomalies.is_empty() {
            // Even compliant flows get analyzed for insights
            let analysis = self.create_compliant_flow_analysis(flow_id, &protocol_sequence)?;
            return Ok(Some(analysis));
        }

        // Find the highest scoring anomaly
        let max_anomaly = anomalies
            .iter()
            .max_by(|a, b| a.anomaly_strength.partial_cmp(&b.anomaly_strength).unwrap())
            .unwrap();

        // Identify protocol violations
        let protocol_violations =
            self.identify_protocol_violations(&protocol_sequence, max_anomaly);

        // Generate optimization insights
        let optimization_insights =
            self.generate_optimization_insights(&protocol_sequence, max_anomaly);

        // Count similar flows for context
        let similar_flows = self.count_similar_flows(&protocol_sequence);

        let compliance = ComplianceLevel::from_score(max_anomaly.anomaly_strength);

        let explanation = format!(
            "Protocol flow analysis: Anomaly strength: {:.3}, Likelihood: {:.6}, Information content: {:.3}. This protocol sequence deviates from standard patterns with {:.1}% confidence. Protocol compliance issues identified.",
            max_anomaly.anomaly_strength,
            max_anomaly.likelihood,
            max_anomaly.information_score,
            (1.0 - max_anomaly.likelihood) * 100.0
        );

        let analysis = ProtocolFlowAnalysis {
            flow_id: flow_id.to_string(),
            protocol_sequence,
            compliance_score: max_anomaly.anomaly_strength,
            protocol_violations,
            optimization_insights,
            compliance_level: compliance,
            explanation,
            similar_flows_found: similar_flows,
        };

        self.analysis_results.push(analysis.clone());

        Ok(Some(analysis))
    }

    /// Create analysis for compliant flows
    fn create_compliant_flow_analysis(
        &mut self,
        flow_id: &str,
        protocol_sequence: &[String],
    ) -> Result<ProtocolFlowAnalysis, Box<dyn std::error::Error>> {
        let protocol_violations = vec![];
        let optimization_insights = self.generate_optimization_insights(
            protocol_sequence,
            &AnomalyScore {
                sequence: protocol_sequence.to_vec(),
                likelihood: 0.9,
                log_likelihood: 0.9_f64.ln(),
                information_score: 1.0,
                anomaly_strength: 0.1,
            },
        );

        let analysis = ProtocolFlowAnalysis {
            flow_id: flow_id.to_string(),
            protocol_sequence: protocol_sequence.to_vec(),
            compliance_score: 0.1,
            protocol_violations,
            optimization_insights,
            compliance_level: ComplianceLevel::Compliant,
            explanation: "Protocol flow follows standard patterns. Fully compliant.".to_string(),
            similar_flows_found: self.count_similar_flows(protocol_sequence),
        };

        self.analysis_results.push(analysis.clone());
        Ok(analysis)
    }

    /// Identify specific protocol violations
    fn identify_protocol_violations(
        &self,
        protocol_sequence: &[String],
        anomaly: &AnomalyScore,
    ) -> Vec<String> {
        let mut violations = Vec::new();

        // Check for TCP handshake violations
        if self.detect_tcp_handshake_violation(protocol_sequence) {
            violations.push("TCP_HANDSHAKE_VIOLATION".to_string());
        }

        // Check for TLS handshake violations
        if self.detect_tls_handshake_violation(protocol_sequence) {
            violations.push("TLS_HANDSHAKE_VIOLATION".to_string());
        }

        // Check for improper connection termination
        if self.detect_improper_termination(protocol_sequence) {
            violations.push("IMPROPER_CONNECTION_TERMINATION".to_string());
        }

        // Check for protocol state violations
        if self.detect_state_violations(protocol_sequence) {
            violations.push("PROTOCOL_STATE_VIOLATION".to_string());
        }

        // Check for excessive error states
        if self.detect_excessive_errors(protocol_sequence) {
            violations.push("EXCESSIVE_ERROR_STATES".to_string());
        }

        // High information content suggests very unusual protocol behavior
        if anomaly.information_score > 5.0 {
            violations.push("HIGHLY_UNUSUAL_PROTOCOL_BEHAVIOR".to_string());
        }

        violations
    }

    /// Generate optimization insights
    fn generate_optimization_insights(
        &self,
        protocol_sequence: &[String],
        anomaly: &AnomalyScore,
    ) -> Vec<String> {
        let mut insights = Vec::new();

        // Analyze connection efficiency
        let connection_ops = protocol_sequence
            .iter()
            .filter(|s| s.contains("CONNECT") || s.contains("SYN"))
            .count();
        let data_ops = protocol_sequence
            .iter()
            .filter(|s| s.contains("DATA") || s.contains("TRANSFER"))
            .count();

        if connection_ops > data_ops {
            insights.push("High connection overhead - consider connection pooling".to_string());
        }

        // Analyze error patterns
        let error_count = protocol_sequence
            .iter()
            .filter(|s| s.contains("ERROR") || s.contains("TIMEOUT"))
            .count();
        if error_count > 2 {
            insights.push("High error rate detected - investigate network stability".to_string());
        }

        // Analyze TLS usage
        let tls_count = protocol_sequence
            .iter()
            .filter(|s| s.contains("TLS"))
            .count();
        let http_count = protocol_sequence
            .iter()
            .filter(|s| s.contains("HTTP"))
            .count();
        if http_count > 0 && tls_count == 0 {
            insights.push("Consider using HTTPS for secure communication".to_string());
        }

        // Analyze keep-alive usage
        let keep_alive_count = protocol_sequence
            .iter()
            .filter(|s| s.contains("KEEP_ALIVE"))
            .count();
        if http_count > 3 && keep_alive_count == 0 {
            insights.push("Consider using HTTP keep-alive for better performance".to_string());
        }

        // Compliance insights
        if anomaly.anomaly_strength < 0.2 {
            insights.push("Protocol flow is well-optimized and compliant".to_string());
        } else if anomaly.anomaly_strength > 0.6 {
            insights.push("Protocol flow has significant optimization opportunities".to_string());
        }

        insights
    }

    /// Detect TCP handshake violations
    fn detect_tcp_handshake_violation(&self, states: &[String]) -> bool {
        // Look for improper TCP handshake sequence
        for i in 0..states.len().saturating_sub(2) {
            if states[i] == "TCP_SYN" {
                // SYN should be followed by SYN_ACK, then ACK
                if i + 2 < states.len()
                    && (states[i + 1] != "TCP_SYN_ACK" || states[i + 2] != "TCP_ACK")
                {
                    return true;
                }
            }
        }
        false
    }

    /// Detect TLS handshake violations
    fn detect_tls_handshake_violation(&self, states: &[String]) -> bool {
        // Look for improper TLS handshake sequence
        let tls_states: Vec<&String> = states.iter().filter(|s| s.contains("TLS")).collect();
        if tls_states.len() >= 2 {
            // First TLS message should be CLIENT_HELLO
            if !tls_states[0].contains("CLIENT_HELLO") {
                return true;
            }
        }
        false
    }

    /// Detect improper connection termination
    fn detect_improper_termination(&self, states: &[String]) -> bool {
        // Look for RST without proper FIN sequence
        states.iter().any(|s| s.contains("RST")) && !states.iter().any(|s| s.contains("FIN"))
    }

    /// Detect protocol state violations
    fn detect_state_violations(&self, states: &[String]) -> bool {
        // Look for data transfer without established connection
        for i in 0..states.len() {
            if states[i].contains("DATA") || states[i].contains("TRANSFER") {
                // Check if connection was established before data transfer
                let established = states[0..i]
                    .iter()
                    .any(|s| s.contains("ESTABLISHED") || s.contains("FINISHED"));
                if !established {
                    return true;
                }
            }
        }
        false
    }

    /// Detect excessive error states
    fn detect_excessive_errors(&self, states: &[String]) -> bool {
        let error_count = states
            .iter()
            .filter(|s| s.contains("ERROR") || s.contains("TIMEOUT") || s.contains("ALERT"))
            .count();

        error_count > states.len() / 4
    }

    /// Count flows with similar protocol patterns
    fn count_similar_flows(&self, target_sequence: &[String]) -> usize {
        self.protocol_flows
            .values()
            .filter(|sequence| {
                let common_states = sequence
                    .iter()
                    .filter(|state| target_sequence.contains(state))
                    .count();

                common_states as f64 / sequence.len() as f64 > 0.6
            })
            .count()
    }

    /// Generate analysis summary
    pub fn generate_analysis_summary(&self) -> ProtocolAnalysisSummary {
        let total_flows = self.analysis_results.len();
        let non_compliant_flows = self
            .analysis_results
            .iter()
            .filter(|r| {
                matches!(
                    r.compliance_level,
                    ComplianceLevel::NonCompliant | ComplianceLevel::MajorViolations
                )
            })
            .count();

        let avg_detection_time = self
            .performance_metrics
            .get("last_detection_time_ms")
            .unwrap_or(&0.0);

        let total_states: usize = self
            .analysis_results
            .iter()
            .map(|r| r.protocol_sequence.len())
            .sum();

        ProtocolAnalysisSummary {
            total_flows_analyzed: total_flows,
            non_compliant_flows_found: non_compliant_flows,
            average_analysis_time_ms: *avg_detection_time,
            violations_identified: self.count_unique_violations(),
            total_states_analyzed: total_states,
            optimization_opportunities: self.count_optimization_opportunities(),
        }
    }

    /// Count unique violations identified
    fn count_unique_violations(&self) -> usize {
        let mut all_violations = std::collections::HashSet::new();
        for result in &self.analysis_results {
            for violation in &result.protocol_violations {
                all_violations.insert(violation.clone());
            }
        }
        all_violations.len()
    }

    /// Count optimization opportunities
    fn count_optimization_opportunities(&self) -> usize {
        self.analysis_results
            .iter()
            .map(|r| r.optimization_insights.len())
            .sum()
    }

    /// Generate compliant protocol patterns for training
    fn generate_compliant_protocol_patterns(&self) -> Vec<Vec<String>> {
        let mut patterns = Vec::new();

        // Base compliant network protocol patterns
        let base_patterns = [
            // Standard TCP connection
            vec![
                ProtocolState::TcpSyn,
                ProtocolState::TcpSynAck,
                ProtocolState::TcpAck,
                ProtocolState::TcpEstablished,
                ProtocolState::TcpFin,
                ProtocolState::TcpFinAck,
                ProtocolState::TcpClosed,
            ],
            // HTTP over TCP
            vec![
                ProtocolState::TcpEstablished,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
                ProtocolState::HttpClose,
            ],
            // HTTPS with TLS
            vec![
                ProtocolState::TcpEstablished,
                ProtocolState::TlsClientHello,
                ProtocolState::TlsServerHello,
                ProtocolState::TlsCertificate,
                ProtocolState::TlsFinished,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
            ],
            // DNS query
            vec![ProtocolState::DnsQuery, ProtocolState::DnsResponse],
            // Application flow
            vec![
                ProtocolState::AppConnect,
                ProtocolState::AppAuthenticate,
                ProtocolState::AppDataTransfer,
                ProtocolState::AppHeartbeat,
                ProtocolState::AppDisconnect,
            ],
            // Keep-alive HTTP
            vec![
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
                ProtocolState::HttpKeepAlive,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
            ],
            // Error recovery
            vec![
                ProtocolState::TcpEstablished,
                ProtocolState::AppError,
                ProtocolState::AppConnect,
                ProtocolState::AppDataTransfer,
            ],
        ];

        // Extended protocol patterns for comprehensive training
        let extended_patterns = vec![
            // Complex HTTPS session
            vec![
                ProtocolState::TcpSyn,
                ProtocolState::TcpSynAck,
                ProtocolState::TcpAck,
                ProtocolState::TcpEstablished,
                ProtocolState::TlsClientHello,
                ProtocolState::TlsServerHello,
                ProtocolState::TlsCertificate,
                ProtocolState::TlsKeyExchange,
                ProtocolState::TlsFinished,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
                ProtocolState::HttpKeepAlive,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
                ProtocolState::TcpFin,
                ProtocolState::TcpFinAck,
                ProtocolState::TcpClosed,
            ],
            // DNS with fallback
            vec![
                ProtocolState::DnsQuery,
                ProtocolState::DnsTimeout,
                ProtocolState::DnsQuery,
                ProtocolState::DnsResponse,
            ],
            // Application with heartbeat
            vec![
                ProtocolState::AppConnect,
                ProtocolState::AppAuthenticate,
                ProtocolState::AppDataTransfer,
                ProtocolState::AppHeartbeat,
                ProtocolState::AppDataTransfer,
                ProtocolState::AppHeartbeat,
                ProtocolState::AppDisconnect,
            ],
            // HTTP redirect flow
            vec![
                ProtocolState::TcpEstablished,
                ProtocolState::HttpRequest,
                ProtocolState::HttpRedirect,
                ProtocolState::TcpClosed,
                ProtocolState::TcpSyn,
                ProtocolState::TcpSynAck,
                ProtocolState::TcpAck,
                ProtocolState::TcpEstablished,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
            ],
            // TLS renegotiation
            vec![
                ProtocolState::TlsApplicationData,
                ProtocolState::TlsClientHello,
                ProtocolState::TlsServerHello,
                ProtocolState::TlsFinished,
                ProtocolState::TlsApplicationData,
            ],
            // Multi-request HTTP session
            vec![
                ProtocolState::TcpEstablished,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
                ProtocolState::HttpKeepAlive,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
                ProtocolState::HttpKeepAlive,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
                ProtocolState::HttpClose,
            ],
            // Application error recovery
            vec![
                ProtocolState::AppConnect,
                ProtocolState::AppAuthenticate,
                ProtocolState::AppDataTransfer,
                ProtocolState::AppError,
                ProtocolState::AppConnect,
                ProtocolState::AppAuthenticate,
                ProtocolState::AppDataTransfer,
                ProtocolState::AppDisconnect,
            ],
            // TCP with retransmission
            vec![
                ProtocolState::TcpSyn,
                ProtocolState::TcpSynAck,
                ProtocolState::TcpAck,
                ProtocolState::TcpEstablished,
                ProtocolState::TcpAck,
                ProtocolState::TcpAck,
                ProtocolState::TcpFin,
                ProtocolState::TcpFinAck,
                ProtocolState::TcpClosed,
            ],
            // DNS over HTTPS
            vec![
                ProtocolState::TcpEstablished,
                ProtocolState::TlsApplicationData,
                ProtocolState::DnsQuery,
                ProtocolState::DnsResponse,
                ProtocolState::TlsApplicationData,
            ],
            // WebSocket upgrade
            vec![
                ProtocolState::TcpEstablished,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
                ProtocolState::AppConnect,
                ProtocolState::AppDataTransfer,
                ProtocolState::AppHeartbeat,
                ProtocolState::AppDataTransfer,
                ProtocolState::AppDisconnect,
            ],
            // Load balancer health check
            vec![
                ProtocolState::TcpSyn,
                ProtocolState::TcpSynAck,
                ProtocolState::TcpAck,
                ProtocolState::TcpEstablished,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
                ProtocolState::TcpFin,
                ProtocolState::TcpFinAck,
                ProtocolState::TcpClosed,
            ],
            // API authentication flow
            vec![
                ProtocolState::TcpEstablished,
                ProtocolState::TlsApplicationData,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
                ProtocolState::TlsApplicationData,
            ],
        ];

        // Real-world scenario patterns
        let scenario_patterns = [
            // Web browsing session
            vec![
                ProtocolState::DnsQuery,
                ProtocolState::DnsResponse,
                ProtocolState::TcpSyn,
                ProtocolState::TcpSynAck,
                ProtocolState::TcpAck,
                ProtocolState::TcpEstablished,
                ProtocolState::TlsClientHello,
                ProtocolState::TlsServerHello,
                ProtocolState::TlsCertificate,
                ProtocolState::TlsFinished,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
            ],
            // File download
            vec![
                ProtocolState::TcpEstablished,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
                ProtocolState::TlsApplicationData,
                ProtocolState::TlsApplicationData,
                ProtocolState::TlsApplicationData,
                ProtocolState::HttpClose,
            ],
            // Video streaming
            vec![
                ProtocolState::TcpEstablished,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
                ProtocolState::HttpKeepAlive,
                ProtocolState::TlsApplicationData,
                ProtocolState::AppHeartbeat,
                ProtocolState::TlsApplicationData,
                ProtocolState::AppHeartbeat,
            ],
            // Email client sync
            vec![
                ProtocolState::TcpEstablished,
                ProtocolState::TlsClientHello,
                ProtocolState::TlsServerHello,
                ProtocolState::TlsFinished,
                ProtocolState::AppAuthenticate,
                ProtocolState::AppDataTransfer,
                ProtocolState::AppHeartbeat,
                ProtocolState::AppDisconnect,
            ],
            // Database connection
            vec![
                ProtocolState::TcpSyn,
                ProtocolState::TcpSynAck,
                ProtocolState::TcpAck,
                ProtocolState::TcpEstablished,
                ProtocolState::AppConnect,
                ProtocolState::AppAuthenticate,
                ProtocolState::AppDataTransfer,
                ProtocolState::AppDataTransfer,
                ProtocolState::AppDisconnect,
                ProtocolState::TcpClosed,
            ],
            // Microservice communication
            vec![
                ProtocolState::TcpEstablished,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
                ProtocolState::HttpKeepAlive,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
                ProtocolState::HttpClose,
            ],
            // CDN cache miss
            vec![
                ProtocolState::HttpRequest,
                ProtocolState::HttpRedirect,
                ProtocolState::TcpSyn,
                ProtocolState::TcpSynAck,
                ProtocolState::TcpEstablished,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
            ],
            // Mobile app sync
            vec![
                ProtocolState::AppConnect,
                ProtocolState::AppAuthenticate,
                ProtocolState::AppDataTransfer,
                ProtocolState::AppHeartbeat,
                ProtocolState::AppDataTransfer,
                ProtocolState::AppHeartbeat,
                ProtocolState::AppDisconnect,
            ],
        ];

        // Protocol optimization patterns
        let optimization_states = [
            vec![ProtocolState::AppHeartbeat],
            vec![ProtocolState::HttpKeepAlive],
            vec![ProtocolState::TlsApplicationData],
            vec![ProtocolState::TcpAck],
            vec![
                ProtocolState::AppHeartbeat,
                ProtocolState::TlsApplicationData,
            ],
            vec![ProtocolState::HttpKeepAlive, ProtocolState::TcpAck],
        ];

        // Generate comprehensive training dataset with long sequences
        for iteration in 0..50 {
            // Create long realistic network session sequences by combining multiple patterns
            for base_idx in 0..base_patterns.len() {
                let mut long_sequence = Vec::new();

                // Start with a base protocol pattern
                long_sequence.extend(base_patterns[base_idx].iter().cloned());

                // Add 4-6 network operation segments to create substantial sequences
                let num_segments = 4 + (iteration % 3);
                for segment in 0..num_segments {
                    // Add optimization states between segments
                    if !optimization_states.is_empty() {
                        let optimization =
                            &optimization_states[segment % optimization_states.len()];
                        long_sequence.extend(optimization.iter().cloned());
                    }

                    // Add an extended pattern for complexity
                    let ext_idx = (base_idx + segment) % extended_patterns.len();
                    long_sequence.extend(extended_patterns[ext_idx].iter().cloned());

                    // Add a scenario pattern for realistic flow
                    let scenario_idx = (segment + 1) % scenario_patterns.len();
                    long_sequence.extend(scenario_patterns[scenario_idx].iter().cloned());

                    // Add another base pattern for continuity
                    let next_base_idx = (base_idx + segment + 1) % base_patterns.len();
                    long_sequence.extend(base_patterns[next_base_idx].iter().cloned());
                }

                // Add final optimization states
                if !optimization_states.is_empty() {
                    let final_optimization =
                        &optimization_states[iteration % optimization_states.len()];
                    long_sequence.extend(final_optimization.iter().cloned());
                }

                // Convert to strings and add to patterns
                let string_pattern: Vec<String> =
                    long_sequence.iter().map(|s| s.to_string()).collect();
                patterns.push(string_pattern);
            }

            // Create mega-sequences by combining multiple extended patterns
            if iteration % 3 == 0 {
                let mut mega_sequence = Vec::new();

                // Combine 5-7 extended patterns into one large network session
                let num_patterns = 5 + (iteration % 3);
                for i in 0..num_patterns {
                    let pattern_idx = (iteration + i) % extended_patterns.len();
                    mega_sequence.extend(extended_patterns[pattern_idx].iter().cloned());

                    // Add connecting optimizations and scenario patterns
                    if i < num_patterns - 1 {
                        if !optimization_states.is_empty() {
                            let opt_idx = i % optimization_states.len();
                            mega_sequence.extend(optimization_states[opt_idx].iter().cloned());
                        }
                        let scenario_idx = i % scenario_patterns.len();
                        mega_sequence.extend(scenario_patterns[scenario_idx].iter().cloned());
                    }
                }

                let string_pattern: Vec<String> =
                    mega_sequence.iter().map(|s| s.to_string()).collect();
                patterns.push(string_pattern);
            }
        }

        // Add realistic full-day network traffic patterns (long sequences)
        for _session in 0..35 {
            // Create full day network operations by combining morning, peak, and evening traffic
            let mut full_day_traffic = Vec::new();

            // Morning startup traffic (15-18 states)
            full_day_traffic.extend(vec![
                ProtocolState::DnsQuery,
                ProtocolState::DnsResponse,
                ProtocolState::TcpSyn,
                ProtocolState::TcpSynAck,
                ProtocolState::TcpAck,
                ProtocolState::TcpEstablished,
                ProtocolState::TlsClientHello,
                ProtocolState::TlsServerHello,
                ProtocolState::TlsCertificate,
                ProtocolState::TlsFinished,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
                ProtocolState::HttpKeepAlive,
                ProtocolState::AppConnect,
                ProtocolState::AppAuthenticate,
                ProtocolState::AppDataTransfer,
                ProtocolState::AppHeartbeat,
                ProtocolState::TlsApplicationData,
            ]);

            // Peak usage traffic (20-25 states)
            full_day_traffic.extend(vec![
                ProtocolState::TcpEstablished,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
                ProtocolState::HttpKeepAlive,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
                ProtocolState::TlsApplicationData,
                ProtocolState::AppHeartbeat,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
                ProtocolState::TlsApplicationData,
                ProtocolState::AppDataTransfer,
                ProtocolState::HttpKeepAlive,
                ProtocolState::TlsApplicationData,
                ProtocolState::AppHeartbeat,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
                ProtocolState::TlsApplicationData,
                ProtocolState::AppDataTransfer,
                ProtocolState::HttpKeepAlive,
                ProtocolState::TlsApplicationData,
                ProtocolState::AppHeartbeat,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
                ProtocolState::TlsApplicationData,
            ]);

            // Afternoon application traffic (18-22 states)
            full_day_traffic.extend(vec![
                ProtocolState::AppConnect,
                ProtocolState::AppAuthenticate,
                ProtocolState::AppDataTransfer,
                ProtocolState::AppHeartbeat,
                ProtocolState::AppDataTransfer,
                ProtocolState::TlsApplicationData,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
                ProtocolState::HttpKeepAlive,
                ProtocolState::AppDataTransfer,
                ProtocolState::AppHeartbeat,
                ProtocolState::TlsApplicationData,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
                ProtocolState::AppDataTransfer,
                ProtocolState::AppHeartbeat,
                ProtocolState::TlsApplicationData,
                ProtocolState::HttpKeepAlive,
                ProtocolState::AppDataTransfer,
                ProtocolState::AppHeartbeat,
                ProtocolState::TlsApplicationData,
                ProtocolState::AppDisconnect,
            ]);

            // Evening cleanup and background sync (15-18 states)
            full_day_traffic.extend(vec![
                ProtocolState::AppConnect,
                ProtocolState::AppAuthenticate,
                ProtocolState::AppDataTransfer,
                ProtocolState::AppHeartbeat,
                ProtocolState::AppDataTransfer,
                ProtocolState::AppHeartbeat,
                ProtocolState::AppDataTransfer,
                ProtocolState::AppDisconnect,
                ProtocolState::TcpFin,
                ProtocolState::TcpFinAck,
                ProtocolState::TcpClosed,
                ProtocolState::HttpClose,
                ProtocolState::TlsApplicationData,
                ProtocolState::TcpFin,
                ProtocolState::TcpFinAck,
                ProtocolState::TcpClosed,
                ProtocolState::AppDisconnect,
                ProtocolState::HttpClose,
            ]);

            let string_pattern: Vec<String> =
                full_day_traffic.iter().map(|s| s.to_string()).collect();
            patterns.push(string_pattern);

            // Create alternative full-day patterns with different traffic types
            let mut alt_day_traffic = Vec::new();

            // Heavy download session (25-30 states)
            alt_day_traffic.extend(vec![
                ProtocolState::DnsQuery,
                ProtocolState::DnsResponse,
                ProtocolState::TcpSyn,
                ProtocolState::TcpSynAck,
                ProtocolState::TcpAck,
                ProtocolState::TcpEstablished,
                ProtocolState::TlsClientHello,
                ProtocolState::TlsServerHello,
                ProtocolState::TlsCertificate,
                ProtocolState::TlsKeyExchange,
                ProtocolState::TlsFinished,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
                ProtocolState::TlsApplicationData,
                ProtocolState::TlsApplicationData,
                ProtocolState::TlsApplicationData,
                ProtocolState::TlsApplicationData,
                ProtocolState::TlsApplicationData,
                ProtocolState::TlsApplicationData,
                ProtocolState::TlsApplicationData,
                ProtocolState::TlsApplicationData,
                ProtocolState::TlsApplicationData,
                ProtocolState::TlsApplicationData,
                ProtocolState::HttpClose,
                ProtocolState::TcpFin,
                ProtocolState::TcpFinAck,
                ProtocolState::TcpClosed,
                ProtocolState::AppDisconnect,
                ProtocolState::HttpClose,
                ProtocolState::TcpClosed,
            ]);

            // Video streaming session (22-28 states)
            alt_day_traffic.extend(vec![
                ProtocolState::TcpEstablished,
                ProtocolState::HttpRequest,
                ProtocolState::HttpResponse,
                ProtocolState::HttpKeepAlive,
                ProtocolState::TlsApplicationData,
                ProtocolState::AppHeartbeat,
                ProtocolState::TlsApplicationData,
                ProtocolState::AppHeartbeat,
                ProtocolState::TlsApplicationData,
                ProtocolState::AppHeartbeat,
                ProtocolState::TlsApplicationData,
                ProtocolState::AppHeartbeat,
                ProtocolState::TlsApplicationData,
                ProtocolState::AppHeartbeat,
                ProtocolState::TlsApplicationData,
                ProtocolState::AppHeartbeat,
                ProtocolState::TlsApplicationData,
                ProtocolState::AppHeartbeat,
                ProtocolState::TlsApplicationData,
                ProtocolState::HttpKeepAlive,
                ProtocolState::TlsApplicationData,
                ProtocolState::AppHeartbeat,
                ProtocolState::HttpClose,
                ProtocolState::TcpFin,
                ProtocolState::TcpFinAck,
                ProtocolState::TcpClosed,
                ProtocolState::AppDisconnect,
                ProtocolState::HttpClose,
            ]);

            let alt_string_pattern: Vec<String> =
                alt_day_traffic.iter().map(|s| s.to_string()).collect();
            patterns.push(alt_string_pattern);
        }

        patterns
    }
}

/// Protocol analysis summary data
#[derive(Debug)]
pub struct ProtocolAnalysisSummary {
    pub total_flows_analyzed: usize,
    pub non_compliant_flows_found: usize,
    pub average_analysis_time_ms: f64,
    pub violations_identified: usize,
    pub total_states_analyzed: usize,
    pub optimization_opportunities: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 NETWORK PROTOCOL STATE ANALYSIS");
    println!("===================================");
    println!();

    // Initialize analyzer
    let mut analyzer = NetworkProtocolAnalyzer::new()?;

    // Train on compliant patterns
    analyzer.train_on_compliant_protocols()?;
    println!();

    // Analyze different protocol scenarios
    println!("🔍 ANALYZING NETWORK PROTOCOL FLOWS");
    println!("===================================");

    // Scenario 1: Compliant TCP connection
    println!("\n📊 Scenario 1: Compliant TCP Connection");
    let compliant_tcp = vec![
        ProtocolState::TcpSyn,
        ProtocolState::TcpSynAck,
        ProtocolState::TcpAck,
        ProtocolState::TcpEstablished,
        ProtocolState::TcpFin,
        ProtocolState::TcpFinAck,
        ProtocolState::TcpClosed,
    ];

    let result = analyzer.analyze_protocol_flow(&compliant_tcp, "flow_001", 0.3)?;
    if let Some(analysis) = result {
        println!(
            "✅ Compliance Level: {}",
            analysis.compliance_level.to_string()
        );
        println!("   Violations: {:?}", analysis.protocol_violations);
        println!("   Insights: {:?}", analysis.optimization_insights);
    }

    // Scenario 2: TCP handshake violation
    println!("\n🚨 Scenario 2: TCP Handshake Violation");
    let handshake_violation = vec![
        ProtocolState::TcpSyn,
        ProtocolState::TcpAck, // Missing SYN_ACK
        ProtocolState::TcpEstablished,
        ProtocolState::TcpRst,
    ];

    let result = analyzer.analyze_protocol_flow(&handshake_violation, "flow_002", 0.3)?;
    if let Some(analysis) = result {
        println!(
            "⚠️  Compliance Level: {}",
            analysis.compliance_level.to_string()
        );
        println!("   Compliance Score: {:.3}", analysis.compliance_score);
        println!("   Violations: {:?}", analysis.protocol_violations);
        println!("   Insights: {:?}", analysis.optimization_insights);
    }

    // Scenario 3: TLS handshake issue
    println!("\n🔒 Scenario 3: TLS Handshake Issue");
    let tls_issue = vec![
        ProtocolState::TcpEstablished,
        ProtocolState::TlsServerHello, // Missing CLIENT_HELLO
        ProtocolState::TlsCertificate,
        ProtocolState::TlsAlert,
    ];

    let result = analyzer.analyze_protocol_flow(&tls_issue, "flow_003", 0.3)?;
    if let Some(analysis) = result {
        println!(
            "⚠️  Compliance Level: {}",
            analysis.compliance_level.to_string()
        );
        println!("   Violations: {:?}", analysis.protocol_violations);
        println!("   Similar Flows Found: {}", analysis.similar_flows_found);
        println!("   Insights: {:?}", analysis.optimization_insights);
    }

    // Scenario 4: Excessive errors
    println!("\n❌ Scenario 4: Excessive Error States");
    let excessive_errors = vec![
        ProtocolState::AppConnect,
        ProtocolState::AppError,
        ProtocolState::AppConnect,
        ProtocolState::AppError,
        ProtocolState::DnsTimeout,
        ProtocolState::AppError,
    ];

    let result = analyzer.analyze_protocol_flow(&excessive_errors, "flow_004", 0.3)?;
    if let Some(analysis) = result {
        println!(
            "⚠️  Compliance Level: {}",
            analysis.compliance_level.to_string()
        );
        println!("   Violations: {:?}", analysis.protocol_violations);
        println!("   Insights: {:?}", analysis.optimization_insights);
    }

    // Generate summary
    println!("\n📊 PROTOCOL ANALYSIS SUMMARY");
    println!("============================");
    let summary = analyzer.generate_analysis_summary();
    println!("Flows Analyzed: {}", summary.total_flows_analyzed);
    println!(
        "Non-Compliant Flows Found: {}",
        summary.non_compliant_flows_found
    );
    println!(
        "Average Analysis Time: {:.2}ms",
        summary.average_analysis_time_ms
    );
    println!("Violations Identified: {}", summary.violations_identified);
    println!("Total States Analyzed: {}", summary.total_states_analyzed);
    println!(
        "Optimization Opportunities: {}",
        summary.optimization_opportunities
    );

    Ok(())
}
