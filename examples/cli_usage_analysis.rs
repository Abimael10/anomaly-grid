//! Command Line Interface Usage Analysis
//!
//! This example demonstrates using anomaly-grid for analyzing CLI command sequences
//! to optimize user experience and detect automation patterns.
//!
//! ## Use Case:
//! - CLI commands are naturally categorical (finite alphabet)
//! - Usage patterns ARE the command sequences
//! - No missing fundamental features - command sequences contain all needed information
//! - Mathematical approach (Markov chains) is perfectly suited for command prediction
//! - Limitations are purely computational (scale/performance), not missing features
//!
//! ## Value Provided:
//! - User experience optimization insights
//! - CLI tool improvement guidance
//! - Automation detection and analysis
//! - Training and onboarding support
//!
//! ## Computational Limitations (Not Feature Gaps):
//! - Scale: Processing CLI logs from large user bases
//! - Performance: Real-time analysis of high-frequency command usage
//! - Memory: Large command vocabularies and user patterns
//! - Distribution: Analysis across multiple systems and environments

#![allow(clippy::uninlined_format_args)]

use anomaly_grid::*;
use std::time::Instant;
use std::collections::HashMap;

/// Common CLI commands across different systems
#[derive(Debug, Clone, PartialEq)]
pub enum CliCommand {
    // File operations
    Ls,
    Cd,
    Pwd,
    Mkdir,
    Rmdir,
    Rm,
    Cp,
    Mv,
    Find,
    Locate,
    
    // Text operations
    Cat,
    Less,
    More,
    Head,
    Tail,
    Grep,
    Sed,
    Awk,
    Sort,
    Uniq,
    
    // System operations
    Ps,
    Top,
    Kill,
    Jobs,
    Nohup,
    Crontab,
    Systemctl,
    Service,
    
    // Network operations
    Ping,
    Curl,
    Wget,
    Ssh,
    Scp,
    Rsync,
    Netstat,
    
    // Development tools
    Git,
    Make,
    Gcc,
    Python,
    Node,
    Npm,
    Docker,
    
    // Archive operations
    Tar,
    Zip,
    Unzip,
    Gzip,
    Gunzip,
    
    // System info
    Uname,
    Whoami,
    Id,
    Date,
    Uptime,
    Df,
    Du,
    Free,
    
    // Text editors
    Vim,
    Nano,
    Emacs,
}

impl CliCommand {
    fn to_string(&self) -> String {
        match self {
            CliCommand::Ls => "LS".to_string(),
            CliCommand::Cd => "CD".to_string(),
            CliCommand::Pwd => "PWD".to_string(),
            CliCommand::Mkdir => "MKDIR".to_string(),
            CliCommand::Rmdir => "RMDIR".to_string(),
            CliCommand::Rm => "RM".to_string(),
            CliCommand::Cp => "CP".to_string(),
            CliCommand::Mv => "MV".to_string(),
            CliCommand::Find => "FIND".to_string(),
            CliCommand::Locate => "LOCATE".to_string(),
            CliCommand::Cat => "CAT".to_string(),
            CliCommand::Less => "LESS".to_string(),
            CliCommand::More => "MORE".to_string(),
            CliCommand::Head => "HEAD".to_string(),
            CliCommand::Tail => "TAIL".to_string(),
            CliCommand::Grep => "GREP".to_string(),
            CliCommand::Sed => "SED".to_string(),
            CliCommand::Awk => "AWK".to_string(),
            CliCommand::Sort => "SORT".to_string(),
            CliCommand::Uniq => "UNIQ".to_string(),
            CliCommand::Ps => "PS".to_string(),
            CliCommand::Top => "TOP".to_string(),
            CliCommand::Kill => "KILL".to_string(),
            CliCommand::Jobs => "JOBS".to_string(),
            CliCommand::Nohup => "NOHUP".to_string(),
            CliCommand::Crontab => "CRONTAB".to_string(),
            CliCommand::Systemctl => "SYSTEMCTL".to_string(),
            CliCommand::Service => "SERVICE".to_string(),
            CliCommand::Ping => "PING".to_string(),
            CliCommand::Curl => "CURL".to_string(),
            CliCommand::Wget => "WGET".to_string(),
            CliCommand::Ssh => "SSH".to_string(),
            CliCommand::Scp => "SCP".to_string(),
            CliCommand::Rsync => "RSYNC".to_string(),
            CliCommand::Netstat => "NETSTAT".to_string(),
            CliCommand::Git => "GIT".to_string(),
            CliCommand::Make => "MAKE".to_string(),
            CliCommand::Gcc => "GCC".to_string(),
            CliCommand::Python => "PYTHON".to_string(),
            CliCommand::Node => "NODE".to_string(),
            CliCommand::Npm => "NPM".to_string(),
            CliCommand::Docker => "DOCKER".to_string(),
            CliCommand::Tar => "TAR".to_string(),
            CliCommand::Zip => "ZIP".to_string(),
            CliCommand::Unzip => "UNZIP".to_string(),
            CliCommand::Gzip => "GZIP".to_string(),
            CliCommand::Gunzip => "GUNZIP".to_string(),
            CliCommand::Uname => "UNAME".to_string(),
            CliCommand::Whoami => "WHOAMI".to_string(),
            CliCommand::Id => "ID".to_string(),
            CliCommand::Date => "DATE".to_string(),
            CliCommand::Uptime => "UPTIME".to_string(),
            CliCommand::Df => "DF".to_string(),
            CliCommand::Du => "DU".to_string(),
            CliCommand::Free => "FREE".to_string(),
            CliCommand::Vim => "VIM".to_string(),
            CliCommand::Nano => "NANO".to_string(),
            CliCommand::Emacs => "EMACS".to_string(),
        }
    }
}

/// Analysis result for a user's CLI usage pattern
#[derive(Debug, Clone)]
pub struct CliUsageAnalysis {
    pub user_id: String,
    pub command_sequence: Vec<String>,
    pub efficiency_score: f64,
    pub usage_patterns: Vec<String>,
    pub optimization_insights: Vec<String>,
    pub user_proficiency: UserProficiency,
    pub explanation: String,
    pub similar_users_found: usize,
}

/// User proficiency levels based on CLI usage patterns
#[derive(Debug, Clone)]
pub enum UserProficiency {
    Expert,
    Advanced,
    Intermediate,
    Beginner,
}

impl UserProficiency {
    fn from_score(score: f64) -> Self {
        if score >= 0.8 {
            UserProficiency::Beginner
        } else if score >= 0.6 {
            UserProficiency::Intermediate
        } else if score >= 0.3 {
            UserProficiency::Advanced
        } else {
            UserProficiency::Expert
        }
    }
    
    fn to_string(&self) -> &str {
        match self {
            UserProficiency::Expert => "EXPERT",
            UserProficiency::Advanced => "ADVANCED",
            UserProficiency::Intermediate => "INTERMEDIATE",
            UserProficiency::Beginner => "BEGINNER",
        }
    }
}

/// CLI usage pattern analyzer
pub struct CliUsageAnalyzer {
    detector: AnomalyDetector,
    user_patterns: HashMap<String, Vec<String>>,
    analysis_results: Vec<CliUsageAnalysis>,
    performance_metrics: HashMap<String, f64>,
}

impl CliUsageAnalyzer {
    /// Create new CLI usage analyzer
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let detector = AnomalyDetector::new(4)?; // 4th order for complex command patterns
        
        Ok(Self {
            detector,
            user_patterns: HashMap::new(),
            analysis_results: Vec::new(),
            performance_metrics: HashMap::new(),
        })
    }
    
    /// Train on efficient CLI usage patterns
    pub fn train_on_efficient_patterns(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔍 Training CLI analyzer on efficient usage patterns...");
        
        let efficient_patterns = self.generate_efficient_cli_patterns();
        let start_time = Instant::now();
        
        for pattern in &efficient_patterns {
            self.detector.train(pattern)?;
        }
        
        let training_time = start_time.elapsed();
        self.performance_metrics.insert("training_time_ms".to_string(), training_time.as_millis() as f64);
        
        println!("✅ Training completed in {:?}", training_time);
        println!("📊 Trained on {} efficient CLI patterns", efficient_patterns.len());
        
        Ok(())
    }
    
    /// Analyze a user's CLI usage pattern
    pub fn analyze_cli_usage(&mut self, commands: &[CliCommand], user_id: &str, threshold: f64) 
        -> Result<Option<CliUsageAnalysis>, Box<dyn std::error::Error>> {
        
        let command_sequence: Vec<String> = commands.iter()
            .map(|c| c.to_string())
            .collect();
        
        // Store user pattern
        self.user_patterns.insert(user_id.to_string(), command_sequence.clone());
        
        let detection_start = Instant::now();
        let anomalies = self.detector.detect_anomalies(&command_sequence, threshold)?;
        let detection_time = detection_start.elapsed();
        
        self.performance_metrics.insert("last_detection_time_ms".to_string(), detection_time.as_millis() as f64);
        
        if anomalies.is_empty() {
            // Even efficient patterns get analyzed for insights
            let analysis = self.create_efficient_usage_analysis(user_id, &command_sequence)?;
            return Ok(Some(analysis));
        }
        
        // Find the highest scoring anomaly
        let max_anomaly = anomalies.iter()
            .max_by(|a, b| a.anomaly_strength.partial_cmp(&b.anomaly_strength).unwrap())
            .unwrap();
        
        // Identify usage patterns
        let usage_patterns = self.identify_usage_patterns(&command_sequence, max_anomaly);
        
        // Generate optimization insights
        let optimization_insights = self.generate_optimization_insights(&command_sequence, max_anomaly);
        
        // Count similar users for context
        let similar_users = self.count_similar_users(&command_sequence);
        
        let proficiency = UserProficiency::from_score(max_anomaly.anomaly_strength);
        
        let explanation = format!(
            "CLI usage analysis: Anomaly strength: {:.3}, Likelihood: {:.6}, Information content: {:.3}. This command sequence deviates from efficient CLI patterns with {:.1}% confidence. Usage optimization opportunities identified.",
            max_anomaly.anomaly_strength,
            max_anomaly.likelihood,
            max_anomaly.information_score,
            (1.0 - max_anomaly.likelihood) * 100.0
        );
        
        let analysis = CliUsageAnalysis {
            user_id: user_id.to_string(),
            command_sequence,
            efficiency_score: max_anomaly.anomaly_strength,
            usage_patterns,
            optimization_insights,
            user_proficiency: proficiency,
            explanation,
            similar_users_found: similar_users,
        };
        
        self.analysis_results.push(analysis.clone());
        
        Ok(Some(analysis))
    }
    
    /// Create analysis for efficient usage patterns
    fn create_efficient_usage_analysis(&mut self, user_id: &str, command_sequence: &[String]) 
        -> Result<CliUsageAnalysis, Box<dyn std::error::Error>> {
        
        let usage_patterns = vec!["EFFICIENT_CLI_USAGE".to_string()];
        let optimization_insights = self.generate_optimization_insights(command_sequence, &AnomalyScore {
            sequence: command_sequence.to_vec(),
            likelihood: 0.9,
            log_likelihood: 0.9_f64.ln(),
            information_score: 1.0,
            anomaly_strength: 0.1,
        });
        
        let analysis = CliUsageAnalysis {
            user_id: user_id.to_string(),
            command_sequence: command_sequence.to_vec(),
            efficiency_score: 0.1,
            usage_patterns,
            optimization_insights,
            user_proficiency: UserProficiency::Expert,
            explanation: "Efficient CLI usage pattern demonstrating expert proficiency.".to_string(),
            similar_users_found: self.count_similar_users(command_sequence),
        };
        
        self.analysis_results.push(analysis.clone());
        Ok(analysis)
    }
    
    /// Identify specific usage patterns
    fn identify_usage_patterns(&self, command_sequence: &[String], anomaly: &AnomalyScore) -> Vec<String> {
        let mut patterns = Vec::new();
        
        // Check for automation patterns
        if self.detect_automation_pattern(command_sequence) {
            patterns.push("AUTOMATION_DETECTED".to_string());
        }
        
        // Check for inefficient navigation
        if self.detect_inefficient_navigation(command_sequence) {
            patterns.push("INEFFICIENT_NAVIGATION".to_string());
        }
        
        // Check for redundant operations
        if self.detect_redundant_operations(command_sequence) {
            patterns.push("REDUNDANT_OPERATIONS".to_string());
        }
        
        // Check for beginner patterns
        if self.detect_beginner_patterns(command_sequence) {
            patterns.push("BEGINNER_USAGE_PATTERN".to_string());
        }
        
        // Check for power user patterns
        if self.detect_power_user_patterns(command_sequence) {
            patterns.push("POWER_USER_PATTERN".to_string());
        }
        
        // High information content suggests very unusual usage
        if anomaly.information_score > 4.0 {
            patterns.push("HIGHLY_UNUSUAL_CLI_USAGE".to_string());
        }
        
        patterns
    }
    
    /// Generate optimization insights
    fn generate_optimization_insights(&self, command_sequence: &[String], anomaly: &AnomalyScore) -> Vec<String> {
        let mut insights = Vec::new();
        
        // Analyze navigation efficiency
        let cd_count = command_sequence.iter().filter(|c| c.as_str() == "CD").count();
        let pwd_count = command_sequence.iter().filter(|c| c.as_str() == "PWD").count();
        
        if pwd_count > cd_count {
            insights.push("Consider using shell prompt customization to show current directory".to_string());
        }
        
        // Analyze file operations
        let ls_count = command_sequence.iter().filter(|c| c.as_str() == "LS").count();
        if ls_count > command_sequence.len() / 3 {
            insights.push("Consider using file manager or IDE for frequent file browsing".to_string());
        }
        
        // Analyze text processing
        let cat_count = command_sequence.iter().filter(|c| c.as_str() == "CAT").count();
        let less_count = command_sequence.iter().filter(|c| c.as_str() == "LESS").count();
        if cat_count > less_count * 2 {
            insights.push("Consider using 'less' instead of 'cat' for viewing large files".to_string());
        }
        
        // Analyze development workflow
        let git_count = command_sequence.iter().filter(|c| c.as_str() == "GIT").count();
        let make_count = command_sequence.iter().filter(|c| c.as_str() == "MAKE").count();
        if git_count > 0 && make_count == 0 {
            insights.push("Consider using build automation tools for development workflow".to_string());
        }
        
        // Proficiency insights
        if anomaly.anomaly_strength < 0.2 {
            insights.push("Efficient CLI usage - demonstrating expert proficiency".to_string());
        } else if anomaly.anomaly_strength > 0.6 {
            insights.push("CLI usage has significant optimization opportunities".to_string());
        }
        
        insights
    }
    
    /// Detect automation patterns
    fn detect_automation_pattern(&self, commands: &[String]) -> bool {
        // Look for repetitive patterns that suggest scripting
        if commands.len() < 4 {
            return false;
        }
        
        // Check for exact repetition
        for i in 0..commands.len() - 3 {
            let pattern = &commands[i..i+2];
            for j in i+2..commands.len()-1 {
                if commands[j..j+2] == *pattern {
                    return true;
                }
            }
        }
        false
    }
    
    /// Detect inefficient navigation
    fn detect_inefficient_navigation(&self, commands: &[String]) -> bool {
        let cd_count = commands.iter().filter(|c| c.as_str() == "CD").count();
        let pwd_count = commands.iter().filter(|c| c.as_str() == "PWD").count();
        
        // Too many directory changes or pwd checks
        cd_count > 5 || pwd_count > 3
    }
    
    /// Detect redundant operations
    fn detect_redundant_operations(&self, commands: &[String]) -> bool {
        // Look for consecutive identical commands
        for i in 0..commands.len().saturating_sub(1) {
            if commands[i] == commands[i + 1] && commands[i] != "LS" {
                return true;
            }
        }
        false
    }
    
    /// Detect beginner usage patterns
    fn detect_beginner_patterns(&self, commands: &[String]) -> bool {
        let help_indicators = commands.iter()
            .filter(|c| c.as_str() == "WHOAMI" || c.as_str() == "PWD" || c.as_str() == "DATE")
            .count();
        
        help_indicators > 2
    }
    
    /// Detect power user patterns
    fn detect_power_user_patterns(&self, commands: &[String]) -> bool {
        let advanced_commands = commands.iter()
            .filter(|c| c.as_str() == "AWK" || c.as_str() == "SED" || c.as_str() == "GREP" || c.as_str() == "FIND")
            .count();
        
        advanced_commands > 2
    }
    
    /// Count users with similar CLI patterns
    fn count_similar_users(&self, target_sequence: &[String]) -> usize {
        self.user_patterns.values()
            .filter(|sequence| {
                let common_commands = sequence.iter()
                    .filter(|command| target_sequence.contains(command))
                    .count();
                
                common_commands as f64 / sequence.len() as f64 > 0.6
            })
            .count()
    }
    
    /// Generate analysis summary
    pub fn generate_analysis_summary(&self) -> CliAnalysisSummary {
        let total_users = self.analysis_results.len();
        let inefficient_users = self.analysis_results.iter()
            .filter(|r| matches!(r.user_proficiency, UserProficiency::Beginner | UserProficiency::Intermediate))
            .count();
        
        let avg_detection_time = self.performance_metrics
            .get("last_detection_time_ms")
            .unwrap_or(&0.0);
        
        let total_commands: usize = self.analysis_results.iter()
            .map(|r| r.command_sequence.len())
            .sum();
        
        CliAnalysisSummary {
            total_users_analyzed: total_users,
            inefficient_users_found: inefficient_users,
            average_analysis_time_ms: *avg_detection_time,
            patterns_identified: self.count_unique_patterns(),
            total_commands_analyzed: total_commands,
            optimization_opportunities: self.count_optimization_opportunities(),
        }
    }
    
    /// Count unique patterns identified
    fn count_unique_patterns(&self) -> usize {
        let mut all_patterns = std::collections::HashSet::new();
        for result in &self.analysis_results {
            for pattern in &result.usage_patterns {
                all_patterns.insert(pattern.clone());
            }
        }
        all_patterns.len()
    }
    
    /// Count optimization opportunities
    fn count_optimization_opportunities(&self) -> usize {
        self.analysis_results.iter()
            .map(|r| r.optimization_insights.len())
            .sum()
    }
    
    /// Generate efficient CLI patterns for training
    fn generate_efficient_cli_patterns(&self) -> Vec<Vec<String>> {
        let mut patterns = Vec::new();
        
        // Base efficient CLI usage patterns
        let base_patterns = vec![
            // File management
            vec![CliCommand::Ls, CliCommand::Cd, CliCommand::Ls, CliCommand::Cat],
            
            // Development workflow
            vec![CliCommand::Git, CliCommand::Vim, CliCommand::Make, CliCommand::Git],
            
            // System monitoring
            vec![CliCommand::Top, CliCommand::Ps, CliCommand::Kill],
            
            // Text processing
            vec![CliCommand::Cat, CliCommand::Grep, CliCommand::Sort, CliCommand::Uniq],
            
            // Network operations
            vec![CliCommand::Ping, CliCommand::Ssh, CliCommand::Scp],
            
            // Archive operations
            vec![CliCommand::Tar, CliCommand::Gzip, CliCommand::Mv],
            
            // Search operations
            vec![CliCommand::Find, CliCommand::Grep, CliCommand::Less],
            
            // Docker workflow
            vec![CliCommand::Docker, CliCommand::Docker, CliCommand::Docker],
        ];
        
        // Extended patterns for comprehensive training
        let extended_patterns = vec![
            // Complex development workflow
            vec![CliCommand::Pwd, CliCommand::Ls, CliCommand::Git, CliCommand::Vim, CliCommand::Git, CliCommand::Make, CliCommand::Git, CliCommand::Git],
            
            // System administration
            vec![CliCommand::Ps, CliCommand::Top, CliCommand::Kill, CliCommand::Systemctl, CliCommand::Service, CliCommand::Ps],
            
            // Data processing pipeline
            vec![CliCommand::Cat, CliCommand::Grep, CliCommand::Awk, CliCommand::Sort, CliCommand::Uniq, CliCommand::Head, CliCommand::Tail],
            
            // File organization
            vec![CliCommand::Ls, CliCommand::Mkdir, CliCommand::Cp, CliCommand::Mv, CliCommand::Rm, CliCommand::Ls],
            
            // Remote operations
            vec![CliCommand::Ssh, CliCommand::Ls, CliCommand::Cat, CliCommand::Vim, CliCommand::Scp, CliCommand::Ssh],
            
            // Archive management
            vec![CliCommand::Ls, CliCommand::Tar, CliCommand::Gzip, CliCommand::Mv, CliCommand::Ls, CliCommand::Gunzip, CliCommand::Tar],
            
            // Log analysis
            vec![CliCommand::Tail, CliCommand::Grep, CliCommand::Awk, CliCommand::Sort, CliCommand::Less],
            
            // Build and deployment
            vec![CliCommand::Git, CliCommand::Make, CliCommand::Docker, CliCommand::Docker, CliCommand::Git],
            
            // Network troubleshooting
            vec![CliCommand::Ping, CliCommand::Netstat, CliCommand::Curl, CliCommand::Wget, CliCommand::Ssh],
            
            // Performance monitoring
            vec![CliCommand::Top, CliCommand::Free, CliCommand::Df, CliCommand::Du, CliCommand::Ps],
            
            // Text editing workflow
            vec![CliCommand::Ls, CliCommand::Cat, CliCommand::Vim, CliCommand::Cat, CliCommand::Cp],
            
            // Package management
            vec![CliCommand::Npm, CliCommand::Node, CliCommand::Npm, CliCommand::Git],
        ];
        
        // Professional workflow patterns
        let professional_patterns = vec![
            // DevOps workflow
            vec![CliCommand::Git, CliCommand::Docker, CliCommand::Make, CliCommand::Ssh, CliCommand::Systemctl, CliCommand::Git],
            
            // Data scientist workflow
            vec![CliCommand::Python, CliCommand::Cat, CliCommand::Grep, CliCommand::Awk, CliCommand::Python, CliCommand::Vim],
            
            // System administrator
            vec![CliCommand::Systemctl, CliCommand::Service, CliCommand::Top, CliCommand::Ps, CliCommand::Kill, CliCommand::Crontab],
            
            // Web developer
            vec![CliCommand::Git, CliCommand::Npm, CliCommand::Node, CliCommand::Vim, CliCommand::Git, CliCommand::Curl],
            
            // Database administrator
            vec![CliCommand::Ps, CliCommand::Top, CliCommand::Systemctl, CliCommand::Tail, CliCommand::Grep, CliCommand::Less],
            
            // Security analyst
            vec![CliCommand::Netstat, CliCommand::Ps, CliCommand::Find, CliCommand::Grep, CliCommand::Tail, CliCommand::Less],
            
            // Content creator
            vec![CliCommand::Ls, CliCommand::Cat, CliCommand::Vim, CliCommand::Cp, CliCommand::Mv, CliCommand::Git],
            
            // Research workflow
            vec![CliCommand::Find, CliCommand::Grep, CliCommand::Cat, CliCommand::Less, CliCommand::Awk, CliCommand::Sort],
        ];
        
        // Efficiency enhancement patterns
        let efficiency_commands = vec![
            vec![CliCommand::Pwd],
            vec![CliCommand::Whoami],
            vec![CliCommand::Date],
            vec![CliCommand::Uptime],
            vec![CliCommand::Id],
            vec![CliCommand::Uname],
            vec![CliCommand::Pwd, CliCommand::Ls],
            vec![CliCommand::Date, CliCommand::Uptime],
        ];
        
        // Generate comprehensive training dataset with long sequences
        for iteration in 0..45 {
            // Create long realistic CLI session sequences by combining multiple patterns
            for base_idx in 0..base_patterns.len() {
                let mut long_sequence = Vec::new();
                
                // Start with efficiency commands for session startup
                if !efficiency_commands.is_empty() {
                    let startup_efficiency = &efficiency_commands[base_idx % efficiency_commands.len()];
                    long_sequence.extend(startup_efficiency.iter().cloned());
                }
                
                // Add base pattern
                long_sequence.extend(base_patterns[base_idx].iter().cloned());
                
                // Add 4-6 CLI operation segments to create substantial sequences
                let num_segments = 4 + (iteration % 3);
                for segment in 0..num_segments {
                    // Add efficiency commands between segments
                    if !efficiency_commands.is_empty() {
                        let efficiency = &efficiency_commands[segment % efficiency_commands.len()];
                        long_sequence.extend(efficiency.iter().cloned());
                    }
                    
                    // Add an extended pattern for complexity
                    let ext_idx = (base_idx + segment) % extended_patterns.len();
                    long_sequence.extend(extended_patterns[ext_idx].iter().cloned());
                    
                    // Add a professional pattern for realistic workflow
                    let prof_idx = (segment + 1) % professional_patterns.len();
                    long_sequence.extend(professional_patterns[prof_idx].iter().cloned());
                    
                    // Add another base pattern for continuity
                    let next_base_idx = (base_idx + segment + 1) % base_patterns.len();
                    long_sequence.extend(base_patterns[next_base_idx].iter().cloned());
                }
                
                // Add final efficiency commands for session cleanup
                if !efficiency_commands.is_empty() {
                    let final_efficiency = &efficiency_commands[iteration % efficiency_commands.len()];
                    long_sequence.extend(final_efficiency.iter().cloned());
                }
                
                // Convert to strings and add to patterns
                let string_pattern: Vec<String> = long_sequence.iter()
                    .map(|c| c.to_string())
                    .collect();
                patterns.push(string_pattern);
            }
            
            // Create mega-sequences by combining multiple professional patterns
            if iteration % 3 == 0 {
                let mut mega_sequence = Vec::new();
                
                // Combine 5-7 professional patterns into one large CLI session
                let num_patterns = 5 + (iteration % 3);
                for i in 0..num_patterns {
                    let pattern_idx = (iteration + i) % professional_patterns.len();
                    mega_sequence.extend(professional_patterns[pattern_idx].iter().cloned());
                    
                    // Add connecting efficiency commands and extended patterns
                    if i < num_patterns - 1 {
                        if !efficiency_commands.is_empty() {
                            let eff_idx = i % efficiency_commands.len();
                            mega_sequence.extend(efficiency_commands[eff_idx].iter().cloned());
                        }
                        let ext_idx = i % extended_patterns.len();
                        mega_sequence.extend(extended_patterns[ext_idx].iter().cloned());
                    }
                }
                
                let string_pattern: Vec<String> = mega_sequence.iter()
                    .map(|c| c.to_string())
                    .collect();
                patterns.push(string_pattern);
            }
        }
        
        // Add realistic full-day CLI usage patterns (long sequences)
        for _day in 0..30 {
            // Create full day CLI operations by combining morning, work, and evening sessions
            let mut full_day_usage = Vec::new();
            
            // Morning startup routine (12-15 commands)
            full_day_usage.extend(vec![
                CliCommand::Pwd, CliCommand::Ls, CliCommand::Git, CliCommand::Top, CliCommand::Ps,
                CliCommand::Cd, CliCommand::Ls, CliCommand::Vim, CliCommand::Git, CliCommand::Make,
                CliCommand::Ps, CliCommand::Top, CliCommand::Git, CliCommand::Git, CliCommand::Ls
            ]);
            
            // Active development session (18-22 commands)
            full_day_usage.extend(vec![
                CliCommand::Git, CliCommand::Vim, CliCommand::Make, CliCommand::Git, CliCommand::Docker,
                CliCommand::Curl, CliCommand::Git, CliCommand::Git, CliCommand::Vim, CliCommand::Make,
                CliCommand::Git, CliCommand::Git, CliCommand::Docker, CliCommand::Ps, CliCommand::Top,
                CliCommand::Git, CliCommand::Vim, CliCommand::Make, CliCommand::Git, CliCommand::Git,
                CliCommand::Curl, CliCommand::Wget
            ]);
            
            // Afternoon system administration (15-18 commands)
            full_day_usage.extend(vec![
                CliCommand::Top, CliCommand::Ps, CliCommand::Systemctl, CliCommand::Service, CliCommand::Df,
                CliCommand::Du, CliCommand::Free, CliCommand::Uptime, CliCommand::Ps, CliCommand::Kill,
                CliCommand::Systemctl, CliCommand::Service, CliCommand::Top, CliCommand::Netstat, CliCommand::Ps,
                CliCommand::Systemctl, CliCommand::Free, CliCommand::Uptime
            ]);
            
            // Research and data analysis (16-20 commands)
            full_day_usage.extend(vec![
                CliCommand::Find, CliCommand::Grep, CliCommand::Cat, CliCommand::Less, CliCommand::Awk,
                CliCommand::Sort, CliCommand::Uniq, CliCommand::Head, CliCommand::Tail, CliCommand::Grep,
                CliCommand::Cat, CliCommand::Less, CliCommand::Find, CliCommand::Awk, CliCommand::Sort,
                CliCommand::Uniq, CliCommand::Head, CliCommand::Tail, CliCommand::Cat, CliCommand::Less
            ]);
            
            // End of day cleanup and backup (12-15 commands)
            full_day_usage.extend(vec![
                CliCommand::Git, CliCommand::Ls, CliCommand::Rm, CliCommand::Mv, CliCommand::Tar,
                CliCommand::Gzip, CliCommand::Git, CliCommand::Git, CliCommand::Git, CliCommand::Git,
                CliCommand::Ls, CliCommand::Tar, CliCommand::Gzip, CliCommand::Mv, CliCommand::Ls
            ]);
            
            let string_pattern: Vec<String> = full_day_usage.iter()
                .map(|c| c.to_string())
                .collect();
            patterns.push(string_pattern);
            
            // Create alternative full-day patterns with different focus areas
            let mut alt_day_usage = Vec::new();
            
            // DevOps focused day (25-30 commands)
            alt_day_usage.extend(vec![
                CliCommand::Git, CliCommand::Docker, CliCommand::Make, CliCommand::Ssh, CliCommand::Systemctl,
                CliCommand::Git, CliCommand::Docker, CliCommand::Ps, CliCommand::Top, CliCommand::Systemctl,
                CliCommand::Service, CliCommand::Docker, CliCommand::Make, CliCommand::Git, CliCommand::Git,
                CliCommand::Ssh, CliCommand::Systemctl, CliCommand::Docker, CliCommand::Ps, CliCommand::Top,
                CliCommand::Git, CliCommand::Docker, CliCommand::Make, CliCommand::Systemctl, CliCommand::Service,
                CliCommand::Git, CliCommand::Git, CliCommand::Docker, CliCommand::Ssh, CliCommand::Systemctl
            ]);
            
            // Data analysis focused day (22-28 commands)
            alt_day_usage.extend(vec![
                CliCommand::Python, CliCommand::Cat, CliCommand::Grep, CliCommand::Awk, CliCommand::Python,
                CliCommand::Vim, CliCommand::Cat, CliCommand::Grep, CliCommand::Sort, CliCommand::Uniq,
                CliCommand::Python, CliCommand::Awk, CliCommand::Cat, CliCommand::Less, CliCommand::Head,
                CliCommand::Tail, CliCommand::Python, CliCommand::Grep, CliCommand::Sort, CliCommand::Awk,
                CliCommand::Cat, CliCommand::Python, CliCommand::Vim, CliCommand::Cat, CliCommand::Grep,
                CliCommand::Python, CliCommand::Awk, CliCommand::Sort
            ]);
            
            let alt_string_pattern: Vec<String> = alt_day_usage.iter()
                .map(|c| c.to_string())
                .collect();
            patterns.push(alt_string_pattern);
        }
        
        patterns
    }
}

/// CLI analysis summary data
#[derive(Debug)]
pub struct CliAnalysisSummary {
    pub total_users_analyzed: usize,
    pub inefficient_users_found: usize,
    pub average_analysis_time_ms: f64,
    pub patterns_identified: usize,
    pub total_commands_analyzed: usize,
    pub optimization_opportunities: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 CLI USAGE PATTERN ANALYSIS");
    println!("==============================");
    println!();
    
    // Initialize analyzer
    let mut analyzer = CliUsageAnalyzer::new()?;
    
    // Train on efficient patterns
    analyzer.train_on_efficient_patterns()?;
    println!();
    
    // Analyze different CLI usage scenarios
    println!("ANALYZING CLI USAGE PATTERNS");
    println!("===============================");
    
    // Scenario 1: Efficient development workflow
    println!("\n📊 Scenario 1: Efficient Development Workflow");
    let efficient_dev = vec![
        CliCommand::Git,
        CliCommand::Vim,
        CliCommand::Make,
        CliCommand::Git,
    ];
    
    let result = analyzer.analyze_cli_usage(&efficient_dev, "user_001", 0.3)?;
    if let Some(analysis) = result {
        println!("✅ User Proficiency: {}", analysis.user_proficiency.to_string());
        println!("   Patterns: {:?}", analysis.usage_patterns);
        println!("   Insights: {:?}", analysis.optimization_insights);
    }
    
    // Scenario 2: Inefficient navigation
    println!("\n🚨 Scenario 2: Inefficient Navigation Pattern");
    let inefficient_nav = vec![
        CliCommand::Pwd,
        CliCommand::Ls,
        CliCommand::Cd,
        CliCommand::Pwd,
        CliCommand::Ls,
        CliCommand::Cd,
        CliCommand::Pwd,
        CliCommand::Ls,
    ];
    
    let result = analyzer.analyze_cli_usage(&inefficient_nav, "user_002", 0.3)?;
    if let Some(analysis) = result {
        println!("⚠️  User Proficiency: {}", analysis.user_proficiency.to_string());
        println!("   Efficiency Score: {:.3}", analysis.efficiency_score);
        println!("   Patterns: {:?}", analysis.usage_patterns);
        println!("   Insights: {:?}", analysis.optimization_insights);
    }
    
    // Scenario 3: Automation detected
    println!("\n🤖 Scenario 3: Automation Pattern Detected");
    let automation = vec![
        CliCommand::Curl,
        CliCommand::Grep,
        CliCommand::Curl,
        CliCommand::Grep,
        CliCommand::Curl,
        CliCommand::Grep,
    ];
    
    let result = analyzer.analyze_cli_usage(&automation, "user_003", 0.3)?;
    if let Some(analysis) = result {
        println!("⚠️  User Proficiency: {}", analysis.user_proficiency.to_string());
        println!("   Patterns: {:?}", analysis.usage_patterns);
        println!("   Similar Users Found: {}", analysis.similar_users_found);
        println!("   Insights: {:?}", analysis.optimization_insights);
    }
    
    // Scenario 4: Power user pattern
    println!("\n⚡ Scenario 4: Power User Pattern");
    let power_user = vec![
        CliCommand::Find,
        CliCommand::Grep,
        CliCommand::Awk,
        CliCommand::Sed,
        CliCommand::Sort,
        CliCommand::Uniq,
    ];
    
    let result = analyzer.analyze_cli_usage(&power_user, "user_004", 0.3)?;
    if let Some(analysis) = result {
        println!("⚠️  User Proficiency: {}", analysis.user_proficiency.to_string());
        println!("   Patterns: {:?}", analysis.usage_patterns);
        println!("   Insights: {:?}", analysis.optimization_insights);
    }
    
    // Generate summary
    println!("\n📊 CLI ANALYSIS SUMMARY");
    println!("=======================");
    let summary = analyzer.generate_analysis_summary();
    println!("Users Analyzed: {}", summary.total_users_analyzed);
    println!("Inefficient Users Found: {}", summary.inefficient_users_found);
    println!("Average Analysis Time: {:.2}ms", summary.average_analysis_time_ms);
    println!("Patterns Identified: {}", summary.patterns_identified);
    println!("Total Commands Analyzed: {}", summary.total_commands_analyzed);
    println!("Optimization Opportunities: {}", summary.optimization_opportunities);
    
    Ok(())
}
