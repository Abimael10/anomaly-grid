//! Git Workflow Pattern Analysis
//!
//! This example demonstrates using anomaly-grid for analyzing Git command sequences
//! to understand developer workflows and identify unusual patterns.
//!
//! ## Use Case:
//! - Git commands are naturally categorical (finite alphabet)
//! - Workflow patterns ARE the command sequences
//! - No missing fundamental features - sequences contain all needed information
//! - Mathematical approach (Markov chains) is perfectly suited for workflow analysis
//! - Limitations are purely computational (scale/performance), not missing features
//!
//! ## Value Provided:
//! - Developer productivity insights
//! - Workflow optimization opportunities  
//! - Security pattern detection
//! - Training and onboarding support
//!
//! ## Computational Limitations (Not Feature Gaps):
//! - Scale: Processing git logs from large organizations
//! - Performance: Real-time analysis of high-frequency git usage
//! - Memory: Large command vocabularies and user bases
//! - Distribution: Analysis across multiple repositories/teams

#![allow(clippy::uninlined_format_args)]

use anomaly_grid::*;
use std::collections::HashMap;
use std::time::Instant;

/// Git commands that developers use
#[derive(Debug, Clone, PartialEq)]
pub enum GitCommand {
    // Repository operations
    Clone,
    Init,
    Remote,

    // File operations
    Add,
    Remove,
    Move,

    // Commit operations
    Commit,
    Amend,
    Reset,
    Revert,

    // Branch operations
    Branch,
    Checkout,
    Switch,
    Merge,
    Rebase,

    // Remote operations
    Push,
    Pull,
    Fetch,

    // Information commands
    Status,
    Log,
    Diff,
    Show,

    // Stash operations
    Stash,
    StashPop,
    StashApply,

    // Tag operations
    Tag,

    // Configuration
    Config,
}

impl std::fmt::Display for GitCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            GitCommand::Clone => "CLONE",
            GitCommand::Init => "INIT",
            GitCommand::Remote => "REMOTE",
            GitCommand::Add => "ADD",
            GitCommand::Remove => "REMOVE",
            GitCommand::Move => "MOVE",
            GitCommand::Commit => "COMMIT",
            GitCommand::Amend => "AMEND",
            GitCommand::Reset => "RESET",
            GitCommand::Revert => "REVERT",
            GitCommand::Branch => "BRANCH",
            GitCommand::Checkout => "CHECKOUT",
            GitCommand::Switch => "SWITCH",
            GitCommand::Merge => "MERGE",
            GitCommand::Rebase => "REBASE",
            GitCommand::Push => "PUSH",
            GitCommand::Pull => "PULL",
            GitCommand::Fetch => "FETCH",
            GitCommand::Status => "STATUS",
            GitCommand::Log => "LOG",
            GitCommand::Diff => "DIFF",
            GitCommand::Show => "SHOW",
            GitCommand::Stash => "STASH",
            GitCommand::StashPop => "STASH_POP",
            GitCommand::StashApply => "STASH_APPLY",
            GitCommand::Tag => "TAG",
            GitCommand::Config => "CONFIG",
        };
        write!(f, "{}", s)
    }
}

/// Analysis result for a developer's workflow
#[derive(Debug, Clone)]
pub struct WorkflowAnalysis {
    pub developer_id: String,
    pub command_sequence: Vec<String>,
    pub workflow_score: f64,
    pub workflow_patterns: Vec<String>,
    pub productivity_insights: Vec<String>,
    pub workflow_efficiency: EfficiencyLevel,
    pub explanation: String,
    pub similar_workflows_found: usize,
}

/// Workflow efficiency levels
#[derive(Debug, Clone)]
pub enum EfficiencyLevel {
    Optimal,
    Good,
    Suboptimal,
    Inefficient,
}

impl EfficiencyLevel {
    fn from_score(score: f64) -> Self {
        if score >= 0.8 {
            EfficiencyLevel::Inefficient
        } else if score >= 0.6 {
            EfficiencyLevel::Suboptimal
        } else if score >= 0.3 {
            EfficiencyLevel::Good
        } else {
            EfficiencyLevel::Optimal
        }
    }

    fn to_string(&self) -> &str {
        match self {
            EfficiencyLevel::Optimal => "OPTIMAL",
            EfficiencyLevel::Good => "GOOD",
            EfficiencyLevel::Suboptimal => "SUBOPTIMAL",
            EfficiencyLevel::Inefficient => "INEFFICIENT",
        }
    }
}

/// Git workflow analyzer
pub struct GitWorkflowAnalyzer {
    detector: AnomalyDetector,
    workflow_patterns: HashMap<String, Vec<String>>,
    analysis_results: Vec<WorkflowAnalysis>,
    performance_metrics: HashMap<String, f64>,
}

impl GitWorkflowAnalyzer {
    /// Create new git workflow analyzer
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let detector = AnomalyDetector::new(4)?; // 4th order for complex workflow patterns

        Ok(Self {
            detector,
            workflow_patterns: HashMap::new(),
            analysis_results: Vec::new(),
            performance_metrics: HashMap::new(),
        })
    }

    /// Train on efficient git workflows
    pub fn train_on_efficient_workflows(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔍 Training git workflow analyzer on efficient development patterns...");

        let efficient_patterns = self.generate_efficient_workflow_patterns();
        let start_time = Instant::now();

        for pattern in &efficient_patterns {
            self.detector.train(pattern)?;
        }

        let training_time = start_time.elapsed();
        self.performance_metrics.insert(
            "training_time_ms".to_string(),
            training_time.as_millis() as f64,
        );

        println!("✅ Training completed in {:?}", training_time);
        println!(
            "📊 Trained on {} efficient workflow patterns",
            efficient_patterns.len()
        );

        Ok(())
    }

    /// Analyze a developer's git workflow
    pub fn analyze_workflow(
        &mut self,
        commands: &[GitCommand],
        developer_id: &str,
        threshold: f64,
    ) -> Result<Option<WorkflowAnalysis>, Box<dyn std::error::Error>> {
        let command_sequence: Vec<String> = commands.iter().map(|c| c.to_string()).collect();

        // Store workflow pattern
        self.workflow_patterns
            .insert(developer_id.to_string(), command_sequence.clone());

        let detection_start = Instant::now();
        let anomalies = self
            .detector
            .detect_anomalies(&command_sequence, threshold)?;
        let detection_time = detection_start.elapsed();

        self.performance_metrics.insert(
            "last_detection_time_ms".to_string(),
            detection_time.as_millis() as f64,
        );

        if anomalies.is_empty() {
            // Even efficient workflows get analyzed for insights
            let analysis =
                self.create_efficient_workflow_analysis(developer_id, &command_sequence)?;
            return Ok(Some(analysis));
        }

        // Find the highest scoring anomaly
        let max_anomaly = anomalies
            .iter()
            .max_by(|a, b| a.anomaly_strength.partial_cmp(&b.anomaly_strength).unwrap())
            .unwrap();

        // Identify workflow patterns
        let workflow_patterns = self.identify_workflow_patterns(&command_sequence, max_anomaly);

        // Generate productivity insights
        let productivity_insights =
            self.generate_productivity_insights(&command_sequence, max_anomaly);

        // Count similar workflows for context
        let similar_workflows = self.count_similar_workflows(&command_sequence);

        let efficiency = EfficiencyLevel::from_score(max_anomaly.anomaly_strength);

        let explanation = format!(
            "Workflow analysis: Anomaly strength: {:.3}, Likelihood: {:.6}, Information content: {:.3}. This git workflow deviates from efficient patterns with {:.1}% confidence. Workflow optimization opportunities identified.",
            max_anomaly.anomaly_strength,
            max_anomaly.likelihood,
            max_anomaly.information_score,
            (1.0 - max_anomaly.likelihood) * 100.0
        );

        let analysis = WorkflowAnalysis {
            developer_id: developer_id.to_string(),
            command_sequence,
            workflow_score: max_anomaly.anomaly_strength,
            workflow_patterns,
            productivity_insights,
            workflow_efficiency: efficiency,
            explanation,
            similar_workflows_found: similar_workflows,
        };

        self.analysis_results.push(analysis.clone());

        Ok(Some(analysis))
    }

    /// Create analysis for efficient workflows
    fn create_efficient_workflow_analysis(
        &mut self,
        developer_id: &str,
        command_sequence: &[String],
    ) -> Result<WorkflowAnalysis, Box<dyn std::error::Error>> {
        let workflow_patterns = vec!["EFFICIENT_WORKFLOW".to_string()];
        let productivity_insights = self.generate_productivity_insights(
            command_sequence,
            &AnomalyScore {
                sequence: command_sequence.to_vec(),
                likelihood: 0.9,
                log_likelihood: 0.9_f64.ln(),
                information_score: 1.0,
                anomaly_strength: 0.1,
            },
        );

        let analysis = WorkflowAnalysis {
            developer_id: developer_id.to_string(),
            command_sequence: command_sequence.to_vec(),
            workflow_score: 0.1,
            workflow_patterns,
            productivity_insights,
            workflow_efficiency: EfficiencyLevel::Optimal,
            explanation: "Efficient git workflow following best practices. No optimization needed."
                .to_string(),
            similar_workflows_found: self.count_similar_workflows(command_sequence),
        };

        self.analysis_results.push(analysis.clone());
        Ok(analysis)
    }

    /// Identify specific workflow patterns
    fn identify_workflow_patterns(
        &self,
        command_sequence: &[String],
        anomaly: &AnomalyScore,
    ) -> Vec<String> {
        let mut patterns = Vec::new();

        // Check for inefficient commit patterns
        if self.detect_inefficient_commits(command_sequence) {
            patterns.push("INEFFICIENT_COMMIT_PATTERN".to_string());
        }

        // Check for branch management issues
        if self.detect_poor_branch_management(command_sequence) {
            patterns.push("POOR_BRANCH_MANAGEMENT".to_string());
        }

        // Check for merge vs rebase patterns
        if self.detect_merge_heavy_workflow(command_sequence) {
            patterns.push("MERGE_HEAVY_WORKFLOW".to_string());
        }

        // Check for stash overuse
        if self.detect_stash_overuse(command_sequence) {
            patterns.push("STASH_OVERUSE".to_string());
        }

        // Check for status checking patterns
        if self.detect_excessive_status_checking(command_sequence) {
            patterns.push("EXCESSIVE_STATUS_CHECKING".to_string());
        }

        // Check for reset/revert patterns
        if self.detect_frequent_corrections(command_sequence) {
            patterns.push("FREQUENT_CORRECTIONS".to_string());
        }

        // High information content suggests very unusual workflow
        if anomaly.information_score > 4.0 {
            patterns.push("HIGHLY_UNUSUAL_WORKFLOW".to_string());
        }

        patterns
    }

    /// Generate productivity insights
    fn generate_productivity_insights(
        &self,
        command_sequence: &[String],
        anomaly: &AnomalyScore,
    ) -> Vec<String> {
        let mut insights = Vec::new();

        // Analyze commit frequency
        let commit_count = command_sequence
            .iter()
            .filter(|c| c.contains("COMMIT"))
            .count();
        if commit_count > 10 {
            insights.push("Consider making larger, more meaningful commits".to_string());
        } else if commit_count < 2 {
            insights.push("Consider committing more frequently for better history".to_string());
        }

        // Analyze branch usage
        let branch_ops = command_sequence
            .iter()
            .filter(|c| c.contains("BRANCH") || c.contains("CHECKOUT") || c.contains("SWITCH"))
            .count();
        if branch_ops > 8 {
            insights.push("Consider simplifying branch workflow".to_string());
        }

        // Analyze merge vs rebase
        let merge_count = command_sequence
            .iter()
            .filter(|c| c.contains("MERGE"))
            .count();
        let rebase_count = command_sequence
            .iter()
            .filter(|c| c.contains("REBASE"))
            .count();
        if merge_count > rebase_count * 2 {
            insights.push("Consider using rebase for cleaner history".to_string());
        }

        // Analyze stash usage
        let stash_count = command_sequence
            .iter()
            .filter(|c| c.contains("STASH"))
            .count();
        if stash_count > 5 {
            insights.push("High stash usage - consider better branch management".to_string());
        }

        // Analyze status checking
        let status_count = command_sequence
            .iter()
            .filter(|c| c.contains("STATUS"))
            .count();
        if status_count > command_sequence.len() / 3 {
            insights.push(
                "Frequent status checking - consider using git aliases or IDE integration"
                    .to_string(),
            );
        }

        // Efficiency insights
        if anomaly.anomaly_strength < 0.2 {
            insights.push("Efficient workflow - following git best practices".to_string());
        } else if anomaly.anomaly_strength > 0.6 {
            insights.push("Workflow has significant optimization opportunities".to_string());
        }

        insights
    }

    /// Detect inefficient commit patterns
    fn detect_inefficient_commits(&self, commands: &[String]) -> bool {
        // Too many small commits or amends
        let commit_count = commands.iter().filter(|c| c.contains("COMMIT")).count();
        let amend_count = commands.iter().filter(|c| c.contains("AMEND")).count();

        commit_count > 10 || amend_count > 3
    }

    /// Detect poor branch management
    fn detect_poor_branch_management(&self, commands: &[String]) -> bool {
        let branch_ops = commands
            .iter()
            .filter(|c| c.contains("BRANCH") || c.contains("CHECKOUT") || c.contains("SWITCH"))
            .count();

        branch_ops > 8
    }

    /// Detect merge-heavy workflows
    fn detect_merge_heavy_workflow(&self, commands: &[String]) -> bool {
        let merge_count = commands.iter().filter(|c| c.contains("MERGE")).count();
        let rebase_count = commands.iter().filter(|c| c.contains("REBASE")).count();

        merge_count > 3 && merge_count > rebase_count * 2
    }

    /// Detect stash overuse
    fn detect_stash_overuse(&self, commands: &[String]) -> bool {
        let stash_count = commands.iter().filter(|c| c.contains("STASH")).count();
        stash_count > 5
    }

    /// Detect excessive status checking
    fn detect_excessive_status_checking(&self, commands: &[String]) -> bool {
        let status_count = commands.iter().filter(|c| c.contains("STATUS")).count();
        status_count > commands.len() / 3
    }

    /// Detect frequent corrections
    fn detect_frequent_corrections(&self, commands: &[String]) -> bool {
        let correction_count = commands
            .iter()
            .filter(|c| c.contains("RESET") || c.contains("REVERT") || c.contains("AMEND"))
            .count();

        correction_count > 4
    }

    /// Count workflows with similar patterns
    fn count_similar_workflows(&self, target_sequence: &[String]) -> usize {
        self.workflow_patterns
            .values()
            .filter(|sequence| {
                let common_commands = sequence
                    .iter()
                    .filter(|command| target_sequence.contains(command))
                    .count();

                common_commands as f64 / sequence.len() as f64 > 0.6
            })
            .count()
    }

    /// Generate analysis summary
    pub fn generate_workflow_summary(&self) -> WorkflowSummary {
        let total_workflows = self.analysis_results.len();
        let inefficient_workflows = self
            .analysis_results
            .iter()
            .filter(|r| {
                matches!(
                    r.workflow_efficiency,
                    EfficiencyLevel::Inefficient | EfficiencyLevel::Suboptimal
                )
            })
            .count();

        let avg_detection_time = self
            .performance_metrics
            .get("last_detection_time_ms")
            .unwrap_or(&0.0);

        let total_commands: usize = self
            .analysis_results
            .iter()
            .map(|r| r.command_sequence.len())
            .sum();

        WorkflowSummary {
            total_workflows_analyzed: total_workflows,
            inefficient_workflows_found: inefficient_workflows,
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
            for pattern in &result.workflow_patterns {
                all_patterns.insert(pattern.clone());
            }
        }
        all_patterns.len()
    }

    /// Count optimization opportunities
    fn count_optimization_opportunities(&self) -> usize {
        self.analysis_results
            .iter()
            .map(|r| r.productivity_insights.len())
            .sum()
    }

    /// Generate efficient workflow patterns for training
    fn generate_efficient_workflow_patterns(&self) -> Vec<Vec<String>> {
        let mut patterns = Vec::new();

        // Base efficient git workflow patterns
        let base_patterns = [
            // Feature branch workflow
            vec![
                GitCommand::Pull,
                GitCommand::Branch,
                GitCommand::Checkout,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Push,
                GitCommand::Checkout,
                GitCommand::Merge,
            ],
            // Simple fix workflow
            vec![
                GitCommand::Status,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Push,
            ],
            // Rebase workflow
            vec![
                GitCommand::Fetch,
                GitCommand::Rebase,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Push,
            ],
            // Review and merge
            vec![
                GitCommand::Pull,
                GitCommand::Log,
                GitCommand::Diff,
                GitCommand::Merge,
                GitCommand::Push,
            ],
            // Stash and switch
            vec![
                GitCommand::Stash,
                GitCommand::Checkout,
                GitCommand::Pull,
                GitCommand::Checkout,
                GitCommand::StashPop,
            ],
            // Tag release
            vec![GitCommand::Pull, GitCommand::Tag, GitCommand::Push],
            // Clean workflow
            vec![
                GitCommand::Status,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Pull,
                GitCommand::Push,
            ],
            // Branch cleanup
            vec![GitCommand::Checkout, GitCommand::Branch, GitCommand::Push],
        ];

        // Extended workflow patterns for comprehensive training
        let extended_patterns = vec![
            // Complex feature development
            vec![
                GitCommand::Pull,
                GitCommand::Branch,
                GitCommand::Checkout,
                GitCommand::Status,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Push,
                GitCommand::Pull,
                GitCommand::Merge,
                GitCommand::Push,
            ],
            // Hotfix workflow
            vec![
                GitCommand::Checkout,
                GitCommand::Pull,
                GitCommand::Branch,
                GitCommand::Checkout,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Push,
                GitCommand::Checkout,
                GitCommand::Merge,
                GitCommand::Tag,
                GitCommand::Push,
            ],
            // Collaborative development
            vec![
                GitCommand::Pull,
                GitCommand::Status,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Pull,
                GitCommand::Rebase,
                GitCommand::Push,
            ],
            // Release preparation
            vec![
                GitCommand::Pull,
                GitCommand::Log,
                GitCommand::Diff,
                GitCommand::Status,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Tag,
                GitCommand::Push,
            ],
            // Bug investigation
            vec![
                GitCommand::Status,
                GitCommand::Log,
                GitCommand::Show,
                GitCommand::Diff,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Push,
            ],
            // Merge conflict resolution
            vec![
                GitCommand::Pull,
                GitCommand::Status,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Pull,
                GitCommand::Status,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Push,
            ],
            // Code review workflow
            vec![
                GitCommand::Fetch,
                GitCommand::Checkout,
                GitCommand::Log,
                GitCommand::Diff,
                GitCommand::Show,
                GitCommand::Checkout,
                GitCommand::Merge,
            ],
            // Experimental branch
            vec![
                GitCommand::Branch,
                GitCommand::Checkout,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Checkout,
                GitCommand::Branch,
                GitCommand::Merge,
            ],
            // Documentation update
            vec![
                GitCommand::Pull,
                GitCommand::Status,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Push,
            ],
            // Configuration changes
            vec![
                GitCommand::Status,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Pull,
                GitCommand::Push,
            ],
            // Multi-file changes
            vec![
                GitCommand::Status,
                GitCommand::Add,
                GitCommand::Status,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Push,
            ],
            // Backup before major changes
            vec![
                GitCommand::Branch,
                GitCommand::Checkout,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Checkout,
                GitCommand::Add,
                GitCommand::Commit,
            ],
        ];

        // Variation patterns to create realistic diversity
        let variation_commands = [
            vec![GitCommand::Status],
            vec![GitCommand::Diff],
            vec![GitCommand::Log],
            vec![GitCommand::Show],
            vec![GitCommand::Status, GitCommand::Diff],
            vec![GitCommand::Log, GitCommand::Show],
            vec![GitCommand::Fetch],
            vec![GitCommand::Pull],
        ];

        // Generate comprehensive training dataset with long sequences
        for repetition in 0..80 {
            // Create long realistic workflow sequences by combining multiple patterns
            for base_idx in 0..base_patterns.len() {
                let mut long_sequence = Vec::new();

                // Start with a base pattern
                long_sequence.extend(base_patterns[base_idx].iter().cloned());

                // Add 3-5 additional workflow segments to create substantial sequences
                let num_segments = 3 + (repetition % 3);
                for segment in 0..num_segments {
                    // Add variation commands between segments
                    if !variation_commands.is_empty() {
                        let variation = &variation_commands[segment % variation_commands.len()];
                        long_sequence.extend(variation.iter().cloned());
                    }

                    // Add an extended pattern
                    let ext_idx = (base_idx + segment) % extended_patterns.len();
                    long_sequence.extend(extended_patterns[ext_idx].iter().cloned());

                    // Add another base pattern for continuity
                    let next_base_idx = (base_idx + segment + 1) % base_patterns.len();
                    long_sequence.extend(base_patterns[next_base_idx].iter().cloned());
                }

                // Add final variation to complete the workflow
                if !variation_commands.is_empty() {
                    let final_variation =
                        &variation_commands[repetition % variation_commands.len()];
                    long_sequence.extend(final_variation.iter().cloned());
                }

                // Convert to strings and add to patterns
                let string_pattern: Vec<String> =
                    long_sequence.iter().map(|c| c.to_string()).collect();
                patterns.push(string_pattern);
            }

            // Create mega-sequences by combining multiple extended patterns
            if repetition % 3 == 0 {
                let mut mega_sequence = Vec::new();

                // Combine 4-6 extended patterns into one large sequence
                let num_patterns = 4 + (repetition % 3);
                for i in 0..num_patterns {
                    let pattern_idx = (repetition + i) % extended_patterns.len();
                    mega_sequence.extend(extended_patterns[pattern_idx].iter().cloned());

                    // Add connecting variations
                    if i < num_patterns - 1 && !variation_commands.is_empty() {
                        let var_idx = i % variation_commands.len();
                        mega_sequence.extend(variation_commands[var_idx].iter().cloned());
                    }
                }

                let string_pattern: Vec<String> =
                    mega_sequence.iter().map(|c| c.to_string()).collect();
                patterns.push(string_pattern);
            }
        }

        // Add realistic full-day workflow patterns (long sequences)
        for _day in 0..30 {
            // Create full day workflow by combining morning, afternoon, and evening
            let mut full_day_workflow = Vec::new();

            // Morning startup (8-10 commands)
            full_day_workflow.extend(vec![
                GitCommand::Pull,
                GitCommand::Status,
                GitCommand::Log,
                GitCommand::Branch,
                GitCommand::Checkout,
                GitCommand::Status,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Push,
                GitCommand::Status,
            ]);

            // Mid-morning development (10-12 commands)
            full_day_workflow.extend(vec![
                GitCommand::Pull,
                GitCommand::Status,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Status,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Pull,
                GitCommand::Rebase,
                GitCommand::Push,
                GitCommand::Status,
                GitCommand::Log,
            ]);

            // Afternoon feature work (12-15 commands)
            full_day_workflow.extend(vec![
                GitCommand::Branch,
                GitCommand::Checkout,
                GitCommand::Status,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Status,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Push,
                GitCommand::Pull,
                GitCommand::Status,
                GitCommand::Checkout,
                GitCommand::Merge,
                GitCommand::Push,
                GitCommand::Log,
            ]);

            // End of day cleanup (8-10 commands)
            full_day_workflow.extend(vec![
                GitCommand::Status,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Push,
                GitCommand::Log,
                GitCommand::Status,
                GitCommand::Branch,
                GitCommand::Tag,
                GitCommand::Push,
                GitCommand::Status,
            ]);

            let string_pattern: Vec<String> =
                full_day_workflow.iter().map(|c| c.to_string()).collect();
            patterns.push(string_pattern);

            // Create alternative full-day patterns with different workflows
            let mut alt_day_workflow = Vec::new();

            // Bug fixing day pattern (15-20 commands)
            alt_day_workflow.extend(vec![
                GitCommand::Pull,
                GitCommand::Status,
                GitCommand::Log,
                GitCommand::Show,
                GitCommand::Branch,
                GitCommand::Checkout,
                GitCommand::Status,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Status,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Push,
                GitCommand::Pull,
                GitCommand::Status,
                GitCommand::Checkout,
                GitCommand::Merge,
                GitCommand::Push,
                GitCommand::Log,
                GitCommand::Status,
            ]);

            // Feature development continuation (12-15 commands)
            alt_day_workflow.extend(vec![
                GitCommand::Status,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Status,
                GitCommand::Add,
                GitCommand::Commit,
                GitCommand::Pull,
                GitCommand::Rebase,
                GitCommand::Push,
                GitCommand::Status,
                GitCommand::Log,
                GitCommand::Show,
                GitCommand::Status,
                GitCommand::Branch,
                GitCommand::Tag,
            ]);

            let alt_string_pattern: Vec<String> =
                alt_day_workflow.iter().map(|c| c.to_string()).collect();
            patterns.push(alt_string_pattern);
        }

        patterns
    }
}

/// Workflow analysis summary data
#[derive(Debug)]
pub struct WorkflowSummary {
    pub total_workflows_analyzed: usize,
    pub inefficient_workflows_found: usize,
    pub average_analysis_time_ms: f64,
    pub patterns_identified: usize,
    pub total_commands_analyzed: usize,
    pub optimization_opportunities: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 GIT WORKFLOW PATTERN ANALYSIS");
    println!("=================================");
    println!();

    // Initialize analyzer
    let mut analyzer = GitWorkflowAnalyzer::new()?;

    // Train on efficient patterns
    analyzer.train_on_efficient_workflows()?;
    println!();

    // Analyze different workflow scenarios
    println!("🔍 ANALYZING DEVELOPER WORKFLOWS");
    println!("=================================");

    // Scenario 1: Efficient feature branch workflow
    println!("\n📊 Scenario 1: Efficient Feature Branch Workflow");
    let efficient_workflow = vec![
        GitCommand::Pull,
        GitCommand::Branch,
        GitCommand::Checkout,
        GitCommand::Add,
        GitCommand::Commit,
        GitCommand::Push,
        GitCommand::Checkout,
        GitCommand::Merge,
    ];

    let result = analyzer.analyze_workflow(&efficient_workflow, "dev_001", 0.3)?;
    if let Some(analysis) = result {
        println!(
            "✅ Workflow Efficiency: {}",
            analysis.workflow_efficiency.to_string()
        );
        println!("   Patterns: {:?}", analysis.workflow_patterns);
        println!("   Insights: {:?}", analysis.productivity_insights);
    }

    // Scenario 2: Inefficient commit pattern
    println!("\n🚨 Scenario 2: Inefficient Commit Pattern");
    let inefficient_commits = vec![
        GitCommand::Add,
        GitCommand::Commit,
        GitCommand::Amend,
        GitCommand::Add,
        GitCommand::Commit,
        GitCommand::Amend,
        GitCommand::Add,
        GitCommand::Commit,
        GitCommand::Amend,
        GitCommand::Push,
    ];

    let result = analyzer.analyze_workflow(&inefficient_commits, "dev_002", 0.3)?;
    if let Some(analysis) = result {
        println!(
            "⚠️  Workflow Efficiency: {}",
            analysis.workflow_efficiency.to_string()
        );
        println!("   Workflow Score: {:.3}", analysis.workflow_score);
        println!("   Patterns: {:?}", analysis.workflow_patterns);
        println!("   Insights: {:?}", analysis.productivity_insights);
    }

    // Scenario 3: Stash-heavy workflow
    println!("\n💾 Scenario 3: Stash-Heavy Workflow");
    let stash_heavy = vec![
        GitCommand::Stash,
        GitCommand::Checkout,
        GitCommand::Stash,
        GitCommand::Checkout,
        GitCommand::StashPop,
        GitCommand::Add,
        GitCommand::Stash,
        GitCommand::Checkout,
        GitCommand::StashApply,
        GitCommand::Commit,
    ];

    let result = analyzer.analyze_workflow(&stash_heavy, "dev_003", 0.3)?;
    if let Some(analysis) = result {
        println!(
            "⚠️  Workflow Efficiency: {}",
            analysis.workflow_efficiency.to_string()
        );
        println!("   Patterns: {:?}", analysis.workflow_patterns);
        println!(
            "   Similar Workflows Found: {}",
            analysis.similar_workflows_found
        );
        println!("   Insights: {:?}", analysis.productivity_insights);
    }

    // Scenario 4: Status-checking heavy workflow
    println!("\n🔍 Scenario 4: Status-Heavy Workflow");
    let status_heavy = vec![
        GitCommand::Status,
        GitCommand::Add,
        GitCommand::Status,
        GitCommand::Commit,
        GitCommand::Status,
        GitCommand::Push,
        GitCommand::Status,
        GitCommand::Pull,
        GitCommand::Status,
    ];

    let result = analyzer.analyze_workflow(&status_heavy, "dev_004", 0.3)?;
    if let Some(analysis) = result {
        println!(
            "⚠️  Workflow Efficiency: {}",
            analysis.workflow_efficiency.to_string()
        );
        println!("   Patterns: {:?}", analysis.workflow_patterns);
        println!("   Insights: {:?}", analysis.productivity_insights);
    }

    // Generate summary
    println!("\n📊 WORKFLOW ANALYSIS SUMMARY");
    println!("============================");
    let summary = analyzer.generate_workflow_summary();
    println!("Workflows Analyzed: {}", summary.total_workflows_analyzed);
    println!(
        "Inefficient Workflows Found: {}",
        summary.inefficient_workflows_found
    );
    println!(
        "Average Analysis Time: {:.2}ms",
        summary.average_analysis_time_ms
    );
    println!(
        "Unique Patterns Identified: {}",
        summary.patterns_identified
    );
    println!(
        "Total Commands Analyzed: {}",
        summary.total_commands_analyzed
    );
    println!(
        "Optimization Opportunities: {}",
        summary.optimization_opportunities
    );

    Ok(())
}
