"""
Dual Calibration Analysis Tool
=============================

This tool analyzes and compares the performance of the dual calibration pipeline,
providing insights into personalization effectiveness and calibration quality.

Features:
- Calibration quality assessment
- Adaptive vs. universal MI comparison
- Individual threshold analysis
- Session-to-session consistency evaluation
- Calibration effectiveness metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import glob
from datetime import datetime
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DualCalibrationAnalyzer:
    """Comprehensive analysis of dual calibration system performance"""
    
    def __init__(self, base_dir='.'):
        self.base_dir = Path(base_dir)
        self.user_config_dir = self.base_dir / 'user_configs'
        self.logs_dir = self.base_dir / 'logs'
        self.vis_dir = self.base_dir / 'visualizations'
        self.analysis_dir = self.base_dir / 'analysis'
        
        # Ensure analysis directory exists
        self.analysis_dir.mkdir(exist_ok=True)
        
        self.users_data = {}
        self.calibration_data = {}
        
    def load_all_user_data(self):
        """Load all available user calibration and session data"""
        print("Loading user data...")
        
        # Load calibration configs
        config_files = list(self.user_config_dir.glob('*_dual_calibration.json'))
        
        for config_file in config_files:
            user_id = config_file.stem.replace('_dual_calibration', '')
            
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                self.calibration_data[user_id] = config_data
                
                # Load baseline features
                baseline_csv = config_data.get('baseline_csv')
                if baseline_csv and os.path.exists(baseline_csv):
                    baseline_df = pd.read_csv(baseline_csv)
                    self.calibration_data[user_id]['baseline_features'] = baseline_df
                
                print(f"  ✓ Loaded calibration for user: {user_id}")
                
            except Exception as e:
                print(f"  ✗ Failed to load calibration for {user_id}: {e}")
        
        # Load session data
        session_files = list(self.logs_dir.glob('*_dual_calibration_session_*.csv'))
        
        for session_file in session_files:
            try:
                # Extract user_id from filename
                filename = session_file.stem
                parts = filename.split('_dual_calibration_session_')
                if len(parts) == 2:
                    user_id = parts[0]
                    timestamp = parts[1]
                    
                    session_df = pd.read_csv(session_file)
                    
                    if user_id not in self.users_data:
                        self.users_data[user_id] = []
                    
                    session_df['session_timestamp'] = timestamp
                    session_df['session_file'] = str(session_file)
                    self.users_data[user_id].append(session_df)
                    
                    print(f"  ✓ Loaded session for user {user_id}: {timestamp}")
                
            except Exception as e:
                print(f"  ✗ Failed to load session {session_file}: {e}")
        
        print(f"\nLoaded data for {len(self.calibration_data)} users with calibration")
        print(f"Loaded session data for {len(self.users_data)} users")
    
    def analyze_calibration_quality(self, user_id=None):
        """Analyze the quality of dual calibration for users"""
        if user_id:
            users_to_analyze = [user_id] if user_id in self.calibration_data else []
        else:
            users_to_analyze = list(self.calibration_data.keys())
        
        if not users_to_analyze:
            print("No calibration data found!")
            return
        
        print(f"\n{'='*60}")
        print("DUAL CALIBRATION QUALITY ANALYSIS")
        print(f"{'='*60}")
        
        quality_results = []
        
        for user_id in users_to_analyze:
            calib_data = self.calibration_data[user_id]
            
            if 'baseline_features' not in calib_data:
                continue
            
            df = calib_data['baseline_features']
            relaxed_df = df[df['phase'] == 'relaxed']
            focused_df = df[df['phase'] == 'focused']
            
            if len(relaxed_df) == 0 or len(focused_df) == 0:
                continue
            
            # Calculate quality metrics
            quality_metrics = self._calculate_calibration_quality(relaxed_df, focused_df, user_id)
            quality_results.append(quality_metrics)
        
        if quality_results:
            self._plot_calibration_quality_summary(quality_results)
            self._save_calibration_quality_report(quality_results)
        
        return quality_results
    
    def _calculate_calibration_quality(self, relaxed_df, focused_df, user_id):
        """Calculate comprehensive calibration quality metrics"""
        feature_cols = ['theta_fz', 'alpha_po', 'faa', 'beta_frontal', 'eda_norm']
        
        metrics = {
            'user_id': user_id,
            'relaxed_samples': len(relaxed_df),
            'focused_samples': len(focused_df),
            'feature_separability': {},
            'stability_metrics': {},
            'dynamic_range': {},
            'overall_quality': 0
        }
        
        print(f"\n[{user_id}] Calibration Quality Analysis:")
        print(f"  Relaxed samples: {len(relaxed_df)}")
        print(f"  Focused samples: {len(focused_df)}")
        
        # Feature separability analysis
        separability_scores = []
        
        for feature in feature_cols:
            relaxed_vals = relaxed_df[feature].values
            focused_vals = focused_df[feature].values
            
            # Effect size (Cohen's d)
            cohens_d = self._calculate_cohens_d(relaxed_vals, focused_vals)
            
            # Statistical significance
            t_stat, p_value = stats.ttest_ind(relaxed_vals, focused_vals)
            
            # Overlap coefficient (lower is better separation)
            overlap = self._calculate_overlap_coefficient(relaxed_vals, focused_vals)
            
            separability_score = abs(cohens_d) * (1 - overlap) * (1 if p_value < 0.05 else 0.5)
            separability_scores.append(separability_score)
            
            metrics['feature_separability'][feature] = {
                'cohens_d': cohens_d,
                'p_value': p_value,
                'overlap_coeff': overlap,
                'separability_score': separability_score
            }
            
            print(f"  {feature}: d={cohens_d:.3f}, p={p_value:.3f}, overlap={overlap:.3f}")
        
        # Stability within phases
        for phase_name, phase_df in [('relaxed', relaxed_df), ('focused', focused_df)]:
            stability_metrics = {}
            
            for feature in feature_cols:
                vals = phase_df[feature].values
                cv = np.std(vals) / np.mean(vals) if np.mean(vals) != 0 else float('inf')
                stability_metrics[feature] = cv
            
            metrics['stability_metrics'][phase_name] = stability_metrics
        
        # Dynamic range assessment
        adaptive_thresholds = self.calibration_data[user_id].get('adaptive_thresholds', {})
        if 'adaptive_mapping' in adaptive_thresholds:
            mapping = adaptive_thresholds['adaptive_mapping']
            dynamic_range = mapping.get('dynamic_range', 0)
            low_thresh = mapping.get('low_threshold', 0)
            high_thresh = mapping.get('high_threshold', 1)
            
            metrics['dynamic_range'] = {
                'mi_range': dynamic_range,
                'low_threshold': low_thresh,
                'high_threshold': high_thresh,
                'range_quality': min(dynamic_range / 0.3, 1.0)  # Normalize to 0.3 as good range
            }
            
            print(f"  MI Dynamic Range: {dynamic_range:.3f}")
            print(f"  Range Quality: {metrics['dynamic_range']['range_quality']:.3f}")
        
        # Overall quality score
        avg_separability = np.mean(separability_scores)
        range_quality = metrics['dynamic_range'].get('range_quality', 0.5)
        
        # Stability penalty (higher CV = lower quality)
        stability_penalties = []
        for phase in ['relaxed', 'focused']:
            if phase in metrics['stability_metrics']:
                cvs = list(metrics['stability_metrics'][phase].values())
                avg_cv = np.mean([cv for cv in cvs if not np.isinf(cv)])
                stability_penalty = max(0, 1 - avg_cv / 0.5)  # Penalize CV > 0.5
                stability_penalties.append(stability_penalty)
        
        avg_stability = np.mean(stability_penalties) if stability_penalties else 0.5
        
        overall_quality = (avg_separability * 0.5 + range_quality * 0.3 + avg_stability * 0.2)
        metrics['overall_quality'] = overall_quality
        
        print(f"  Overall Quality Score: {overall_quality:.3f}")
        
        return metrics
    
    def _calculate_cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0
        
        return (mean1 - mean2) / pooled_std
    
    def _calculate_overlap_coefficient(self, group1, group2):
        """Calculate overlap coefficient between two distributions"""
        min_val = min(np.min(group1), np.min(group2))
        max_val = max(np.max(group1), np.max(group2))
        
        # Create histograms
        bins = np.linspace(min_val, max_val, 50)
        hist1, _ = np.histogram(group1, bins=bins, density=True)
        hist2, _ = np.histogram(group2, bins=bins, density=True)
        
        # Calculate overlap
        overlap = np.sum(np.minimum(hist1, hist2)) * (bins[1] - bins[0])
        return overlap
    
    def analyze_adaptive_effectiveness(self, user_id=None):
        """Analyze how effective adaptive thresholds are compared to universal"""
        if user_id:
            users_to_analyze = [user_id] if user_id in self.users_data else []
        else:
            users_to_analyze = list(self.users_data.keys())
        
        if not users_to_analyze:
            print("No session data found!")
            return
        
        print(f"\n{'='*60}")
        print("ADAPTIVE VS UNIVERSAL MI EFFECTIVENESS")
        print(f"{'='*60}")
        
        effectiveness_results = []
        
        for user_id in users_to_analyze:
            sessions = self.users_data[user_id]
            
            for session_df in sessions:
                if 'adaptive_mi' not in session_df.columns or 'universal_mi' not in session_df.columns:
                    continue
                
                # Calculate effectiveness metrics
                effectiveness = self._calculate_adaptive_effectiveness(session_df, user_id)
                effectiveness_results.append(effectiveness)
        
        if effectiveness_results:
            self._plot_adaptive_effectiveness(effectiveness_results)
            self._save_adaptive_effectiveness_report(effectiveness_results)
        
        return effectiveness_results
    
    def _calculate_adaptive_effectiveness(self, session_df, user_id):
        """Calculate metrics comparing adaptive vs universal MI"""
        adaptive_mi = session_df['adaptive_mi'].values
        universal_mi = session_df['universal_mi'].values
        
        metrics = {
            'user_id': user_id,
            'session_timestamp': session_df['session_timestamp'].iloc[0],
            'n_samples': len(session_df),
            'duration_minutes': len(session_df) / 60.0,  # Assuming 1 Hz
        }
        
        # Range and variability analysis
        metrics.update({
            'adaptive_range': np.max(adaptive_mi) - np.min(adaptive_mi),
            'universal_range': np.max(universal_mi) - np.min(universal_mi),
            'adaptive_std': np.std(adaptive_mi),
            'universal_std': np.std(universal_mi),
            'adaptive_mean': np.mean(adaptive_mi),
            'universal_mean': np.mean(universal_mi),
        })
        
        # Range enhancement ratio
        range_enhancement = metrics['adaptive_range'] / max(metrics['universal_range'], 0.001)
        metrics['range_enhancement'] = range_enhancement
        
        # Utilization of full scale (0-1 for adaptive, variable for universal)
        adaptive_utilization = metrics['adaptive_range']  # Already 0-1 scale
        universal_utilization = metrics['universal_range'] / (np.max(universal_mi) - np.min(universal_mi) + 0.001)
        metrics['scale_utilization'] = adaptive_utilization / max(universal_utilization, 0.001)
        
        # Responsiveness (rate of change)
        adaptive_diff = np.abs(np.diff(adaptive_mi))
        universal_diff = np.abs(np.diff(universal_mi))
        
        metrics['adaptive_responsiveness'] = np.mean(adaptive_diff)
        metrics['universal_responsiveness'] = np.mean(universal_diff)
        metrics['responsiveness_ratio'] = metrics['adaptive_responsiveness'] / max(metrics['universal_responsiveness'], 0.001)
        
        # Correlation analysis
        correlation = np.corrcoef(adaptive_mi, universal_mi)[0, 1]
        metrics['adaptive_universal_correlation'] = correlation
        
        # Personalization effectiveness score
        personalization_score = (
            min(range_enhancement / 2.0, 1.0) * 0.4 +  # Range enhancement (capped at 2x)
            min(adaptive_utilization / 0.8, 1.0) * 0.3 +  # Scale utilization
            min(metrics['responsiveness_ratio'] / 1.5, 1.0) * 0.3  # Responsiveness
        )
        
        metrics['personalization_effectiveness'] = personalization_score
        
        return metrics
    
    def analyze_session_consistency(self, user_id=None):
        """Analyze consistency across sessions for users"""
        if user_id:
            users_to_analyze = [user_id] if user_id in self.users_data else []
        else:
            # Only analyze users with multiple sessions
            users_to_analyze = [uid for uid, sessions in self.users_data.items() if len(sessions) > 1]
        
        if not users_to_analyze:
            print("No users with multiple sessions found!")
            return
        
        print(f"\n{'='*60}")
        print("SESSION-TO-SESSION CONSISTENCY ANALYSIS")
        print(f"{'='*60}")
        
        consistency_results = []
        
        for user_id in users_to_analyze:
            sessions = self.users_data[user_id]
            
            if len(sessions) < 2:
                continue
            
            consistency = self._calculate_session_consistency(sessions, user_id)
            consistency_results.append(consistency)
        
        if consistency_results:
            self._plot_session_consistency(consistency_results)
            self._save_session_consistency_report(consistency_results)
        
        return consistency_results
    
    def _calculate_session_consistency(self, sessions, user_id):
        """Calculate consistency metrics across sessions"""
        metrics = {
            'user_id': user_id,
            'n_sessions': len(sessions),
            'session_timestamps': []
        }
        
        # Collect session means and characteristics
        session_means = {'adaptive': [], 'universal': []}
        session_stds = {'adaptive': [], 'universal': []}
        session_ranges = {'adaptive': [], 'universal': []}
        
        for session_df in sessions:
            metrics['session_timestamps'].append(session_df['session_timestamp'].iloc[0])
            
            for mi_type in ['adaptive', 'universal']:
                col_name = f'{mi_type}_mi'
                if col_name in session_df.columns:
                    values = session_df[col_name].values
                    session_means[mi_type].append(np.mean(values))
                    session_stds[mi_type].append(np.std(values))
                    session_ranges[mi_type].append(np.max(values) - np.min(values))
        
        # Calculate consistency metrics
        for mi_type in ['adaptive', 'universal']:
            if len(session_means[mi_type]) > 1:
                means = np.array(session_means[mi_type])
                stds = np.array(session_stds[mi_type])
                ranges = np.array(session_ranges[mi_type])
                
                # Coefficient of variation across sessions
                mean_cv = np.std(means) / max(np.mean(means), 0.001)
                std_cv = np.std(stds) / max(np.mean(stds), 0.001)
                range_cv = np.std(ranges) / max(np.mean(ranges), 0.001)
                
                metrics[f'{mi_type}_mean_consistency'] = 1 / (1 + mean_cv)  # Higher is more consistent
                metrics[f'{mi_type}_std_consistency'] = 1 / (1 + std_cv)
                metrics[f'{mi_type}_range_consistency'] = 1 / (1 + range_cv)
                
                # Overall consistency score
                overall_consistency = np.mean([
                    metrics[f'{mi_type}_mean_consistency'],
                    metrics[f'{mi_type}_std_consistency'],
                    metrics[f'{mi_type}_range_consistency']
                ])
                metrics[f'{mi_type}_overall_consistency'] = overall_consistency
        
        return metrics
    
    def _plot_calibration_quality_summary(self, quality_results):
        """Plot comprehensive calibration quality summary"""
        if not quality_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dual Calibration Quality Analysis', fontsize=16, fontweight='bold')
        
        # Overall quality scores
        user_ids = [r['user_id'] for r in quality_results]
        quality_scores = [r['overall_quality'] for r in quality_results]
        
        axes[0, 0].bar(range(len(user_ids)), quality_scores, alpha=0.7, color='steelblue')
        axes[0, 0].set_title('Overall Calibration Quality Scores')
        axes[0, 0].set_ylabel('Quality Score (0-1)')
        axes[0, 0].set_xticks(range(len(user_ids)))
        axes[0, 0].set_xticklabels(user_ids, rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Dynamic range quality
        range_qualities = []
        for r in quality_results:
            if 'dynamic_range' in r and 'range_quality' in r['dynamic_range']:
                range_qualities.append(r['dynamic_range']['range_quality'])
            else:
                range_qualities.append(0)
        
        axes[0, 1].bar(range(len(user_ids)), range_qualities, alpha=0.7, color='darkgreen')
        axes[0, 1].set_title('MI Dynamic Range Quality')
        axes[0, 1].set_ylabel('Range Quality (0-1)')
        axes[0, 1].set_xticks(range(len(user_ids)))
        axes[0, 1].set_xticklabels(user_ids, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Feature separability heatmap
        features = ['theta_fz', 'alpha_po', 'faa', 'beta_frontal', 'eda_norm']
        separability_matrix = []
        
        for r in quality_results:
            row = []
            for feature in features:
                if feature in r['feature_separability']:
                    row.append(r['feature_separability'][feature]['separability_score'])
                else:
                    row.append(0)
            separability_matrix.append(row)
        
        im = axes[1, 0].imshow(separability_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=2)
        axes[1, 0].set_title('Feature Separability (Relaxed vs Focused)')
        axes[1, 0].set_ylabel('Users')
        axes[1, 0].set_xlabel('Features')
        axes[1, 0].set_yticks(range(len(user_ids)))
        axes[1, 0].set_yticklabels(user_ids)
        axes[1, 0].set_xticks(range(len(features)))
        axes[1, 0].set_xticklabels(features, rotation=45)
        plt.colorbar(im, ax=axes[1, 0], label='Separability Score')
        
        # Sample size adequacy
        relaxed_samples = [r['relaxed_samples'] for r in quality_results]
        focused_samples = [r['focused_samples'] for r in quality_results]
        
        x = np.arange(len(user_ids))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, relaxed_samples, width, label='Relaxed', alpha=0.7, color='lightblue')
        axes[1, 1].bar(x + width/2, focused_samples, width, label='Focused', alpha=0.7, color='lightcoral')
        axes[1, 1].set_title('Calibration Sample Counts')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(user_ids, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add target lines
        axes[1, 1].axhline(y=15, color='green', linestyle='--', alpha=0.7, label='Minimum (15)')
        axes[1, 1].axhline(y=25, color='blue', linestyle='--', alpha=0.7, label='Good (25)')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = self.analysis_dir / f'calibration_quality_analysis_{timestamp}.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"[PLOT] Calibration quality analysis saved to {fname}")
        plt.close()
    
    def _plot_adaptive_effectiveness(self, effectiveness_results):
        """Plot adaptive vs universal MI effectiveness"""
        if not effectiveness_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Adaptive MI Effectiveness Analysis', fontsize=16, fontweight='bold')
        
        # Extract data
        user_ids = [r['user_id'] for r in effectiveness_results]
        range_enhancements = [r['range_enhancement'] for r in effectiveness_results]
        scale_utilizations = [r['scale_utilization'] for r in effectiveness_results]
        responsiveness_ratios = [r['responsiveness_ratio'] for r in effectiveness_results]
        personalization_scores = [r['personalization_effectiveness'] for r in effectiveness_results]
        
        # Range enhancement
        axes[0, 0].scatter(range(len(user_ids)), range_enhancements, alpha=0.7, s=60)
        axes[0, 0].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No Enhancement')
        axes[0, 0].axhline(y=2, color='green', linestyle='--', alpha=0.7, label='2x Enhancement')
        axes[0, 0].set_title('Range Enhancement (Adaptive vs Universal)')
        axes[0, 0].set_ylabel('Enhancement Ratio')
        axes[0, 0].set_xticks(range(len(user_ids)))
        axes[0, 0].set_xticklabels(user_ids, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Scale utilization
        axes[0, 1].scatter(range(len(user_ids)), scale_utilizations, alpha=0.7, s=60, color='orange')
        axes[0, 1].axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Full Utilization')
        axes[0, 1].set_title('Scale Utilization Effectiveness')
        axes[0, 1].set_ylabel('Utilization Ratio')
        axes[0, 1].set_xticks(range(len(user_ids)))
        axes[0, 1].set_xticklabels(user_ids, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Responsiveness comparison
        axes[1, 0].scatter(range(len(user_ids)), responsiveness_ratios, alpha=0.7, s=60, color='purple')
        axes[1, 0].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Same Responsiveness')
        axes[1, 0].set_title('Responsiveness Enhancement')
        axes[1, 0].set_ylabel('Responsiveness Ratio')
        axes[1, 0].set_xticks(range(len(user_ids)))
        axes[1, 0].set_xticklabels(user_ids, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Overall personalization effectiveness
        axes[1, 1].bar(range(len(user_ids)), personalization_scores, alpha=0.7, color='teal')
        axes[1, 1].axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Good (0.7)')
        axes[1, 1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Adequate (0.5)')
        axes[1, 1].set_title('Overall Personalization Effectiveness')
        axes[1, 1].set_ylabel('Effectiveness Score (0-1)')
        axes[1, 1].set_xticks(range(len(user_ids)))
        axes[1, 1].set_xticklabels(user_ids, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = self.analysis_dir / f'adaptive_effectiveness_{timestamp}.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"[PLOT] Adaptive effectiveness analysis saved to {fname}")
        plt.close()
    
    def _plot_session_consistency(self, consistency_results):
        """Plot session-to-session consistency analysis"""
        if not consistency_results:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Session-to-Session Consistency Analysis', fontsize=16, fontweight='bold')
        
        # Extract data
        user_ids = [r['user_id'] for r in consistency_results]
        adaptive_consistency = [r.get('adaptive_overall_consistency', 0) for r in consistency_results]
        universal_consistency = [r.get('universal_overall_consistency', 0) for r in consistency_results]
        n_sessions = [r['n_sessions'] for r in consistency_results]
        
        # Consistency comparison
        x = np.arange(len(user_ids))
        width = 0.35
        
        axes[0].bar(x - width/2, adaptive_consistency, width, label='Adaptive MI', alpha=0.7, color='blue')
        axes[0].bar(x + width/2, universal_consistency, width, label='Universal MI', alpha=0.7, color='red')
        axes[0].set_title('Consistency Scores by User')
        axes[0].set_ylabel('Consistency Score (0-1)')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(user_ids, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Number of sessions
        axes[1].bar(range(len(user_ids)), n_sessions, alpha=0.7, color='green')
        axes[1].set_title('Number of Sessions per User')
        axes[1].set_ylabel('Session Count')
        axes[1].set_xticks(range(len(user_ids)))
        axes[1].set_xticklabels(user_ids, rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = self.analysis_dir / f'session_consistency_{timestamp}.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"[PLOT] Session consistency analysis saved to {fname}")
        plt.close()
    
    def _save_calibration_quality_report(self, quality_results):
        """Save detailed calibration quality report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.analysis_dir / f'calibration_quality_report_{timestamp}.txt'
        
        with open(report_file, 'w') as f:
            f.write("DUAL CALIBRATION QUALITY ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total users analyzed: {len(quality_results)}\n\n")
            
            # Summary statistics
            quality_scores = [r['overall_quality'] for r in quality_results]
            f.write("SUMMARY STATISTICS:\n")
            f.write(f"  Mean quality score: {np.mean(quality_scores):.3f}\n")
            f.write(f"  Std quality score: {np.std(quality_scores):.3f}\n")
            f.write(f"  Min quality score: {np.min(quality_scores):.3f}\n")
            f.write(f"  Max quality score: {np.max(quality_scores):.3f}\n\n")
            
            # Individual user reports
            for i, result in enumerate(quality_results):
                f.write(f"USER {i+1}: {result['user_id']}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Overall Quality Score: {result['overall_quality']:.3f}\n")
                f.write(f"Relaxed samples: {result['relaxed_samples']}\n")
                f.write(f"Focused samples: {result['focused_samples']}\n")
                
                if 'dynamic_range' in result:
                    dr = result['dynamic_range']
                    f.write(f"MI Dynamic Range: {dr.get('mi_range', 'N/A'):.3f}\n")
                    f.write(f"Range Quality: {dr.get('range_quality', 'N/A'):.3f}\n")
                
                f.write("\nFeature Separability:\n")
                for feature, sep_data in result['feature_separability'].items():
                    f.write(f"  {feature}: Cohen's d={sep_data['cohens_d']:.3f}, ")
                    f.write(f"p={sep_data['p_value']:.3f}, ")
                    f.write(f"score={sep_data['separability_score']:.3f}\n")
                
                f.write("\n")
        
        print(f"[REPORT] Calibration quality report saved to {report_file}")
    
    def _save_adaptive_effectiveness_report(self, effectiveness_results):
        """Save adaptive effectiveness report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.analysis_dir / f'adaptive_effectiveness_report_{timestamp}.txt'
        
        # Convert to DataFrame for easy analysis
        df = pd.DataFrame(effectiveness_results)
        
        with open(report_file, 'w') as f:
            f.write("ADAPTIVE MI EFFECTIVENESS ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total sessions analyzed: {len(effectiveness_results)}\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS:\n")
            numeric_cols = ['range_enhancement', 'scale_utilization', 'responsiveness_ratio', 'personalization_effectiveness']
            for col in numeric_cols:
                if col in df.columns:
                    f.write(f"  {col}:\n")
                    f.write(f"    Mean: {df[col].mean():.3f}\n")
                    f.write(f"    Std: {df[col].std():.3f}\n")
                    f.write(f"    Min: {df[col].min():.3f}\n")
                    f.write(f"    Max: {df[col].max():.3f}\n\n")
            
            f.write("EFFECTIVENESS INTERPRETATION:\n")
            f.write("- Range Enhancement > 1.5: Excellent personalization\n")
            f.write("- Range Enhancement 1.1-1.5: Good personalization\n")
            f.write("- Range Enhancement < 1.1: Limited personalization benefit\n")
            f.write("- Personalization Score > 0.7: Highly effective\n")
            f.write("- Personalization Score 0.5-0.7: Moderately effective\n")
            f.write("- Personalization Score < 0.5: Low effectiveness\n\n")
        
        print(f"[REPORT] Adaptive effectiveness report saved to {report_file}")
    
    def _save_session_consistency_report(self, consistency_results):
        """Save session consistency report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.analysis_dir / f'session_consistency_report_{timestamp}.txt'
        
        with open(report_file, 'w') as f:
            f.write("SESSION-TO-SESSION CONSISTENCY ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Users with multiple sessions: {len(consistency_results)}\n\n")
            
            for result in consistency_results:
                f.write(f"USER: {result['user_id']}\n")
                f.write("-" * 20 + "\n")
                f.write(f"Number of sessions: {result['n_sessions']}\n")
                
                adaptive_consistency = result.get('adaptive_overall_consistency', 'N/A')
                universal_consistency = result.get('universal_overall_consistency', 'N/A')
                
                f.write(f"Adaptive MI consistency: {adaptive_consistency:.3f if adaptive_consistency != 'N/A' else 'N/A'}\n")
                f.write(f"Universal MI consistency: {universal_consistency:.3f if universal_consistency != 'N/A' else 'N/A'}\n")
                f.write("\n")
        
        print(f"[REPORT] Session consistency report saved to {report_file}")
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report covering all aspects"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE DUAL CALIBRATION ANALYSIS")
        print(f"{'='*80}")
        
        # Load all data
        self.load_all_user_data()
        
        # Run all analyses
        quality_results = self.analyze_calibration_quality()
        effectiveness_results = self.analyze_adaptive_effectiveness()
        consistency_results = self.analyze_session_consistency()
        
        # Generate summary report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = self.analysis_dir / f'comprehensive_analysis_summary_{timestamp}.txt'
        
        with open(summary_file, 'w') as f:
            f.write("COMPREHENSIVE DUAL CALIBRATION ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("ANALYSIS OVERVIEW:\n")
            f.write(f"  Users with calibration data: {len(self.calibration_data)}\n")
            f.write(f"  Users with session data: {len(self.users_data)}\n")
            f.write(f"  Calibration quality analyses: {len(quality_results) if quality_results else 0}\n")
            f.write(f"  Effectiveness analyses: {len(effectiveness_results) if effectiveness_results else 0}\n")
            f.write(f"  Consistency analyses: {len(consistency_results) if consistency_results else 0}\n\n")
            
            if quality_results:
                quality_scores = [r['overall_quality'] for r in quality_results]
                f.write("CALIBRATION QUALITY SUMMARY:\n")
                f.write(f"  Average quality score: {np.mean(quality_scores):.3f}\n")
                f.write(f"  Users with high quality (>0.7): {sum(1 for q in quality_scores if q > 0.7)}\n")
                f.write(f"  Users with adequate quality (0.5-0.7): {sum(1 for q in quality_scores if 0.5 <= q <= 0.7)}\n")
                f.write(f"  Users with low quality (<0.5): {sum(1 for q in quality_scores if q < 0.5)}\n\n")
            
            if effectiveness_results:
                eff_scores = [r['personalization_effectiveness'] for r in effectiveness_results]
                f.write("PERSONALIZATION EFFECTIVENESS SUMMARY:\n")
                f.write(f"  Average effectiveness: {np.mean(eff_scores):.3f}\n")
                f.write(f"  Highly effective sessions (>0.7): {sum(1 for e in eff_scores if e > 0.7)}\n")
                f.write(f"  Moderately effective (0.5-0.7): {sum(1 for e in eff_scores if 0.5 <= e <= 0.7)}\n")
                f.write(f"  Low effectiveness (<0.5): {sum(1 for e in eff_scores if e < 0.5)}\n\n")
            
            if consistency_results:
                adaptive_consistencies = [r.get('adaptive_overall_consistency', 0) for r in consistency_results]
                universal_consistencies = [r.get('universal_overall_consistency', 0) for r in consistency_results]
                
                f.write("CONSISTENCY SUMMARY:\n")
                f.write(f"  Average adaptive consistency: {np.mean(adaptive_consistencies):.3f}\n")
                f.write(f"  Average universal consistency: {np.mean(universal_consistencies):.3f}\n")
                
                if len(adaptive_consistencies) > 0 and len(universal_consistencies) > 0:
                    paired_comparison = [a - u for a, u in zip(adaptive_consistencies, universal_consistencies)]
                    improvement = np.mean(paired_comparison)
                    f.write(f"  Adaptive consistency advantage: {improvement:+.3f}\n")
            
            f.write("\nRECOMMendations:\n")
            f.write("- Users with low calibration quality should recalibrate\n")
            f.write("- Users with low effectiveness may benefit from longer calibration periods\n")
            f.write("- High consistency indicates successful personalization\n")
            f.write("- Compare adaptive vs universal MI to assess personalization benefit\n")
        
        print(f"[SUMMARY] Comprehensive analysis summary saved to {summary_file}")
        
        return {
            'quality_results': quality_results,
            'effectiveness_results': effectiveness_results,
            'consistency_results': consistency_results,
            'summary_file': summary_file
        }

def main():
    """Main analysis function"""
    analyzer = DualCalibrationAnalyzer()
    
    print("Dual Calibration Analysis Tool")
    print("=" * 40)
    print("1. Load all data and run comprehensive analysis")
    print("2. Analyze specific user")
    print("3. Quick quality check")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        # Comprehensive analysis
        results = analyzer.generate_comprehensive_report()
        print(f"\n✓ Comprehensive analysis complete!")
        print(f"Check the 'analysis' directory for detailed reports and visualizations.")
    
    elif choice == '2':
        # Specific user analysis
        analyzer.load_all_user_data()
        available_users = list(analyzer.calibration_data.keys())
        
        if not available_users:
            print("No user calibration data found!")
            return
        
        print(f"\nAvailable users: {', '.join(available_users)}")
        user_id = input("Enter user ID to analyze: ").strip()
        
        if user_id in available_users:
            print(f"\nAnalyzing user: {user_id}")
            analyzer.analyze_calibration_quality(user_id)
            analyzer.analyze_adaptive_effectiveness(user_id)
            
            if user_id in analyzer.users_data and len(analyzer.users_data[user_id]) > 1:
                analyzer.analyze_session_consistency(user_id)
            else:
                print(f"User {user_id} has only one session - skipping consistency analysis")
        else:
            print(f"User {user_id} not found!")
    
    elif choice == '3':
        # Quick quality check
        analyzer.load_all_user_data()
        quality_results = analyzer.analyze_calibration_quality()
        
        if quality_results:
            print(f"\nQuick Quality Summary:")
            for result in quality_results:
                user_id = result['user_id']
                quality = result['overall_quality']
                status = "HIGH" if quality > 0.7 else "MEDIUM" if quality > 0.5 else "LOW"
                print(f"  {user_id}: {quality:.3f} ({status})")
        else:
            print("No calibration data found for quality analysis!")
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
