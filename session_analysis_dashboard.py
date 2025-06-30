"""
Session Analysis Dashboard for Mindfulness Index Pipeline
========================================================

This tool analyzes all collected session data to:
1. Identify the most predictive mindfulness features
2. Detect methodological signal processing issues
3. Optimize feature extraction parameters
4. Compare user-specific vs universal patterns
5. Generate comprehensive reports and recommendations

Features:
- Cross-session feature correlation analysis
- Signal quality assessment and artifact detection
- Feature importance ranking and selection
- Calibration quality evaluation
- Methodological recommendations for signal processing
- Interactive visualizations and statistical reports
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import welch, periodogram
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import glob
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
LOG_DIR = 'logs'
USER_CONFIG_DIR = 'user_configs'
VIS_DIR = 'visualizations'
ANALYSIS_DIR = 'analysis_reports'

# Ensure analysis directory exists
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Feature names for comprehensive analysis
FEATURE_NAMES = [
    'theta_fz',      # Attention Regulation (4-8 Hz at Fz)
    'beta_fz',       # Effortful Control (13-30 Hz at Fz)
    'alpha_c3',      # Left Body Awareness (8-13 Hz at C3)
    'alpha_c4',      # Right Body Awareness (8-13 Hz at C4)
    'faa_c3c4',      # Emotion Regulation (C4-C3 alpha asymmetry)
    'alpha_pz',      # DMN Suppression (8-13 Hz at Pz)
    'alpha_po',      # Visual Detachment (8-13 Hz at PO7/PO8)
    'alpha_oz',      # Relaxation (8-13 Hz at Oz)
    'eda_norm'       # Arousal/Stress (normalized EDA)
]

MI_TYPES = ['adaptive_mi', 'universal_mi', 'emi', 'raw_mi']

class SessionAnalyzer:
    """Comprehensive analysis of all mindfulness sessions"""
    
    def __init__(self):
        self.session_data = {}
        self.calibration_data = {}
        self.feature_stats = {}
        self.analysis_results = {}
        
    def load_all_session_data(self):
        """Load all available session and calibration data"""
        print("Loading session data...")
        
        # Load session CSV files
        session_files = glob.glob(os.path.join(LOG_DIR, '*_session_*.csv'))
        print(f"Found {len(session_files)} session files")
        
        for file_path in session_files:
            try:
                df = pd.read_csv(file_path)
                
                # Extract user ID and timestamp from filename
                filename = os.path.basename(file_path)
                parts = filename.split('_')
                if len(parts) >= 4:
                    user_id = parts[0]
                    timestamp = '_'.join(parts[-2:]).replace('.csv', '')
                    
                    session_key = f"{user_id}_{timestamp}"
                    self.session_data[session_key] = {
                        'user_id': user_id,
                        'timestamp': timestamp,
                        'data': df,
                        'file_path': file_path
                    }
                    
                    print(f"  Loaded: {session_key} ({len(df)} samples)")
                    
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
        
        # Load calibration data
        calibration_files = glob.glob(os.path.join(USER_CONFIG_DIR, '*_dual_calibration.json'))
        print(f"\nFound {len(calibration_files)} calibration files")
        
        for file_path in calibration_files:
            try:
                with open(file_path, 'r') as f:
                    config = json.load(f)
                
                user_id = config['user_id']
                self.calibration_data[user_id] = config
                print(f"  Loaded calibration: {user_id}")
                
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
        
        # Load baseline feature data
        baseline_files = glob.glob(os.path.join(USER_CONFIG_DIR, '*_dual_baseline.csv'))
        print(f"\nFound {len(baseline_files)} baseline files")
        
        for file_path in baseline_files:
            try:
                df = pd.read_csv(file_path)
                filename = os.path.basename(file_path)
                user_id = filename.split('_')[0]
                
                if user_id in self.calibration_data:
                    self.calibration_data[user_id]['baseline_features'] = df
                    print(f"  Loaded baseline features: {user_id} ({len(df)} samples)")
                    
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
        
        print(f"\nData loading complete:")
        print(f"  Sessions: {len(self.session_data)}")
        print(f"  Users with calibration: {len(self.calibration_data)}")
    
    def analyze_feature_importance(self):
        """Analyze which features are most predictive of mindfulness states"""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Combine all session data
        all_sessions = []
        for session_key, session_info in self.session_data.items():
            df = session_info['data'].copy()
            df['user_id'] = session_info['user_id']
            df['session'] = session_key
            all_sessions.append(df)
        
        if not all_sessions:
            print("No session data available for analysis")
            return
        
        combined_df = pd.concat(all_sessions, ignore_index=True)
        print(f"Combined dataset: {len(combined_df)} samples from {len(all_sessions)} sessions")
        
        # Feature importance analysis for each MI type
        importance_results = {}
        
        for mi_type in MI_TYPES:
            if mi_type not in combined_df.columns:
                continue
                
            print(f"\nAnalyzing feature importance for {mi_type}...")
            
            # Prepare feature matrix
            X = combined_df[FEATURE_NAMES].values
            y = combined_df[mi_type].values
            
            # Remove NaN values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            if len(X_clean) < 10:
                print(f"  Insufficient valid data for {mi_type}")
                continue
            
            # Random Forest feature importance
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_clean, y_clean)
            rf_importance = rf.feature_importances_
            
            # Mutual information
            mi_scores = mutual_info_regression(X_clean, y_clean, random_state=42)
            
            # F-test scores
            f_scores, f_pvalues = f_regression(X_clean, y_clean)
            
            # Correlation analysis
            correlations = []
            for i, feature in enumerate(FEATURE_NAMES):
                corr, p_val = stats.pearsonr(X_clean[:, i], y_clean)
                correlations.append((corr, p_val))
            
            importance_results[mi_type] = {
                'rf_importance': rf_importance,
                'mutual_info': mi_scores,
                'f_scores': f_scores,
                'f_pvalues': f_pvalues,
                'correlations': correlations,
                'n_samples': len(X_clean)
            }
            
            # Print top features
            feature_ranking = list(zip(FEATURE_NAMES, rf_importance))
            feature_ranking.sort(key=lambda x: x[1], reverse=True)
            
            print(f"  Top 5 features for {mi_type}:")
            for i, (feature, importance) in enumerate(feature_ranking[:5]):
                corr, p_val = correlations[FEATURE_NAMES.index(feature)]
                print(f"    {i+1}. {feature}: RF={importance:.3f}, r={corr:.3f} (p={p_val:.3e})")
        
        self.analysis_results['feature_importance'] = importance_results
        return importance_results
    
    def analyze_signal_quality(self):
        """Analyze signal quality and identify processing issues"""
        print("\n" + "="*60)
        print("SIGNAL QUALITY ANALYSIS")
        print("="*60)
        
        quality_issues = {}
        
        for session_key, session_info in self.session_data.items():
            df = session_info['data']
            user_id = session_info['user_id']
            
            print(f"\nAnalyzing {session_key}...")
            
            issues = {
                'outliers': {},
                'artifacts': {},
                'stability': {},
                'calibration_quality': None
            }
            
            # Outlier detection for each feature
            for feature in FEATURE_NAMES:
                if feature in df.columns:
                    values = df[feature].dropna()
                    if len(values) > 10:
                        # IQR method
                        q1, q3 = np.percentile(values, [25, 75])
                        iqr = q3 - q1
                        outlier_mask = (values < (q1 - 1.5 * iqr)) | (values > (q3 + 1.5 * iqr))
                        outlier_pct = np.sum(outlier_mask) / len(values) * 100
                        
                        # Z-score method
                        z_scores = np.abs(stats.zscore(values))
                        extreme_outliers = np.sum(z_scores > 3) / len(values) * 100
                        
                        issues['outliers'][feature] = {
                            'iqr_outliers_pct': outlier_pct,
                            'extreme_outliers_pct': extreme_outliers,
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'range': [np.min(values), np.max(values)]
                        }
                        
                        if outlier_pct > 10:
                            print(f"  WARNING: {feature} has {outlier_pct:.1f}% outliers")
            
            # Artifact detection using isolation forest
            feature_matrix = df[FEATURE_NAMES].dropna()
            if len(feature_matrix) > 20:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                artifact_labels = iso_forest.fit_predict(feature_matrix)
                artifact_pct = np.sum(artifact_labels == -1) / len(artifact_labels) * 100
                
                issues['artifacts']['isolation_forest_pct'] = artifact_pct
                if artifact_pct > 20:
                    print(f"  WARNING: {artifact_pct:.1f}% samples flagged as artifacts")
            
            # Signal stability analysis
            for mi_type in MI_TYPES:
                if mi_type in df.columns:
                    mi_values = df[mi_type].dropna()
                    if len(mi_values) > 10:
                        # Calculate stability metrics
                        stability_cv = np.std(mi_values) / np.mean(mi_values) if np.mean(mi_values) != 0 else float('inf')
                        
                        # Trend analysis
                        x = np.arange(len(mi_values))
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, mi_values)
                        
                        issues['stability'][mi_type] = {
                            'coefficient_of_variation': stability_cv,
                            'trend_slope': slope,
                            'trend_r_squared': r_value**2,
                            'trend_p_value': p_value
                        }
                        
                        if stability_cv > 1.0:
                            print(f"  WARNING: {mi_type} shows high variability (CV={stability_cv:.2f})")
            
            # Calibration quality assessment
            if user_id in self.calibration_data:
                cal_data = self.calibration_data[user_id]
                if 'adaptive_thresholds' in cal_data:
                    mapping = cal_data['adaptive_thresholds']['adaptive_mapping']
                    dynamic_range = mapping['dynamic_range']
                    
                    issues['calibration_quality'] = {
                        'dynamic_range': dynamic_range,
                        'quality': 'excellent' if dynamic_range > 0.3 else 'good' if dynamic_range > 0.15 else 'poor'
                    }
                    
                    if dynamic_range < 0.15:
                        print(f"  WARNING: Poor calibration quality (range={dynamic_range:.3f})")
            
            quality_issues[session_key] = issues
        
        self.analysis_results['signal_quality'] = quality_issues
        return quality_issues
    
    def analyze_user_patterns(self):
        """Analyze user-specific patterns and individual differences"""
        print("\n" + "="*60)
        print("USER PATTERN ANALYSIS")
        print("="*60)
        
        user_patterns = {}
        
        # Group sessions by user
        user_sessions = {}
        for session_key, session_info in self.session_data.items():
            user_id = session_info['user_id']
            if user_id not in user_sessions:
                user_sessions[user_id] = []
            user_sessions[user_id].append(session_info)
        
        print(f"Analyzing patterns for {len(user_sessions)} users")
        
        for user_id, sessions in user_sessions.items():
            print(f"\nUser {user_id}: {len(sessions)} sessions")
            
            # Combine all sessions for this user
            user_data = []
            for session in sessions:
                df = session['data'].copy()
                df['session_index'] = len(user_data)
                user_data.append(df)
            
            if not user_data:
                continue
                
            combined_user_df = pd.concat(user_data, ignore_index=True)
            
            patterns = {
                'session_count': len(sessions),
                'total_samples': len(combined_user_df),
                'feature_profiles': {},
                'mi_characteristics': {},
                'consistency': {},
                'learning_trends': {}
            }
            
            # Feature profiles
            for feature in FEATURE_NAMES:
                if feature in combined_user_df.columns:
                    values = combined_user_df[feature].dropna()
                    if len(values) > 0:
                        patterns['feature_profiles'][feature] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'median': np.median(values),
                            'q25': np.percentile(values, 25),
                            'q75': np.percentile(values, 75)
                        }
            
            # MI characteristics
            for mi_type in MI_TYPES:
                if mi_type in combined_user_df.columns:
                    values = combined_user_df[mi_type].dropna()
                    if len(values) > 0:
                        patterns['mi_characteristics'][mi_type] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'range': [np.min(values), np.max(values)],
                            'dynamic_range': np.max(values) - np.min(values)
                        }
            
            # Session-to-session consistency
            if len(sessions) > 1:
                session_means = {}
                for mi_type in MI_TYPES:
                    session_means[mi_type] = []
                    for session in sessions:
                        if mi_type in session['data'].columns:
                            mean_val = session['data'][mi_type].dropna().mean()
                            if not np.isnan(mean_val):
                                session_means[mi_type].append(mean_val)
                
                for mi_type, means in session_means.items():
                    if len(means) > 1:
                        consistency_cv = np.std(means) / np.mean(means) if np.mean(means) != 0 else float('inf')
                        patterns['consistency'][mi_type] = {
                            'coefficient_of_variation': consistency_cv,
                            'session_means': means
                        }
                        
                        print(f"  {mi_type} consistency (CV): {consistency_cv:.3f}")
            
            # Learning trends (if multiple sessions)
            if len(sessions) > 2:
                for mi_type in MI_TYPES:
                    if mi_type in session_means and len(session_means[mi_type]) > 2:
                        x = np.arange(len(session_means[mi_type]))
                        y = session_means[mi_type]
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                        
                        patterns['learning_trends'][mi_type] = {
                            'slope': slope,
                            'r_squared': r_value**2,
                            'p_value': p_value,
                            'trend': 'improving' if slope > 0 and p_value < 0.05 else 'stable' if abs(slope) < 0.01 else 'declining'
                        }
                        
                        if p_value < 0.05:
                            trend = 'improving' if slope > 0 else 'declining'
                            print(f"  {mi_type} shows {trend} trend (slope={slope:.4f}, p={p_value:.3f})")
            
            user_patterns[user_id] = patterns
        
        self.analysis_results['user_patterns'] = user_patterns
        return user_patterns
    
    def detect_methodological_issues(self):
        """Detect systematic methodological issues in signal processing"""
        print("\n" + "="*60)
        print("METHODOLOGICAL ISSUE DETECTION")
        print("="*60)
        
        issues = {
            'systematic_biases': {},
            'processing_artifacts': {},
            'calibration_issues': {},
            'feature_problems': {},
            'recommendations': []
        }
        
        # Analyze systematic biases across all sessions
        all_data = []
        for session_info in self.session_data.values():
            df = session_info['data'].copy()
            df['user_id'] = session_info['user_id']
            all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Check for systematic feature biases
            print("Checking for systematic biases...")
            
            for feature in FEATURE_NAMES:
                if feature in combined_df.columns:
                    values = combined_df[feature].dropna()
                    if len(values) > 100:
                        # Check for extreme skewness
                        skewness = stats.skew(values)
                        kurtosis = stats.kurtosis(values)
                        
                        if abs(skewness) > 2:
                            issues['systematic_biases'][feature] = {
                                'issue': 'extreme_skewness',
                                'skewness': skewness,
                                'severity': 'high' if abs(skewness) > 5 else 'moderate'
                            }
                            print(f"  WARNING: {feature} shows extreme skewness ({skewness:.2f})")
                        
                        if kurtosis > 10:
                            if feature not in issues['systematic_biases']:
                                issues['systematic_biases'][feature] = {}
                            issues['systematic_biases'][feature]['extreme_kurtosis'] = kurtosis
                            print(f"  WARNING: {feature} shows extreme kurtosis ({kurtosis:.2f})")
            
            # Check for processing artifacts
            print("\nChecking for processing artifacts...")
            
            # Detect identical consecutive values (potential processing errors)
            for feature in FEATURE_NAMES:
                if feature in combined_df.columns:
                    values = combined_df[feature].dropna()
                    if len(values) > 10:
                        # Count consecutive identical values
                        consecutive_identical = 0
                        max_consecutive = 0
                        for i in range(1, len(values)):
                            if values.iloc[i] == values.iloc[i-1]:
                                consecutive_identical += 1
                                max_consecutive = max(max_consecutive, consecutive_identical)
                            else:
                                consecutive_identical = 0
                        
                        if max_consecutive > 5:
                            issues['processing_artifacts'][feature] = {
                                'max_consecutive_identical': max_consecutive,
                                'severity': 'high' if max_consecutive > 20 else 'moderate'
                            }
                            print(f"  WARNING: {feature} has {max_consecutive} consecutive identical values")
            
            # Check MI relationships
            print("\nChecking MI relationships...")
            
            # Check if different MI types are properly correlated
            mi_correlations = {}
            for i, mi1 in enumerate(MI_TYPES):
                for j, mi2 in enumerate(MI_TYPES[i+1:], i+1):
                    if mi1 in combined_df.columns and mi2 in combined_df.columns:
                        valid_data = combined_df[[mi1, mi2]].dropna()
                        if len(valid_data) > 10:
                            corr, p_val = stats.pearsonr(valid_data[mi1], valid_data[mi2])
                            mi_correlations[f"{mi1}_vs_{mi2}"] = {
                                'correlation': corr,
                                'p_value': p_val,
                                'n_samples': len(valid_data)
                            }
                            
                            if mi1 != 'raw_mi' and mi2 != 'raw_mi' and corr < 0.3:
                                print(f"  WARNING: Low correlation between {mi1} and {mi2} (r={corr:.3f})")
            
            issues['mi_correlations'] = mi_correlations
        
        # Analyze calibration quality across users
        print("\nAnalyzing calibration quality...")
        
        calibration_stats = []
        for user_id, cal_data in self.calibration_data.items():
            if 'adaptive_thresholds' in cal_data:
                mapping = cal_data['adaptive_thresholds']['adaptive_mapping']
                calibration_stats.append(mapping['dynamic_range'])
        
        if calibration_stats:
            mean_range = np.mean(calibration_stats)
            poor_calibrations = sum(1 for r in calibration_stats if r < 0.15)
            
            issues['calibration_issues'] = {
                'mean_dynamic_range': mean_range,
                'poor_calibrations_count': poor_calibrations,
                'total_calibrations': len(calibration_stats),
                'poor_calibration_rate': poor_calibrations / len(calibration_stats)
            }
            
            if poor_calibrations / len(calibration_stats) > 0.3:
                print(f"  WARNING: {poor_calibrations}/{len(calibration_stats)} calibrations are poor quality")
        
        # Generate recommendations
        print("\nGenerating recommendations...")
        
        recommendations = []
        
        # Feature-specific recommendations
        if issues['systematic_biases']:
            recommendations.append("FEATURE PROCESSING: Apply robust normalization for features with extreme skewness")
            recommendations.append("OUTLIER HANDLING: Implement adaptive outlier detection and replacement")
        
        if issues['processing_artifacts']:
            recommendations.append("ARTIFACT DETECTION: Enhance median filtering and implement change-point detection")
            recommendations.append("SIGNAL VALIDATION: Add real-time signal quality monitoring")
        
        if issues['calibration_issues'] and issues['calibration_issues']['poor_calibration_rate'] > 0.3:
            recommendations.append("CALIBRATION: Extend calibration duration or improve instructions")
            recommendations.append("THRESHOLD ADAPTATION: Implement progressive threshold learning")
        
        # General recommendations
        recommendations.extend([
            "FEATURE SELECTION: Focus on top 5 most predictive features identified in analysis",
            "TEMPORAL SMOOTHING: Adjust smoothing parameters based on signal stability analysis",
            "USER ADAPTATION: Implement user-specific feature weighting based on individual patterns",
            "VALIDATION: Add cross-validation metrics for real-time quality assessment"
        ])
        
        issues['recommendations'] = recommendations
        
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        self.analysis_results['methodological_issues'] = issues
        return issues
    
    def generate_optimization_recommendations(self):
        """Generate specific recommendations for system optimization"""
        print("\n" + "="*60)
        print("OPTIMIZATION RECOMMENDATIONS")
        print("="*60)
        
        recommendations = {
            'feature_optimization': {},
            'processing_optimization': {},
            'calibration_optimization': {},
            'user_experience_optimization': {}
        }
        
        # Feature optimization based on importance analysis
        if 'feature_importance' in self.analysis_results:
            importance_data = self.analysis_results['feature_importance']
            
            # Identify consistently important features across MI types
            feature_scores = {feature: [] for feature in FEATURE_NAMES}
            
            for mi_type, results in importance_data.items():
                rf_importance = results['rf_importance']
                for i, feature in enumerate(FEATURE_NAMES):
                    feature_scores[feature].append(rf_importance[i])
            
            # Calculate average importance
            avg_importance = {feature: np.mean(scores) for feature, scores in feature_scores.items()}
            top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            
            recommendations['feature_optimization'] = {
                'top_features': top_features[:5],
                'recommended_feature_set': [feat[0] for feat in top_features[:5]],
                'importance_scores': avg_importance
            }
            
            print("Top 5 recommended features:")
            for i, (feature, score) in enumerate(top_features[:5], 1):
                print(f"  {i}. {feature}: {score:.3f}")
        
        # Processing optimization
        if 'signal_quality' in self.analysis_results:
            quality_data = self.analysis_results['signal_quality']
            
            # Analyze common quality issues
            high_outlier_features = []
            high_artifact_sessions = []
            
            for session_key, issues in quality_data.items():
                for feature, outlier_info in issues['outliers'].items():
                    if outlier_info['iqr_outliers_pct'] > 15:
                        high_outlier_features.append(feature)
                
                if 'isolation_forest_pct' in issues['artifacts'] and issues['artifacts']['isolation_forest_pct'] > 25:
                    high_artifact_sessions.append(session_key)
            
            recommendations['processing_optimization'] = {
                'problematic_features': list(set(high_outlier_features)),
                'high_artifact_sessions': high_artifact_sessions,
                'suggested_filters': self._suggest_processing_filters(high_outlier_features)
            }
        
        # Calibration optimization
        if 'user_patterns' in self.analysis_results:
            patterns_data = self.analysis_results['user_patterns']
            
            consistency_scores = []
            for user_id, patterns in patterns_data.items():
                if 'consistency' in patterns:
                    for mi_type, consistency in patterns['consistency'].items():
                        consistency_scores.append(consistency['coefficient_of_variation'])
            
            if consistency_scores:
                mean_consistency = np.mean(consistency_scores)
                recommendations['calibration_optimization'] = {
                    'mean_consistency_cv': mean_consistency,
                    'calibration_quality': 'good' if mean_consistency < 0.5 else 'needs_improvement',
                    'suggested_improvements': self._suggest_calibration_improvements(mean_consistency)
                }
        
        # User experience optimization
        recommendations['user_experience_optimization'] = {
            'session_length': self._recommend_session_length(),
            'feedback_frequency': 'Every 5 seconds for responsive feedback',
            'calibration_duration': '45 seconds per phase for better stability',
            'user_instructions': 'Add real-time signal quality indicators'
        }
        
        self.analysis_results['optimization_recommendations'] = recommendations
        return recommendations
    
    def _suggest_processing_filters(self, problematic_features):
        """Suggest specific processing improvements for problematic features"""
        suggestions = []
        
        for feature in set(problematic_features):
            if 'eda' in feature:
                suggestions.append(f"{feature}: Apply stronger low-pass filter (0.5 Hz cutoff)")
            elif 'alpha' in feature:
                suggestions.append(f"{feature}: Increase median filter window size to 7-9 samples")
            elif 'theta' in feature or 'beta' in feature:
                suggestions.append(f"{feature}: Apply robust z-score normalization with MAD")
            elif 'faa' in feature:
                suggestions.append(f"{feature}: Use log-ratio with baseline correction")
        
        return suggestions
    
    def _suggest_calibration_improvements(self, consistency_cv):
        """Suggest calibration improvements based on consistency analysis"""
        if consistency_cv > 1.0:
            return [
                "Extend calibration phases to 45-60 seconds each",
                "Add practice session before actual calibration",
                "Implement adaptive calibration validation",
                "Use median-based threshold calculation instead of mean"
            ]
        elif consistency_cv > 0.5:
            return [
                "Add brief practice before focused calibration",
                "Implement quality check during calibration",
                "Consider user-specific calibration duration"
            ]
        else:
            return ["Current calibration approach is working well"]
    
    def _recommend_session_length(self):
        """Recommend optimal session length based on data analysis"""
        if hasattr(self, 'session_data'):
            session_lengths = []
            for session_info in self.session_data.values():
                df = session_info['data']
                if 'timestamp' in df.columns and len(df) > 10:
                    duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
                    session_lengths.append(duration)
            
            if session_lengths:
                median_length = np.median(session_lengths)
                if median_length < 300:  # 5 minutes
                    return "5-10 minutes for training, 15-20 minutes for research"
                else:
                    return "10-15 minutes optimal based on current data"
        
        return "10-15 minutes recommended"
    
    def create_comprehensive_report(self):
        """Generate a comprehensive analysis report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(ANALYSIS_DIR, f'comprehensive_analysis_report_{timestamp}.md')
        
        with open(report_path, 'w') as f:
            f.write("# Mindfulness Index Pipeline - Comprehensive Analysis Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Data summary
            f.write("## Data Summary\n\n")
            f.write(f"- Total sessions analyzed: {len(self.session_data)}\n")
            f.write(f"- Users with calibration data: {len(self.calibration_data)}\n")
            f.write(f"- Unique users: {len(set(info['user_id'] for info in self.session_data.values()))}\n\n")
            
            # Feature importance
            if 'feature_importance' in self.analysis_results:
                f.write("## Feature Importance Analysis\n\n")
                importance_data = self.analysis_results['feature_importance']
                
                for mi_type, results in importance_data.items():
                    f.write(f"### {mi_type.upper()}\n\n")
                    f.write("| Feature | RF Importance | Correlation | P-value |\n")
                    f.write("|---------|---------------|-------------|----------|\n")
                    
                    rf_importance = results['rf_importance']
                    correlations = results['correlations']
                    
                    feature_data = []
                    for i, feature in enumerate(FEATURE_NAMES):
                        corr, p_val = correlations[i]
                        feature_data.append((feature, rf_importance[i], corr, p_val))
                    
                    feature_data.sort(key=lambda x: x[1], reverse=True)
                    
                    for feature, importance, corr, p_val in feature_data:
                        f.write(f"| {feature} | {importance:.3f} | {corr:.3f} | {p_val:.3e} |\n")
                    f.write("\n")
            
            # Signal quality issues
            if 'signal_quality' in self.analysis_results:
                f.write("## Signal Quality Assessment\n\n")
                quality_data = self.analysis_results['signal_quality']
                
                f.write("### Sessions with Quality Issues\n\n")
                for session_key, issues in quality_data.items():
                    has_issues = False
                    issue_list = []
                    
                    # Check for outlier issues
                    for feature, outlier_info in issues['outliers'].items():
                        if outlier_info['iqr_outliers_pct'] > 10:
                            has_issues = True
                            issue_list.append(f"High outliers in {feature} ({outlier_info['iqr_outliers_pct']:.1f}%)")
                    
                    # Check for artifact issues
                    if 'isolation_forest_pct' in issues['artifacts'] and issues['artifacts']['isolation_forest_pct'] > 20:
                        has_issues = True
                        issue_list.append(f"High artifact rate ({issues['artifacts']['isolation_forest_pct']:.1f}%)")
                    
                    # Check calibration quality
                    if issues['calibration_quality'] and issues['calibration_quality']['quality'] == 'poor':
                        has_issues = True
                        issue_list.append(f"Poor calibration quality (range={issues['calibration_quality']['dynamic_range']:.3f})")
                    
                    if has_issues:
                        f.write(f"**{session_key}:**\n")
                        for issue in issue_list:
                            f.write(f"- {issue}\n")
                        f.write("\n")
            
            # Optimization recommendations
            if 'optimization_recommendations' in self.analysis_results:
                f.write("## Optimization Recommendations\n\n")
                opt_data = self.analysis_results['optimization_recommendations']
                
                if 'feature_optimization' in opt_data:
                    f.write("### Recommended Feature Set\n\n")
                    top_features = opt_data['feature_optimization']['recommended_feature_set']
                    for i, feature in enumerate(top_features, 1):
                        f.write(f"{i}. **{feature}**\n")
                    f.write("\n")
                
                if 'processing_optimization' in opt_data:
                    f.write("### Processing Improvements\n\n")
                    proc_data = opt_data['processing_optimization']
                    if 'suggested_filters' in proc_data:
                        for suggestion in proc_data['suggested_filters']:
                            f.write(f"- {suggestion}\n")
                    f.write("\n")
            
            # Methodological recommendations
            if 'methodological_issues' in self.analysis_results:
                f.write("## Methodological Recommendations\n\n")
                method_data = self.analysis_results['methodological_issues']
                
                if 'recommendations' in method_data:
                    for i, rec in enumerate(method_data['recommendations'], 1):
                        f.write(f"{i}. {rec}\n")
                    f.write("\n")
        
        print(f"Comprehensive report saved to: {report_path}")
        return report_path
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Feature importance heatmap
        if 'feature_importance' in self.analysis_results:
            self._plot_feature_importance_heatmap(timestamp)
        
        # Signal quality dashboard
        if 'signal_quality' in self.analysis_results:
            self._plot_signal_quality_dashboard(timestamp)
        
        # User pattern analysis
        if 'user_patterns' in self.analysis_results:
            self._plot_user_patterns(timestamp)
    
    def _plot_feature_importance_heatmap(self, timestamp):
        """Create feature importance heatmap"""
        importance_data = self.analysis_results['feature_importance']
        
        # Create importance matrix
        mi_types = list(importance_data.keys())
        importance_matrix = np.zeros((len(FEATURE_NAMES), len(mi_types)))
        
        for j, mi_type in enumerate(mi_types):
            rf_importance = importance_data[mi_type]['rf_importance']
            importance_matrix[:, j] = rf_importance
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(importance_matrix, 
                   xticklabels=mi_types,
                   yticklabels=FEATURE_NAMES,
                   annot=True, 
                   fmt='.3f',
                   cmap='viridis',
                   cbar_kws={'label': 'Random Forest Importance'})
        
        plt.title('Feature Importance Across MI Types')
        plt.xlabel('MI Type')
        plt.ylabel('Features')
        plt.tight_layout()
        
        plot_path = os.path.join(ANALYSIS_DIR, f'feature_importance_heatmap_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance heatmap saved to: {plot_path}")
    
    def _plot_signal_quality_dashboard(self, timestamp):
        """Create signal quality dashboard"""
        quality_data = self.analysis_results['signal_quality']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Outlier percentages by feature
        outlier_data = {}
        for session_key, issues in quality_data.items():
            for feature, outlier_info in issues['outliers'].items():
                if feature not in outlier_data:
                    outlier_data[feature] = []
                outlier_data[feature].append(outlier_info['iqr_outliers_pct'])
        
        if outlier_data:
            features = list(outlier_data.keys())
            outlier_means = [np.mean(outlier_data[f]) for f in features]
            
            axes[0, 0].bar(range(len(features)), outlier_means)
            axes[0, 0].set_xticks(range(len(features)))
            axes[0, 0].set_xticklabels(features, rotation=45)
            axes[0, 0].set_ylabel('Mean Outlier %')
            axes[0, 0].set_title('Average Outlier Percentage by Feature')
            axes[0, 0].axhline(y=10, color='r', linestyle='--', alpha=0.7, label='Warning Threshold')
            axes[0, 0].legend()
        
        # Artifact rates by session
        session_names = []
        artifact_rates = []
        for session_key, issues in quality_data.items():
            if 'isolation_forest_pct' in issues['artifacts']:
                session_names.append(session_key.split('_')[-1][:8])  # Shortened name
                artifact_rates.append(issues['artifacts']['isolation_forest_pct'])
        
        if artifact_rates:
            axes[0, 1].bar(range(len(session_names)), artifact_rates)
            axes[0, 1].set_xticks(range(len(session_names)))
            axes[0, 1].set_xticklabels(session_names, rotation=45)
            axes[0, 1].set_ylabel('Artifact Rate %')
            axes[0, 1].set_title('Artifact Detection by Session')
            axes[0, 1].axhline(y=20, color='r', linestyle='--', alpha=0.7, label='Warning Threshold')
            axes[0, 1].legend()
        
        # Calibration quality distribution
        cal_qualities = []
        for session_key, issues in quality_data.items():
            if issues['calibration_quality']:
                cal_qualities.append(issues['calibration_quality']['dynamic_range'])
        
        if cal_qualities:
            axes[1, 0].hist(cal_qualities, bins=10, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(x=0.15, color='r', linestyle='--', alpha=0.7, label='Poor Quality Threshold')
            axes[1, 0].axvline(x=0.3, color='g', linestyle='--', alpha=0.7, label='Good Quality Threshold')
            axes[1, 0].set_xlabel('Dynamic Range')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Calibration Quality Distribution')
            axes[1, 0].legend()
        
        # Feature stability (coefficient of variation)
        stability_data = {}
        for session_key, issues in quality_data.items():
            for mi_type, stability in issues['stability'].items():
                if mi_type not in stability_data:
                    stability_data[mi_type] = []
                stability_data[mi_type].append(stability['coefficient_of_variation'])
        
        if stability_data:
            mi_types = list(stability_data.keys())
            stability_means = [np.mean(stability_data[mi]) for mi in mi_types]
            
            axes[1, 1].bar(range(len(mi_types)), stability_means)
            axes[1, 1].set_xticks(range(len(mi_types)))
            axes[1, 1].set_xticklabels(mi_types, rotation=45)
            axes[1, 1].set_ylabel('Mean CV')
            axes[1, 1].set_title('MI Stability by Type')
            axes[1, 1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Moderate Stability')
            axes[1, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Poor Stability')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        plot_path = os.path.join(ANALYSIS_DIR, f'signal_quality_dashboard_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Signal quality dashboard saved to: {plot_path}")
    
    def _plot_user_patterns(self, timestamp):
        """Create user pattern analysis plots"""
        patterns_data = self.analysis_results['user_patterns']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # User feature profiles
        user_ids = list(patterns_data.keys())
        if len(user_ids) > 0 and 'feature_profiles' in patterns_data[user_ids[0]]:
            feature_means = {feature: [] for feature in FEATURE_NAMES}
            
            for user_id, patterns in patterns_data.items():
                for feature in FEATURE_NAMES:
                    if feature in patterns['feature_profiles']:
                        feature_means[feature].append(patterns['feature_profiles'][feature]['mean'])
                    else:
                        feature_means[feature].append(0)
            
            # Create feature profile heatmap
            user_feature_matrix = np.array([feature_means[f] for f in FEATURE_NAMES])
            
            im = axes[0, 0].imshow(user_feature_matrix, aspect='auto', cmap='viridis')
            axes[0, 0].set_xticks(range(len(user_ids)))
            axes[0, 0].set_xticklabels(user_ids, rotation=45)
            axes[0, 0].set_yticks(range(len(FEATURE_NAMES)))
            axes[0, 0].set_yticklabels(FEATURE_NAMES)
            axes[0, 0].set_title('User Feature Profiles')
            plt.colorbar(im, ax=axes[0, 0])
        
        # Session consistency
        consistency_data = {}
        for user_id, patterns in patterns_data.items():
            if 'consistency' in patterns:
                for mi_type, consistency in patterns['consistency'].items():
                    if mi_type not in consistency_data:
                        consistency_data[mi_type] = []
                    consistency_data[mi_type].append(consistency['coefficient_of_variation'])
        
        if consistency_data:
            mi_types = list(consistency_data.keys())
            boxplot_data = [consistency_data[mi] for mi in mi_types]
            
            axes[0, 1].boxplot(boxplot_data, labels=mi_types)
            axes[0, 1].set_ylabel('Coefficient of Variation')
            axes[0, 1].set_title('Session-to-Session Consistency by MI Type')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Learning trends
        learning_data = {'improving': 0, 'stable': 0, 'declining': 0}
        for user_id, patterns in patterns_data.items():
            if 'learning_trends' in patterns:
                for mi_type, trend_info in patterns['learning_trends'].items():
                    trend = trend_info['trend']
                    if trend in learning_data:
                        learning_data[trend] += 1
        
        if sum(learning_data.values()) > 0:
            axes[1, 0].pie(learning_data.values(), 
                          labels=learning_data.keys(), 
                          autopct='%1.1f%%',
                          startangle=90)
            axes[1, 0].set_title('Learning Trends Distribution')
        
        # MI characteristics by user
        mi_ranges = {}
        for user_id, patterns in patterns_data.items():
            if 'mi_characteristics' in patterns:
                for mi_type, characteristics in patterns['mi_characteristics'].items():
                    if mi_type not in mi_ranges:
                        mi_ranges[mi_type] = []
                    mi_ranges[mi_type].append(characteristics['dynamic_range'])
        
        if mi_ranges:
            mi_types = list(mi_ranges.keys())
            range_data = [mi_ranges[mi] for mi in mi_types]
            
            axes[1, 1].boxplot(range_data, labels=mi_types)
            axes[1, 1].set_ylabel('Dynamic Range')
            axes[1, 1].set_title('MI Dynamic Range by Type')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        plot_path = os.path.join(ANALYSIS_DIR, f'user_patterns_analysis_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"User patterns analysis saved to: {plot_path}")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("="*80)
        print("COMPREHENSIVE MINDFULNESS INDEX ANALYSIS")
        print("="*80)
        
        # Load all data
        self.load_all_session_data()
        
        if not self.session_data:
            print("No session data found. Please ensure session files exist in the logs directory.")
            return
        
        # Run all analyses
        print("\n" + "="*60)
        print("RUNNING COMPREHENSIVE ANALYSIS")
        print("="*60)
        
        self.analyze_feature_importance()
        self.analyze_signal_quality()
        self.analyze_user_patterns()
        self.detect_methodological_issues()
        self.generate_optimization_recommendations()
        
        # Generate outputs
        print("\n" + "="*60)
        print("GENERATING REPORTS AND VISUALIZATIONS")
        print("="*60)
        
        report_path = self.create_comprehensive_report()
        self.create_visualizations()
        
        # Summary
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"📊 Sessions analyzed: {len(self.session_data)}")
        print(f"👥 Unique users: {len(set(info['user_id'] for info in self.session_data.values()))}")
        print(f"📋 Comprehensive report: {report_path}")
        print(f"📈 Visualizations saved to: {ANALYSIS_DIR}")
        
        # Display key findings
        if 'optimization_recommendations' in self.analysis_results:
            opt_data = self.analysis_results['optimization_recommendations']
            if 'feature_optimization' in opt_data:
                print(f"\n🎯 Top recommended features:")
                top_features = opt_data['feature_optimization']['recommended_feature_set']
                for i, feature in enumerate(top_features, 1):
                    print(f"   {i}. {feature}")
        
        print(f"\n✅ Analysis complete! Check {ANALYSIS_DIR} for detailed results.")

def main():
    """Main function for session analysis"""
    analyzer = SessionAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
