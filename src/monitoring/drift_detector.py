import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import cv2

logger = logging.getLogger(__name__)

class ModelDriftDetector:
    """Detects model drift using statistical methods and performance metrics."""
    
    def __init__(self, 
                 reference_file: str = "reference_stats.pkl",
                 drift_threshold: float = 0.05,
                 performance_threshold: float = 0.1):
        self.reference_file = Path(reference_file)
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        self.reference_stats = self._load_reference_stats()
        
    def _load_reference_stats(self) -> Optional[Dict]:
        """Load reference statistics from file."""
        if self.reference_file.exists():
            try:
                with open(self.reference_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Could not load reference stats: {e}")
        return None
    
    def _save_reference_stats(self, stats: Dict) -> None:
        """Save reference statistics to file."""
        try:
            with open(self.reference_file, 'wb') as f:
                pickle.dump(stats, f)
        except Exception as e:
            logger.error(f"Could not save reference stats: {e}")
    
    def calculate_image_statistics(self, images: List[np.ndarray]) -> Dict:
        """Calculate statistical properties of images."""
        if not images:
            return {}
            
        stats = {
            'mean_brightness': [],
            'std_brightness': [],
            'mean_contrast': [],
            'histogram_bins': []
        }
        
        for img in images:
            if img is None:
                continue
                
            # Convert to grayscale for analysis
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            stats['mean_brightness'].append(np.mean(gray))
            stats['std_brightness'].append(np.std(gray))
            stats['mean_contrast'].append(np.std(gray))
            
            # Histogram
            hist, _ = np.histogram(gray, bins=50, range=(0, 255))
            stats['histogram_bins'].append(hist)
        
        # Aggregate statistics
        aggregated = {
            'mean_brightness': np.mean(stats['mean_brightness']) if stats['mean_brightness'] else 0,
            'std_brightness': np.mean(stats['std_brightness']) if stats['std_brightness'] else 0,
            'mean_contrast': np.mean(stats['mean_contrast']) if stats['mean_contrast'] else 0,
            'brightness_distribution': np.mean(stats['histogram_bins'], axis=0).tolist() if stats['histogram_bins'] else []
        }
        
        return aggregated
    
    def detect_data_drift(self, current_images: List[np.ndarray]) -> Dict:
        """Detect data drift using statistical tests."""
        if not self.reference_stats:
            logger.warning("No reference statistics found. Creating baseline.")
            current_stats = self.calculate_image_statistics(current_images)
            self._save_reference_stats(current_stats)
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'message': 'Baseline created',
                'timestamp': datetime.now().isoformat()
            }
        
        current_stats = self.calculate_image_statistics(current_images)
        drift_results = {}
        
        # KS test for brightness distribution
        if (len(self.reference_stats.get('brightness_distribution', [])) > 0 and 
            len(current_stats.get('brightness_distribution', [])) > 0):
            try:
                ks_stat, p_value = stats.ks_2samp(
                    self.reference_stats['brightness_distribution'],
                    current_stats['brightness_distribution']
                )
                drift_results['ks_test'] = {
                    'statistic': float(ks_stat),
                    'p_value': float(p_value),
                    'drift_detected': p_value < self.drift_threshold
                }
            except Exception as e:
                logger.error(f"KS test failed: {e}")
                drift_results['ks_test'] = {
                    'statistic': 0.0,
                    'p_value': 1.0,
                    'drift_detected': False,
                    'error': str(e)
                }
        
        # Statistical comparison
        metrics_comparison = {}
        for metric in ['mean_brightness', 'std_brightness', 'mean_contrast']:
            if metric in self.reference_stats and metric in current_stats:
                ref_val = self.reference_stats[metric]
                curr_val = current_stats[metric]
                
                if ref_val != 0:
                    change_pct = abs(curr_val - ref_val) / ref_val
                else:
                    change_pct = 1.0 if curr_val != 0 else 0.0
                
                metrics_comparison[metric] = {
                    'reference': float(ref_val),
                    'current': float(curr_val),
                    'change_percentage': float(change_pct),
                    'drift_detected': change_pct > self.performance_threshold
                }
        
        # Overall drift decision
        drift_detected = any([
            drift_results.get('ks_test', {}).get('drift_detected', False),
            any(m.get('drift_detected', False) for m in metrics_comparison.values())
        ])
        
        drift_score = drift_results.get('ks_test', {}).get('statistic', 0)
        
        return {
            'drift_detected': drift_detected,
            'drift_score': float(drift_score),
            'metrics_comparison': metrics_comparison,
            'statistical_tests': drift_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def detect_model_performance_drift(self, 
                                     y_true: List, 
                                     y_pred: List,
                                     model_name: str = "default") -> Dict:
        """Detect performance drift by comparing current metrics with historical ones."""
        try:
            current_metrics = {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            current_metrics = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        
        # Load historical performance
        perf_file = Path(f"performance_history_{model_name}.json")
        if perf_file.exists():
            try:
                with open(perf_file, 'r') as f:
                    history = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load performance history: {e}")
                history = {'metrics': []}
        else:
            history = {'metrics': []}
        
        # Add current metrics to history
        history['metrics'].append({
            'timestamp': datetime.now().isoformat(),
            **current_metrics
        })
        
        # Keep only last 100 records
        history['metrics'] = history['metrics'][-100:]
        
        # Save updated history
        try:
            with open(perf_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save performance history: {e}")
        
        # Detect drift if we have enough history
        drift_detected = False
        performance_drop = 0.0
        
        if len(history['metrics']) >= 10:
            try:
                recent_f1 = np.mean([m['f1_score'] for m in history['metrics'][-10:]])
                baseline_f1 = np.mean([m['f1_score'] for m in history['metrics'][:10]])
                
                if baseline_f1 > 0:
                    performance_drop = (baseline_f1 - recent_f1) / baseline_f1
                    drift_detected = performance_drop > self.performance_threshold
            except Exception as e:
                logger.error(f"Error calculating performance drift: {e}")
        
        return {
            'drift_detected': drift_detected,
            'current_metrics': current_metrics,
            'performance_drop': float(performance_drop),
            'timestamp': datetime.now().isoformat()
        }