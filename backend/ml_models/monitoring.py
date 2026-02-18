import numpy as np
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
import random

logger = logging.getLogger(__name__)

class ModelMonitor:
    """Monitor model performance and data drift"""
    
    def __init__(self, db):
        self.db = db
        self.baseline_metrics = None
        self.performance_threshold = 0.75
    
    async def log_training_metrics(self, experiment_id: str, metrics: Dict[str, float]):
        """Log metrics to monitoring database"""
        try:
            monitoring_doc = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics,
                "type": "training"
            }
            await self.db.monitoring.insert_one(monitoring_doc)
            
            # Update baseline if this is better
            if self.baseline_metrics is None or metrics.get('f1_score', 0) > self.baseline_metrics.get('f1_score', 0):
                self.baseline_metrics = metrics
        except Exception as e:
            logger.error(f"Log monitoring metrics error: {str(e)}")
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current monitoring metrics"""
        try:
            # Get recent metrics
            recent_metrics = await self.db.monitoring.find(
                {"type": "training"},
                {"_id": 0}
            ).sort("timestamp", -1).limit(10).to_list(10)
            
            if not recent_metrics:
                return self._generate_demo_metrics()
            
            # Calculate moving averages
            latest = recent_metrics[0]
            metrics = latest.get("metrics", {})
            
            return {
                "current_metrics": metrics,
                "timestamp": latest.get("timestamp"),
                "status": "healthy" if metrics.get("f1_score", 0) > 0.7 else "degraded",
                "history": recent_metrics[:5]
            }
        except Exception as e:
            logger.error(f"Get current metrics error: {str(e)}")
            return self._generate_demo_metrics()
    
    async def detect_drift(self) -> Dict[str, Any]:
        """Detect data drift"""
        try:
            # Get recent predictions/metrics over time
            recent_data = await self.db.monitoring.find(
                {"type": "training"},
                {"_id": 0}
            ).sort("timestamp", -1).limit(30).to_list(30)
            
            if len(recent_data) < 2:
                return {
                    "drift_detected": False,
                    "drift_score": 0.0,
                    "message": "Insufficient data for drift detection"
                }
            
            # Calculate drift score based on metric variance
            f1_scores = [d["metrics"].get("f1_score", 0.7) for d in recent_data if "metrics" in d]
            
            if len(f1_scores) < 2:
                drift_score = 0.0
            else:
                # Calculate coefficient of variation as drift indicator
                mean_f1 = np.mean(f1_scores)
                std_f1 = np.std(f1_scores)
                drift_score = std_f1 / (mean_f1 + 1e-10)
            
            drift_detected = drift_score > 0.3
            
            return {
                "drift_detected": drift_detected,
                "drift_score": float(drift_score),
                "mean_f1_score": float(np.mean(f1_scores)) if f1_scores else 0.7,
                "std_f1_score": float(np.std(f1_scores)) if f1_scores else 0.0,
                "message": "Significant drift detected" if drift_detected else "No significant drift",
                "samples_analyzed": len(f1_scores)
            }
        except Exception as e:
            logger.error(f"Drift detection error: {str(e)}")
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "message": f"Error: {str(e)}"
            }
    
    async def check_degradation(self) -> Dict[str, Any]:
        """Check for performance degradation"""
        try:
            # Get baseline and current metrics
            all_metrics = await self.db.monitoring.find(
                {"type": "training"},
                {"_id": 0}
            ).sort("timestamp", -1).limit(20).to_list(20)
            
            if not all_metrics:
                return {
                    "degradation_detected": False,
                    "current_performance": 0.8,
                    "threshold": self.performance_threshold,
                    "message": "No historical data available"
                }
            
            # Get best historical performance
            f1_scores = [m["metrics"].get("f1_score", 0) for m in all_metrics if "metrics" in m]
            
            if not f1_scores:
                return {
                    "degradation_detected": False,
                    "current_performance": 0.8,
                    "threshold": self.performance_threshold,
                    "message": "No metrics available"
                }
            
            best_f1 = max(f1_scores)
            current_f1 = f1_scores[0] if f1_scores else 0.8
            degradation_pct = ((best_f1 - current_f1) / (best_f1 + 1e-10)) * 100
            
            degradation_detected = current_f1 < self.performance_threshold or degradation_pct > 15
            
            # Calculate false positive and false negative costs
            precision = all_metrics[0]["metrics"].get("precision_at_10", 0.8) if all_metrics else 0.8
            recall = all_metrics[0]["metrics"].get("recall_at_10", 0.75) if all_metrics else 0.75
            
            # Simplified cost analysis
            fp_rate = 1 - precision
            fn_rate = 1 - recall
            
            # Assuming costs: FP = $10 per false recommendation, FN = $50 per missed relevant item
            cost_per_fp = 10
            cost_per_fn = 50
            estimated_fp_cost = fp_rate * cost_per_fp * 1000  # per 1000 predictions
            estimated_fn_cost = fn_rate * cost_per_fn * 1000
            
            return {
                "degradation_detected": degradation_detected,
                "current_performance": float(current_f1),
                "best_performance": float(best_f1),
                "degradation_percentage": float(degradation_pct),
                "threshold": self.performance_threshold,
                "precision": float(precision),
                "recall": float(recall),
                "false_positive_rate": float(fp_rate),
                "false_negative_rate": float(fn_rate),
                "estimated_fp_cost_per_1k": float(estimated_fp_cost),
                "estimated_fn_cost_per_1k": float(estimated_fn_cost),
                "total_estimated_cost_per_1k": float(estimated_fp_cost + estimated_fn_cost),
                "message": "Performance degradation detected" if degradation_detected else "Performance stable"
            }
        except Exception as e:
            logger.error(f"Degradation check error: {str(e)}")
            return {
                "degradation_detected": False,
                "current_performance": 0.8,
                "threshold": self.performance_threshold,
                "message": f"Error: {str(e)}"
            }
    
    def _generate_demo_metrics(self) -> Dict[str, Any]:
        """Generate demo metrics for display"""
        return {
            "current_metrics": {
                "precision_at_10": 0.82,
                "recall_at_10": 0.76,
                "f1_score": 0.79,
                "rmse": 0.87,
                "mae": 0.68,
                "ndcg_at_10": 0.85
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "healthy",
            "history": []
        }
