import mlflow
import wandb
import logging
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

class ExperimentTracker:
    """Unified experiment tracking for MLflow and Weights & Biases"""
    
    def __init__(self):
        self.mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        self.wandb_project = os.getenv('WANDB_PROJECT', 'ml-recommendation-system')
        
    def log_to_mlflow(self, experiment_name: str, parameters: Dict[str, Any], 
                     metrics: Dict[str, float]) -> Optional[str]:
        """Log experiment to MLflow"""
        try:
            mlflow.set_tracking_uri(self.mlflow_uri)
            mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run() as run:
                # Log parameters
                for key, value in parameters.items():
                    if isinstance(value, (int, float, str, bool)):
                        mlflow.log_param(key, value)
                
                # Log metrics
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                
                # Log tags
                mlflow.set_tag("model_type", "hybrid_recommendation")
                mlflow.set_tag("framework", "tensorflow")
                
                logger.info(f"Logged to MLflow: run_id={run.info.run_id}")
                return run.info.run_id
        except Exception as e:
            logger.error(f"MLflow logging error: {str(e)}")
            return None
    
    def log_to_wandb(self, experiment_name: str, parameters: Dict[str, Any],
                    metrics: Dict[str, float]) -> Optional[str]:
        """Log experiment to Weights & Biases"""
        try:
            # Initialize wandb in offline mode to avoid needing API key
            os.environ["WANDB_MODE"] = "offline"
            
            run = wandb.init(
                project=self.wandb_project,
                name=experiment_name,
                config=parameters,
                reinit=True
            )
            
            # Log metrics
            wandb.log(metrics)
            
            # Log system info
            wandb.log({
                "model_type": "hybrid_recommendation",
                "framework": "tensorflow"
            })
            
            run_id = run.id
            wandb.finish()
            
            logger.info(f"Logged to W&B: run_id={run_id}")
            return run_id
        except Exception as e:
            logger.error(f"W&B logging error: {str(e)}")
            return None
