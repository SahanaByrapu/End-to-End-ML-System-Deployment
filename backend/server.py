from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
import os
import logging
import uuid
import numpy as np
import pandas as pd
from pathlib import Path

from ml_models.recommendation_model import HybridRecommender
from ml_models.experiment_tracker import ExperimentTracker
from ml_models.monitoring import ModelMonitor
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent / '.env')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'test_database')]

app = FastAPI(title="ML Experiment Tracking System")
api_router = APIRouter(prefix="/api")

# Initialize components
recommender = HybridRecommender()
tracker = ExperimentTracker()
monitor = ModelMonitor(db)

# Models
class TrainingRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    experiment_name: str = "recommendation_experiment"
    epochs: int = 10
    learning_rate: float = 0.001
    embedding_dim: int = 50
    batch_size: int = 256
    reg_lambda: float = 0.01
    use_mlflow: bool = True
    use_wandb: bool = True

class ExperimentResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    experiment_name: str
    status: str
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    timestamp: str
    mlflow_run_id: Optional[str] = None
    wandb_run_id: Optional[str] = None

class ModelMetrics(BaseModel):
    model_config = ConfigDict(extra="ignore")
    precision: float
    recall: float
    f1_score: float
    rmse: float
    mae: float
    ndcg_at_10: float

class MonitoringData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    timestamp: str
    metrics: ModelMetrics
    drift_score: float
    performance_degradation: float

class RetrainingTrigger(BaseModel):
    model_config = ConfigDict(extra="ignore")
    trigger_type: str  # "manual", "scheduled", "performance_drop", "data_drift"
    threshold_met: bool
    current_performance: float
    threshold: float
    recommendation: str

@api_router.post("/train", response_model=ExperimentResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train recommendation model with experiment tracking"""
    try:
        experiment_id = str(uuid.uuid4())
        
        # Save experiment to DB
        experiment_doc = {
            "id": experiment_id,
            "experiment_name": request.experiment_name,
            "status": "running",
            "parameters": request.model_dump(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {}
        }
        await db.experiments.insert_one(experiment_doc)
        
        # Start training in background
        background_tasks.add_task(
            train_model_task,
            experiment_id,
            request
        )
        
        return ExperimentResponse(
            id=experiment_id,
            experiment_name=request.experiment_name,
            status="running",
            metrics={},
            parameters=request.model_dump(),
            timestamp=experiment_doc["timestamp"]
        )
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def train_model_task(experiment_id: str, config: TrainingRequest):
    """Background task for model training"""
    try:
        # Train model with tracking
        metrics = await recommender.train(
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            embedding_dim=config.embedding_dim,
            batch_size=config.batch_size,
            reg_lambda=config.reg_lambda
        )
        
        # Track with MLflow and W&B
        mlflow_run_id, wandb_run_id = None, None
        if config.use_mlflow:
            mlflow_run_id = tracker.log_to_mlflow(config.experiment_name, config.model_dump(), metrics)
        if config.use_wandb:
            wandb_run_id = tracker.log_to_wandb(config.experiment_name, config.model_dump(), metrics)
        
        # Update experiment in DB
        await db.experiments.update_one(
            {"id": experiment_id},
            {"$set": {
                "status": "completed",
                "metrics": metrics,
                "mlflow_run_id": mlflow_run_id,
                "wandb_run_id": wandb_run_id,
                "completed_at": datetime.now(timezone.utc).isoformat()
            }}
        )
        
        # Log to monitoring
        await monitor.log_training_metrics(experiment_id, metrics)
        
    except Exception as e:
        logger.error(f"Training task error: {str(e)}")
        await db.experiments.update_one(
            {"id": experiment_id},
            {"$set": {"status": "failed", "error": str(e)}}
        )

@api_router.get("/experiments", response_model=List[ExperimentResponse])
async def list_experiments(limit: int = 50):
    """List all experiments"""
    try:
        experiments = await db.experiments.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit).to_list(limit)
        return [
            ExperimentResponse(
                id=exp["id"],
                experiment_name=exp["experiment_name"],
                status=exp["status"],
                metrics=exp.get("metrics", {}),
                parameters=exp.get("parameters", {}),
                timestamp=exp["timestamp"],
                mlflow_run_id=exp.get("mlflow_run_id"),
                wandb_run_id=exp.get("wandb_run_id")
            )
            for exp in experiments
        ]
    except Exception as e:
        logger.error(f"List experiments error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: str):
    """Get experiment details"""
    try:
        exp = await db.experiments.find_one({"id": experiment_id}, {"_id": 0})
        if not exp:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        return ExperimentResponse(
            id=exp["id"],
            experiment_name=exp["experiment_name"],
            status=exp["status"],
            metrics=exp.get("metrics", {}),
            parameters=exp.get("parameters", {}),
            timestamp=exp["timestamp"],
            mlflow_run_id=exp.get("mlflow_run_id"),
            wandb_run_id=exp.get("wandb_run_id")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get experiment error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/metrics/comparison")
async def compare_metrics(experiment_ids: str):
    """Compare metrics across experiments"""
    try:
        ids = experiment_ids.split(",")
        experiments = await db.experiments.find(
            {"id": {"$in": ids}},
            {"_id": 0, "id": 1, "experiment_name": 1, "metrics": 1}
        ).to_list(100)
        
        return {"experiments": experiments}
    except Exception as e:
        logger.error(f"Compare metrics error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/monitoring/current")
async def get_current_monitoring():
    """Get current monitoring metrics"""
    try:
        return await monitor.get_current_metrics()
    except Exception as e:
        logger.error(f"Get monitoring error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/monitoring/drift")
async def get_drift_detection():
    """Get data drift detection metrics"""
    try:
        return await monitor.detect_drift()
    except Exception as e:
        logger.error(f"Drift detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/monitoring/degradation")
async def check_performance_degradation():
    """Check model performance degradation"""
    try:
        return await monitor.check_degradation()
    except Exception as e:
        logger.error(f"Performance check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/monitoring/retraining-check", response_model=RetrainingTrigger)
async def check_retraining_needed():
    """Check if model retraining is needed"""
    try:
        degradation_data = await monitor.check_degradation()
        drift_data = await monitor.detect_drift()
        
        performance = degradation_data.get("current_performance", 0.8)
        threshold = degradation_data.get("threshold", 0.75)
        drift_score = drift_data.get("drift_score", 0.0)
        
        trigger_type = "none"
        threshold_met = False
        recommendation = "Model performance is stable. No retraining needed."
        
        if performance < threshold:
            trigger_type = "performance_drop"
            threshold_met = True
            recommendation = f"Performance dropped below threshold ({threshold:.2f}). Recommend immediate retraining."
        elif drift_score > 0.3:
            trigger_type = "data_drift"
            threshold_met = True
            recommendation = f"Significant data drift detected (score: {drift_score:.2f}). Consider retraining soon."
        
        return RetrainingTrigger(
            trigger_type=trigger_type,
            threshold_met=threshold_met,
            current_performance=performance,
            threshold=threshold,
            recommendation=recommendation
        )
    except Exception as e:
        logger.error(f"Retraining check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/recommend/{user_id}")
async def get_recommendations(user_id: int, n: int = 10):
    """Get recommendations for user"""
    try:
        recommendations = recommender.recommend(user_id, n=n)
        return {"user_id": user_id, "recommendations": recommendations}
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
