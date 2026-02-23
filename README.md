
## ML Ops Pro: End-to-End ML Experiment Tracking System
## 1. Executive Summary

ML Ops Pro is a comprehensive experiment tracking and model monitoring platform designed for machine learning teams building recommendation systems. The platform provides end-to-end tracking of ML experiments using MLflow and Weights & Biases, with integrated monitoring, performance degradation detection, and automated retraining triggers.

### Key Objectives
- Track ML experiments with dual integration (MLflow + W&B)
- Monitor model performance in real-time with actionable metrics
- Detect data drift and performance degradation automatically
- Provide cost analysis for false positives and false negatives
- Guide retraining decisions with clear triggers and thresholds

## 2. Target Audience

### Primary Users
- **ML Engineers**: Training, tuning, and deploying recommendation models
- **Data Scientists**: Experimenting with different model architectures and hyperparameters
- **ML Ops Engineers**: Monitoring production models and managing retraining pipelines

### Secondary Users
- **Engineering Managers**: Tracking team productivity and model performance trends
- **Product Managers**: Understanding model quality and business impact

## 3. Product Overview

### 3.1 Core Features

#### Feature 1: Model Training Dashboard
**Description:** Configure and launch training experiments with comprehensive hyperparameter control.

**Capabilities:**
- Configure experiment parameters (epochs, learning rate, embedding dimension, batch size, regularization)
- Select experiment tracking tools (MLflow, W&B, or both)
- Initiate training with background job processing
- Real-time status updates with toast notifications
- Display current dataset and model information

**User Value:** Streamlined experiment setup with instant feedback and flexible tracking options.

#### Feature 2: Experiment History & Tracking
**Description:** View and analyze all past training experiments with sortable metrics.

**Capabilities:**
- List all experiments with status indicators (running, completed, failed)
- Display key metrics: Precision@10, Recall@10, F1 Score, RMSE, MAE, NDCG@10
- Show experiment metadata (IDs, timestamps, configurations)
- Filter and sort experiments by performance
- Link to MLflow and W&B run IDs for detailed analysis

**User Value:** Centralized view of all experiments enabling quick performance comparison and historical analysis.

#### Feature 3: Model Comparison
**Description:** Side-by-side comparison of multiple experiments with visual analytics.

**Capabilities:**
- Select multiple experiments for comparison
- Visualize metrics with bar charts (Precision, Recall, F1)
- Compare RMSE across models (lower is better)
- Detailed comparison table with all metrics
- Interactive charts built with Recharts

**User Value:** Data-driven model selection with clear visual comparison of performance metrics.

#### Feature 4: Real-time Monitoring Dashboard
**Description:** Monitor production model performance with live metrics and alerts.

**Capabilities:**
- Display current model status (healthy, degraded)
- Show drift score with threshold-based alerts
- Track performance percentage against baseline
- Calculate estimated costs (FP and FN per 1k predictions)
- Display comprehensive current metrics (6 key metrics)
- Visualize false positive and false negative costs
- Data drift detection with statistical analysis
- Performance degradation alerts with detailed breakdown

**User Value:** Proactive monitoring prevents quality degradation and provides cost transparency.

#### Feature 5: Cost Analysis System
**Description:** Analyze business impact of prediction errors with cost breakdown.

**Capabilities:**
- Calculate false positive costs ($10/item for irrelevant recommendations)
- Calculate false negative costs ($50/item for missed relevant items)
- Estimate total cost per 1,000 predictions
- Display FP and FN rates as percentages
- Show cost breakdown with visual indicators

**User Value:** Connect model performance to business metrics, enabling ROI-driven decisions.

#### Feature 6: Data Drift Detection
**Description:** Automatically detect shifts in data distribution that may degrade performance.

**Capabilities:**
- Calculate drift score using coefficient of variation
- Set threshold at 0.30 for drift detection
- Analyze sample size and statistical significance
- Display mean F1 score and standard deviation
- Visual indicators (green = no drift, orange = drift detected)

**User Value:** Early warning system prevents gradual model decay from data shifts.

#### Feature 7: Retraining Management
**Description:** Intelligent retraining recommendations based on multiple triggers.

**Capabilities:**
- **Performance Drop Trigger**: Activate when F1 < 75%
- **Data Drift Trigger**: Activate when drift score > 0.30
- **Scheduled Trigger**: Weekly retraining (Monday 2:00 AM)
- Display current performance vs. threshold
- Provide actionable recommendations
- Three-tier decision guide (Immediate, Consider Soon, Monitor)

**User Value:** Automated guardrails ensure model quality while minimizing unnecessary retraining.

## 4. Technical Architecture

### 4.1 Technology Stack

**Backend:**
- Framework: FastAPI (Python)
- ML Framework: TensorFlow/Keras 2.x
- Database: MongoDB (async with Motor)
- Experiment Tracking: MLflow + Weights & Biases
- Model Type: Hybrid Recommendation (Collaborative + Content-based)

**Frontend:**
- Framework: React 18.x
- Styling: Tailwind CSS
- Charts: Recharts
- Icons: Lucide React
- Routing: React Router v6
- State Management: React Hooks
- Notifications: Sonner (toast library)

**Infrastructure:**
- Container: Kubernetes pod
- Services: Supervisor-managed (hot reload enabled)
- Reverse Proxy: Nginx
- Deployment: Emergent platform



### 4.3 Data Models

#### Experiment Document
```javascript
{
  id: string (UUID),
  experiment_name: string,
  status: "running" | "completed" | "failed",
  parameters: {
    epochs: number,
    learning_rate: number,
    embedding_dim: number,
    batch_size: number,
    reg_lambda: number,
    use_mlflow: boolean,
    use_wandb: boolean
  },
  metrics: {
    precision_at_10: float,
    recall_at_10: float,
    f1_score: float,
    rmse: float,
    mae: float,
    ndcg_at_10: float,
    train_loss: float,
    val_loss: float
  },
  timestamp: ISO8601 string,
  completed_at: ISO8601 string,
  mlflow_run_id: string,
  wandb_run_id: string,
  error: string (optional)
}
```

#### Monitoring Document
```javascript
{
  experiment_id: string,
  timestamp: ISO8601 string,
  metrics: ModelMetrics,
  type: "training" | "inference"
}
```

### 4.4 API Endpoints

#### Training & Experiments
- `POST /api/train` - Start training experiment
- `GET /api/experiments` - List all experiments (limit: 50)
- `GET /api/experiments/{id}` - Get experiment details
- `GET /api/metrics/comparison?experiment_ids=id1,id2` - Compare experiments

#### Monitoring
- `GET /api/monitoring/current` - Get current model metrics
- `GET /api/monitoring/drift` - Detect data drift
- `GET /api/monitoring/degradation` - Check performance degradation
- `POST /api/monitoring/retraining-check` - Evaluate retraining needs

#### Recommendations
- `GET /api/recommend/{user_id}?n=10` - Get recommendations

---

## 5. ML Model Specifications

### 5.1 Hybrid Recommendation Model

**Architecture:**
```
Input: User ID + Item ID

User Embedding (50-dim) ──┐
                          │
                          ├──> Dot Product ──┐
                          │                   │
Item Embedding (50-dim) ──┘                   ├──> Add ──> Output (Rating)
                                              │
User + Item Features ──> Dense(128) ──> Dense(64) ──> Dense(1) ──┘
```

**Components:**
1. **Collaborative Filtering**: User-item interaction via embeddings and dot product
2. **Content-based Filtering**: Deep neural network on concatenated features
3. **Hybrid**: Additive combination of both approaches

**Hyperparameters:**
- Embedding dimension: 50 (default)
- Learning rate: 0.001 (default)
- Batch size: 256 (default)
- Regularization: L2 with lambda 0.01 (default)
- Optimizer: Adam
- Loss: Mean Squared Error (MSE)

### 5.2 Dataset

**Source:** Synthetic MovieLens-style data
- Users: 1,000
- Items: 500
- Ratings: 50,000 interactions
- Rating scale: 1-5 stars
- Features: Genre, popularity, average rating

**Train/Test Split:** 80/20

### 5.3 Evaluation Metrics

#### Primary Metrics
- **Precision@10**: Proportion of relevant items in top 10 recommendations
- **Recall@10**: Proportion of relevant items retrieved in top 10
- **F1 Score**: Harmonic mean of precision and recall
- **NDCG@10**: Normalized Discounted Cumulative Gain at 10

#### Secondary Metrics
- **RMSE**: Root Mean Squared Error for rating prediction
- **MAE**: Mean Absolute Error for rating prediction

#### Threshold for Production
- **Minimum F1 Score**: 0.75 (75%)
- **Maximum RMSE**: 1.0
- **Minimum Precision@10**: 0.70

---

## 6. Design Specifications

### 6.1 Theme: Professional Dark Mode

**Color Palette:**
```
Background:      #09090B (deep black)
Surface:         #0A0A0A (near black)
Borders:         #27272A (dark gray)
Primary:         #3B82F6 (electric blue)
Secondary:       #22C55E (neon green)
Warning:         #F97316 (orange)
Error:           #EF4444 (red)
Text Primary:    #FAFAFA (off-white)
Text Secondary:  #A1A1AA (gray)
```

**Typography:**
- Headings: Space Grotesk (700 weight)
- Body: Inter (400-600 weight)
- Code/Data: JetBrains Mono

**Visual Effects:**
- Text glow on title (blue shadow)
- Card glow on hover (subtle blue shadow)
- Smooth transitions (200-300ms)
- Rounded corners (lg = 8px)

### 6.2 Component Library

**Using:** Shadcn/UI components from `/app/frontend/src/components/ui/`
- Button
- Input
- Card (custom built)
- Toast (Sonner)
- Icons (Lucide React)

### 6.3 Responsive Breakpoints
- Mobile: < 768px
- Tablet: 768px - 1024px
- Desktop: > 1024px
- Large Desktop: > 1920px

---

## 7. User Workflows

### 7.1 Training a New Model

1. Navigate to Training Dashboard
2. Configure hyperparameters:
   - Set experiment name
   - Adjust epochs, learning rate, embedding dim
   - Configure batch size and regularization
3. Select tracking tools (MLflow, W&B, or both)
4. Click "Start Training"
5. Receive toast notification with experiment ID
6. Monitor status (changes from "Idle" to "Running")
7. Wait for completion (background job)
8. Navigate to Experiments to view results

**Expected Time:** 30-60 seconds for default configuration (10 epochs)

### 7.2 Comparing Models

1. Navigate to Experiments History
2. Review completed experiments in table
3. Navigate to Comparison page
4. Select 2-4 experiments using checkboxes
5. View bar charts comparing metrics
6. Analyze comparison table with detailed metrics
7. Identify best-performing model
8. Note experiment ID for deployment

**Key Decision Criteria:**
- Highest F1 Score
- Balance of Precision and Recall
- Acceptable RMSE (< 1.0)
- High NDCG@10 (ranking quality)

### 7.3 Monitoring Production Model

1. Navigate to Monitoring Dashboard
2. Check Model Status card (should be "healthy")
3. Review Drift Score (should be < 0.30)
4. Verify Performance percentage (should be > 80%)
5. Analyze Cost Analysis section:
   - Review FP and FN rates
   - Check total estimated cost
6. Examine Data Drift Detection panel
7. Look for Performance Degradation alerts
8. Take action if thresholds exceeded

**Monitoring Frequency:** Real-time (auto-refresh every 10s)

### 7.4 Deciding on Retraining

1. Navigate to Retraining Management
2. Click "Refresh Status" for latest check
3. Review Retraining Status panel:
   - Check trigger type
   - Compare current performance to threshold
4. Read recommendation message
5. Review Retraining Triggers section
6. Consult Decision Guide:
   - **Immediate**: Red alert, retrain now
   - **Consider Soon**: Orange alert, plan retraining
   - **Monitor**: Green, no action needed
7. If retraining needed, return to Training Dashboard

**Decision Matrix:**
- F1 < 75% → Immediate retraining
- Drift > 0.30 → Consider retraining
- Degradation > 15% → Investigate and retrain
- Cost > $60/1k → Optimize or retrain

---

## 8. Performance Requirements

### 8.1 Response Times
- Dashboard page load: < 2 seconds
- API endpoint response: < 500ms
- Training initiation: < 1 second
- Chart rendering: < 300ms
- Navigation transitions: < 200ms

### 8.2 Scalability
- Support 100+ concurrent experiments in history
- Handle 10 simultaneous training jobs
- Store 1000+ experiment records
- Display 50 experiments per page (pagination ready)

### 8.3 Reliability
- 99.5% uptime for monitoring dashboard
- Graceful degradation if MLflow/W&B unavailable
- Automatic retry for failed training jobs (future)
- Data persistence via MongoDB

---

## 9. Success Metrics

### 9.1 Product Metrics
- **Experiment Velocity**: Number of experiments per week
- **Model Quality**: Average F1 score across experiments
- **Time to Production**: Days from training to deployment
- **Retraining Frequency**: Number of retrains per month
- **Cost Savings**: Reduction in FP/FN costs over time

### 9.2 User Engagement
- **Daily Active Users**: ML engineers using platform
- **Feature Adoption**: % using comparison, monitoring, retraining pages
- **Average Session Duration**: Time spent in platform
- **Experiments per User**: Productivity indicator

---

## 10. Known Limitations

### 10.1 Current Constraints
- MLflow runs in offline mode (not connected to hosted server)
- W&B runs in offline mode (no API key configured)
- Single model type supported (hybrid recommendation)
- Synthetic dataset (not production data)
- Manual retraining trigger (no automation)
- No A/B testing framework
- Limited to 50 experiments display (pagination needed)

### 10.2 Browser Compatibility
- Tested: Chrome 120+, Firefox 121+, Safari 17+
- Not tested: IE 11, Edge Legacy

---

## 11. Future Enhancements

### Phase 2 (Q2 2026)
- [ ] Connect to hosted MLflow server
- [ ] Integrate W&B API key for cloud sync
- [ ] Add automated retraining pipeline
- [ ] Implement A/B testing framework
- [ ] Add confusion matrix visualization
- [ ] Support multiple model architectures

### Phase 3 (Q3 2026)
- [ ] Real production dataset integration
- [ ] Email alerts for performance degradation
- [ ] Slack integration for team notifications
- [ ] Custom metric definitions
- [ ] Model versioning and rollback
- [ ] ROC curve and PR curve visualization

### Phase 4 (Q4 2026)
- [ ] Multi-tenancy support
- [ ] Role-based access control (RBAC)
- [ ] Model explainability dashboard
- [ ] Hyperparameter optimization (HPO) integration
- [ ] Feature importance tracking
- [ ] Data quality monitoring

---

## 12. Dependencies

### 12.1 External Services
- **MLflow**: Experiment tracking and model registry
- **Weights & Biases**: Experiment visualization and collaboration
- **MongoDB**: Metadata storage
- **Emergent Platform**: Deployment and hosting

### 12.2 Python Libraries
- tensorflow==2.16.1
- mlflow==2.18.0
- wandb==0.18.7
- fastapi==0.115.6
- motor==3.6.0 (async MongoDB)
- pydantic==2.10.5
- numpy==1.26.4
- pandas==2.2.3
- scikit-learn==1.5.2

### 12.3 JavaScript Libraries
- react==18.3.1
- react-router-dom==6.28.0
- recharts==2.15.0
- axios==1.7.9
- lucide-react==0.468.0
- sonner==1.7.1
- tailwindcss==3.4.17

---

## 13. Security & Privacy

### 13.1 Data Security
- MongoDB credentials stored in environment variables
- API endpoints require CORS validation
- No sensitive data in client-side code
- HTTPS enforced for all communications

### 13.2 Access Control
- Currently no authentication (single-tenant)
- Future: OAuth 2.0 integration planned
- Future: API key authentication for programmatic access

---

## 14. Deployment

### 14.1 Environment Variables

**Backend (.env):**
```bash
MONGO_URL=mongodb://localhost:27017
DB_NAME=ml_ops_db
CORS_ORIGINS=*
```

**Frontend (.env):**
```bash
REACT_APP_BACKEND_URL=https://[app-name].preview.emergentagent.com
```

### 14.2 Service Configuration
- Backend: FastAPI on 0.0.0.0:8001
- Frontend: React on 0.0.0.0:3000
- MongoDB: Default port 27017
- Hot reload enabled for development

### 14.3 Health Checks
- Backend: `GET /docs` (FastAPI auto-docs)
- Frontend: Root path loads React app
- MongoDB: Connection test on startup

---

## 15. Support & Maintenance

### 15.1 Monitoring
- Supervisor logs: `/var/log/supervisor/backend.*.log`
- Frontend logs: `/var/log/supervisor/frontend.*.log`
- MongoDB logs: System logs

### 15.2 Troubleshooting
- Service status: `sudo supervisorctl status`
- Restart services: `sudo supervisorctl restart backend frontend`
- View logs: `tail -f /var/log/supervisor/backend.err.log`

### 15.3 Backup & Recovery
- MongoDB backup: Regular exports recommended
- Experiment data: Stored in MongoDB (backed up with DB)
- MLflow artifacts: Stored locally (backup recommended)

---

## 16. Compliance

### 16.1 Code Quality
- Python: PEP 8 compliant (Ruff linter)
- JavaScript: ESLint configured
- Type checking: Pydantic models (backend), PropTypes (frontend)

### 16.2 Testing
- Backend: Unit tests recommended (pytest)
- Frontend: Component tests recommended (React Testing Library)
- E2E: Playwright tests recommended

---

## Appendix A: Glossary

- **Embedding Dimension**: Size of learned feature vectors for users/items
- **F1 Score**: Harmonic mean of precision and recall
- **NDCG**: Normalized Discounted Cumulative Gain (ranking metric)
- **Drift**: Statistical change in data distribution over time
- **False Positive**: Recommending irrelevant item
- **False Negative**: Missing relevant item
- **Hybrid Model**: Combination of collaborative and content-based filtering

---

## Appendix B: References

- MLflow Documentation: https://mlflow.org/docs/latest/
- Weights & Biases: https://docs.wandb.ai/
- TensorFlow Recommenders: https://www.tensorflow.org/recommenders
- FastAPI: https://fastapi.tiangolo.com/
- React: https://react.dev/

---



*End of Product Requirements Document*
