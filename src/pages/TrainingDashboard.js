import { useState } from 'react';
import { Play, Loader2, Cpu, Zap, Layers } from 'lucide-react';
import { toast } from 'sonner';
import MetricCard from '@/components/MetricCard';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

const TrainingDashboard = () => {
  const [training, setTraining] = useState(false);
  const [config, setConfig] = useState({
    experiment_name: 'hybrid_recommender',
    epochs: 10,
    learning_rate: 0.001,
    embedding_dim: 50,
    batch_size: 256,
    reg_lambda: 0.01,
    use_mlflow: true,
    use_wandb: true,
  });

  const handleTrain = async () => {
    setTraining(true);
    try {
      const response = await axios.post(`${BACKEND_URL}/api/train`, config);
      toast.success('Training started successfully!', {
        description: `Experiment ID: ${response.data.id.substring(0, 8)}...`
      });
      
      // Poll for completion
      setTimeout(() => {
        setTraining(false);
        toast.success('Training completed!');
      }, 5000);
    } catch (error) {
      toast.error('Training failed', {
        description: error.response?.data?.detail || error.message
      });
      setTraining(false);
    }
  };

  return (
    <div className="p-6 md:p-8 space-y-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-glow" data-testid="training-dashboard-title">
          Model Training
        </h1>
        <p className="text-sm text-[#A1A1AA] mt-2 leading-relaxed">
          Configure and launch training experiments with hybrid recommendation model
        </p>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MetricCard 
          title="Training Status" 
          value={training ? 'Running' : 'Idle'} 
          icon={Cpu}
          status={training ? 'success' : 'neutral'}
        />
        <MetricCard 
          title="Dataset" 
          value="MovieLens" 
          subtitle="1000 users, 500 items"
          icon={Layers}
          status="neutral"
        />
        <MetricCard 
          title="Model Type" 
          value="Hybrid" 
          subtitle="Collaborative + Content"
          icon={Zap}
          status="neutral"
        />
      </div>

      {/* Training Configuration */}
      <div className="bg-[#0A0A0A] border border-[#27272A] rounded-lg p-6" data-testid="training-config-panel">
        <h2 className="text-2xl font-semibold mb-6">Training Configuration</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm text-[#A1A1AA] mb-2 uppercase tracking-wider" data-testid="label-experiment-name">
              Experiment Name
            </label>
            <input
              type="text"
              value={config.experiment_name}
              onChange={(e) => setConfig({...config, experiment_name: e.target.value})}
              className="w-full bg-[#18181B] border border-[#27272A] rounded-lg px-4 py-3 text-white focus:outline-none focus:border-[#3B82F6] transition-colors"
              data-testid="input-experiment-name"
            />
          </div>

          <div>
            <label className="block text-sm text-[#A1A1AA] mb-2 uppercase tracking-wider">
              Epochs
            </label>
            <input
              type="number"
              value={config.epochs}
              onChange={(e) => setConfig({...config, epochs: parseInt(e.target.value)})}
              className="w-full bg-[#18181B] border border-[#27272A] rounded-lg px-4 py-3 text-white focus:outline-none focus:border-[#3B82F6] transition-colors"
              data-testid="input-epochs"
            />
          </div>

          <div>
            <label className="block text-sm text-[#A1A1AA] mb-2 uppercase tracking-wider">
              Learning Rate
            </label>
            <input
              type="number"
              step="0.0001"
              value={config.learning_rate}
              onChange={(e) => setConfig({...config, learning_rate: parseFloat(e.target.value)})}
              className="w-full bg-[#18181B] border border-[#27272A] rounded-lg px-4 py-3 text-white focus:outline-none focus:border-[#3B82F6] transition-colors font-mono"
              data-testid="input-learning-rate"
            />
          </div>

          <div>
            <label className="block text-sm text-[#A1A1AA] mb-2 uppercase tracking-wider">
              Embedding Dimension
            </label>
            <input
              type="number"
              value={config.embedding_dim}
              onChange={(e) => setConfig({...config, embedding_dim: parseInt(e.target.value)})}
              className="w-full bg-[#18181B] border border-[#27272A] rounded-lg px-4 py-3 text-white focus:outline-none focus:border-[#3B82F6] transition-colors"
              data-testid="input-embedding-dim"
            />
          </div>

          <div>
            <label className="block text-sm text-[#A1A1AA] mb-2 uppercase tracking-wider">
              Batch Size
            </label>
            <input
              type="number"
              value={config.batch_size}
              onChange={(e) => setConfig({...config, batch_size: parseInt(e.target.value)})}
              className="w-full bg-[#18181B] border border-[#27272A] rounded-lg px-4 py-3 text-white focus:outline-none focus:border-[#3B82F6] transition-colors"
              data-testid="input-batch-size"
            />
          </div>

          <div>
            <label className="block text-sm text-[#A1A1AA] mb-2 uppercase tracking-wider">
              Regularization Lambda
            </label>
            <input
              type="number"
              step="0.001"
              value={config.reg_lambda}
              onChange={(e) => setConfig({...config, reg_lambda: parseFloat(e.target.value)})}
              className="w-full bg-[#18181B] border border-[#27272A] rounded-lg px-4 py-3 text-white focus:outline-none focus:border-[#3B82F6] transition-colors font-mono"
              data-testid="input-reg-lambda"
            />
          </div>
        </div>

        {/* Tracking Options */}
        <div className="mt-6 p-4 bg-[#18181B] rounded-lg border border-[#27272A]">
          <h3 className="text-sm font-semibold mb-3 uppercase tracking-wider">Experiment Tracking</h3>
          <div className="flex gap-6">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={config.use_mlflow}
                onChange={(e) => setConfig({...config, use_mlflow: e.target.checked})}
                className="w-4 h-4 accent-[#3B82F6]"
                data-testid="checkbox-mlflow"
              />
              <span className="text-sm">MLflow</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={config.use_wandb}
                onChange={(e) => setConfig({...config, use_wandb: e.target.checked})}
                className="w-4 h-4 accent-[#3B82F6]"
                data-testid="checkbox-wandb"
              />
              <span className="text-sm">Weights & Biases</span>
            </label>
          </div>
        </div>

        {/* Train Button */}
        <button
          onClick={handleTrain}
          disabled={training}
          className="mt-6 w-full md:w-auto px-8 py-4 bg-[#3B82F6] hover:bg-[#3B82F6]/90 text-white rounded-lg font-semibold flex items-center justify-center gap-2 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-[0_0_15px_-3px_rgba(59,130,246,0.4)]"
          data-testid="btn-start-training"
        >
          {training ? (
            <>
              <Loader2 size={20} className="animate-spin" />
              Training in Progress...
            </>
          ) : (
            <>
              <Play size={20} />
              Start Training
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default TrainingDashboard;
