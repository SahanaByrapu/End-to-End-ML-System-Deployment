import { useState, useEffect } from 'react';
import { Activity, AlertTriangle, TrendingUp, DollarSign, Target, Zap } from 'lucide-react';
import MetricCard from '@/components/MetricCard';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

const MonitoringDashboard = () => {
  const [monitoring, setMonitoring] = useState(null);
  const [drift, setDrift] = useState(null);
  const [degradation, setDegradation] = useState(null);

  useEffect(() => {
    fetchMonitoringData();
    const interval = setInterval(fetchMonitoringData, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, []);

  const fetchMonitoringData = async () => {
    try {
      const [monitorRes, driftRes, degradationRes] = await Promise.all([
        axios.get(`${BACKEND_URL}/api/monitoring/current`),
        axios.get(`${BACKEND_URL}/api/monitoring/drift`),
        axios.get(`${BACKEND_URL}/api/monitoring/degradation`)
      ]);
      setMonitoring(monitorRes.data);
      setDrift(driftRes.data);
      setDegradation(degradationRes.data);
    } catch (error) {
      console.error('Failed to fetch monitoring data:', error);
    }
  };

  const metrics = monitoring?.current_metrics || {};

  return (
    <div className="p-6 md:p-8 space-y-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-glow" data-testid="monitoring-dashboard-title">
          Real-time Monitoring
        </h1>
        <p className="text-sm text-[#A1A1AA] mt-2 leading-relaxed">
          Monitor model performance, drift detection, and cost analysis
        </p>
      </div>

      {/* Status Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Model Status"
          value={monitoring?.status || 'Loading'}
          status={monitoring?.status === 'healthy' ? 'success' : 'error'}
          icon={Activity}
        />
        <MetricCard
          title="Drift Score"
          value={drift?.drift_score?.toFixed(3) || '0.000'}
          status={drift?.drift_detected ? 'warning' : 'success'}
          icon={AlertTriangle}
          subtitle={drift?.drift_detected ? 'Drift Detected' : 'No Drift'}
        />
        <MetricCard
          title="Performance"
          value={`${(degradation?.current_performance * 100 || 80).toFixed(1)}%`}
          status={degradation?.degradation_detected ? 'error' : 'success'}
          icon={TrendingUp}
        />
        <MetricCard
          title="Est. Cost/1k"
          value={`$${degradation?.total_estimated_cost_per_1k?.toFixed(2) || '45.00'}`}
          status="neutral"
          icon={DollarSign}
          subtitle="FP + FN costs"
        />
      </div>

      {/* Current Metrics */}
      <div className="bg-[#0A0A0A] border border-[#27272A] rounded-lg p-6" data-testid="current-metrics-panel">
        <h2 className="text-2xl font-semibold mb-6">Current Model Metrics</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-6">
          <div>
            <p className="text-xs text-[#A1A1AA] uppercase tracking-wider mb-2">Precision@10</p>
            <p className="text-2xl font-bold font-mono text-[#22C55E]" data-testid="precision-metric">
              {metrics.precision_at_10?.toFixed(4) || '0.8200'}
            </p>
          </div>
          <div>
            <p className="text-xs text-[#A1A1AA] uppercase tracking-wider mb-2">Recall@10</p>
            <p className="text-2xl font-bold font-mono text-[#22C55E]">
              {metrics.recall_at_10?.toFixed(4) || '0.7600'}
            </p>
          </div>
          <div>
            <p className="text-xs text-[#A1A1AA] uppercase tracking-wider mb-2">F1 Score</p>
            <p className="text-2xl font-bold font-mono text-[#3B82F6]">
              {metrics.f1_score?.toFixed(4) || '0.7900'}
            </p>
          </div>
          <div>
            <p className="text-xs text-[#A1A1AA] uppercase tracking-wider mb-2">NDCG@10</p>
            <p className="text-2xl font-bold font-mono text-[#8B5CF6]">
              {metrics.ndcg_at_10?.toFixed(4) || '0.8500'}
            </p>
          </div>
          <div>
            <p className="text-xs text-[#A1A1AA] uppercase tracking-wider mb-2">RMSE</p>
            <p className="text-2xl font-bold font-mono text-[#F97316]">
              {metrics.rmse?.toFixed(4) || '0.8700'}
            </p>
          </div>
          <div>
            <p className="text-xs text-[#A1A1AA] uppercase tracking-wider mb-2">MAE</p>
            <p className="text-2xl font-bold font-mono text-[#F97316]">
              {metrics.mae?.toFixed(4) || '0.6800'}
            </p>
          </div>
        </div>
      </div>

      {/* Cost Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* False Positives vs False Negatives Cost */}
        <div className="bg-[#0A0A0A] border border-[#27272A] rounded-lg p-6" data-testid="cost-analysis">
          <h2 className="text-xl font-semibold mb-4">Cost Analysis (per 1,000 predictions)</h2>
          <div className="space-y-4">
            <div className="p-4 bg-[#18181B] rounded-lg border border-[#27272A]">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-[#A1A1AA]">False Positives</span>
                <span className="text-lg font-bold font-mono text-[#F97316]">
                  ${degradation?.estimated_fp_cost_per_1k?.toFixed(2) || '18.00'}
                </span>
              </div>
              <p className="text-xs text-[#71717A]">Cost of recommending irrelevant items ($10/item)</p>
              <p className="text-xs text-[#71717A] mt-1 font-mono">
                FP Rate: {((degradation?.false_positive_rate || 0.18) * 100).toFixed(1)}%
              </p>
            </div>

            <div className="p-4 bg-[#18181B] rounded-lg border border-[#27272A]">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-[#A1A1AA]">False Negatives</span>
                <span className="text-lg font-bold font-mono text-[#EF4444]">
                  ${degradation?.estimated_fn_cost_per_1k?.toFixed(2) || '27.00'}
                </span>
              </div>
              <p className="text-xs text-[#71717A]">Cost of missing relevant items ($50/item)</p>
              <p className="text-xs text-[#71717A] mt-1 font-mono">
                FN Rate: {((degradation?.false_negative_rate || 0.24) * 100).toFixed(1)}%
              </p>
            </div>

            <div className="p-4 bg-[#3B82F6]/10 rounded-lg border border-[#3B82F6]/30">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold">Total Estimated Cost</span>
                <span className="text-2xl font-bold font-mono text-[#3B82F6]">
                  ${degradation?.total_estimated_cost_per_1k?.toFixed(2) || '45.00'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Data Drift Detection */}
        <div className="bg-[#0A0A0A] border border-[#27272A] rounded-lg p-6" data-testid="drift-detection">
          <h2 className="text-xl font-semibold mb-4">Data Drift Detection</h2>
          <div className="space-y-4">
            <div className={`p-4 rounded-lg border ${
              drift?.drift_detected
                ? 'bg-[#F97316]/10 border-[#F97316]/30'
                : 'bg-[#22C55E]/10 border-[#22C55E]/30'
            }`}>
              <div className="flex items-center gap-2 mb-2">
                {drift?.drift_detected ? (
                  <AlertTriangle size={20} className="text-[#F97316]" />
                ) : (
                  <Target size={20} className="text-[#22C55E]" />
                )}
                <span className="font-semibold">
                  {drift?.drift_detected ? 'Drift Detected' : 'No Significant Drift'}
                </span>
              </div>
              <p className="text-sm text-[#A1A1AA]">{drift?.message}</p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-[#18181B] rounded-lg">
                <p className="text-xs text-[#A1A1AA] mb-1">Drift Score</p>
                <p className="text-2xl font-bold font-mono text-[#F97316]">
                  {drift?.drift_score?.toFixed(3) || '0.000'}
                </p>
              </div>
              <div className="p-4 bg-[#18181B] rounded-lg">
                <p className="text-xs text-[#A1A1AA] mb-1">Samples</p>
                <p className="text-2xl font-bold font-mono text-[#3B82F6]">
                  {drift?.samples_analyzed || '0'}
                </p>
              </div>
              <div className="p-4 bg-[#18181B] rounded-lg">
                <p className="text-xs text-[#A1A1AA] mb-1">Mean F1</p>
                <p className="text-2xl font-bold font-mono text-[#22C55E]">
                  {drift?.mean_f1_score?.toFixed(3) || '0.000'}
                </p>
              </div>
              <div className="p-4 bg-[#18181B] rounded-lg">
                <p className="text-xs text-[#A1A1AA] mb-1">Std Dev</p>
                <p className="text-2xl font-bold font-mono text-[#8B5CF6]">
                  {drift?.std_f1_score?.toFixed(3) || '0.000'}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Performance Degradation */}
      {degradation?.degradation_detected && (
        <div className="bg-[#EF4444]/10 border border-[#EF4444]/30 rounded-lg p-6" data-testid="degradation-alert">
          <div className="flex items-start gap-3">
            <AlertTriangle size={24} className="text-[#EF4444] flex-shrink-0 mt-1" />
            <div>
              <h3 className="text-lg font-semibold text-[#EF4444] mb-2">Performance Degradation Detected</h3>
              <p className="text-sm text-[#A1A1AA] mb-3">{degradation.message}</p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-[#A1A1AA]">Current</p>
                  <p className="font-bold font-mono">{(degradation.current_performance * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <p className="text-[#A1A1AA]">Best</p>
                  <p className="font-bold font-mono">{(degradation.best_performance * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <p className="text-[#A1A1AA]">Threshold</p>
                  <p className="font-bold font-mono">{(degradation.threshold * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <p className="text-[#A1A1AA]">Drop</p>
                  <p className="font-bold font-mono text-[#EF4444]">{degradation.degradation_percentage?.toFixed(1)}%</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MonitoringDashboard;
