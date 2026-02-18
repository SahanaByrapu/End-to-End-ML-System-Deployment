import { useState, useEffect } from 'react';
import { RefreshCw, AlertCircle, CheckCircle, Clock, Activity } from 'lucide-react';
import { toast } from 'sonner';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

const RetrainingManagement = () => {
  const [retrainingCheck, setRetrainingCheck] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    checkRetrainingNeeded();
  }, []);

  const checkRetrainingNeeded = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${BACKEND_URL}/api/monitoring/retraining-check`);
      setRetrainingCheck(response.data);
    } catch (error) {
      console.error('Failed to check retraining:', error);
      toast.error('Failed to check retraining status');
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = () => {
    if (!retrainingCheck) return <Clock size={24} className="text-[#A1A1AA]" />;
    if (retrainingCheck.trigger_type === 'performance_drop') {
      return <AlertCircle size={24} className="text-[#EF4444]" />;
    }
    if (retrainingCheck.trigger_type === 'data_drift') {
      return <AlertCircle size={24} className="text-[#F97316]" />;
    }
    return <CheckCircle size={24} className="text-[#22C55E]" />;
  };

  const getStatusColor = () => {
    if (!retrainingCheck) return 'border-[#27272A]';
    if (retrainingCheck.trigger_type === 'performance_drop') return 'border-[#EF4444]/30 bg-[#EF4444]/5';
    if (retrainingCheck.trigger_type === 'data_drift') return 'border-[#F97316]/30 bg-[#F97316]/5';
    return 'border-[#22C55E]/30 bg-[#22C55E]/5';
  };

  return (
    <div className="p-6 md:p-8 space-y-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-glow" data-testid="retraining-management-title">
          Retraining Management
        </h1>
        <p className="text-sm text-[#A1A1AA] mt-2 leading-relaxed">
          Monitor triggers and manage automated model retraining
        </p>
      </div>

      {/* Retraining Status */}
      <div className={`bg-[#0A0A0A] border rounded-lg p-6 ${getStatusColor()}`} data-testid="retraining-status-panel">
        <div className="flex items-start gap-4">
          <div className="flex-shrink-0">
            {getStatusIcon()}
          </div>
          <div className="flex-1">
            <h2 className="text-2xl font-semibold mb-2">Retraining Status</h2>
            {retrainingCheck ? (
              <>
                <p className="text-lg text-[#A1A1AA] mb-4">{retrainingCheck.recommendation}</p>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="p-4 bg-[#18181B] border border-[#27272A] rounded-lg">
                    <p className="text-xs text-[#A1A1AA] uppercase tracking-wider mb-2">Trigger Type</p>
                    <p className="text-lg font-semibold capitalize">
                      {retrainingCheck.trigger_type.replace('_', ' ')}
                    </p>
                  </div>
                  <div className="p-4 bg-[#18181B] border border-[#27272A] rounded-lg">
                    <p className="text-xs text-[#A1A1AA] uppercase tracking-wider mb-2">Current Performance</p>
                    <p className="text-lg font-bold font-mono text-[#3B82F6]">
                      {(retrainingCheck.current_performance * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="p-4 bg-[#18181B] border border-[#27272A] rounded-lg">
                    <p className="text-xs text-[#A1A1AA] uppercase tracking-wider mb-2">Threshold</p>
                    <p className="text-lg font-bold font-mono text-[#F97316]">
                      {(retrainingCheck.threshold * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
              </>
            ) : (
              <p className="text-[#A1A1AA]">Loading retraining status...</p>
            )}
          </div>
        </div>
      </div>

      {/* Retraining Triggers */}
      <div className="bg-[#0A0A0A] border border-[#27272A] rounded-lg p-6" data-testid="retraining-triggers">
        <h2 className="text-2xl font-semibold mb-6">Retraining Triggers</h2>
        <div className="space-y-4">
          <div className="p-4 bg-[#18181B] border border-[#27272A] rounded-lg">
            <div className="flex items-center gap-3 mb-3">
              <Activity size={20} className="text-[#EF4444]" />
              <h3 className="font-semibold">Performance Drop</h3>
            </div>
            <p className="text-sm text-[#A1A1AA] mb-2">
              Automatically trigger retraining when model performance (F1 score) drops below threshold.
            </p>
            <div className="flex items-center gap-2 text-sm">
              <span className="text-[#A1A1AA]">Threshold:</span>
              <span className="font-mono font-bold">75%</span>
            </div>
          </div>

          <div className="p-4 bg-[#18181B] border border-[#27272A] rounded-lg">
            <div className="flex items-center gap-3 mb-3">
              <Activity size={20} className="text-[#F97316]" />
              <h3 className="font-semibold">Data Drift Detection</h3>
            </div>
            <p className="text-sm text-[#A1A1AA] mb-2">
              Trigger retraining when significant drift is detected in input data distribution.
            </p>
            <div className="flex items-center gap-2 text-sm">
              <span className="text-[#A1A1AA]">Drift Score Threshold:</span>
              <span className="font-mono font-bold">0.30</span>
            </div>
          </div>

          <div className="p-4 bg-[#18181B] border border-[#27272A] rounded-lg">
            <div className="flex items-center gap-3 mb-3">
              <Clock size={20} className="text-[#3B82F6]" />
              <h3 className="font-semibold">Scheduled Retraining</h3>
            </div>
            <p className="text-sm text-[#A1A1AA] mb-2">
              Periodic retraining to keep model updated with latest data patterns.
            </p>
            <div className="flex items-center gap-2 text-sm">
              <span className="text-[#A1A1AA]">Schedule:</span>
              <span className="font-mono font-bold">Weekly (Every Monday 2:00 AM)</span>
            </div>
          </div>
        </div>
      </div>

      {/* When to Retrain Decision Guide */}
      <div className="bg-[#0A0A0A] border border-[#27272A] rounded-lg p-6" data-testid="retraining-guide">
        <h2 className="text-2xl font-semibold mb-6">When to Retrain: Decision Guide</h2>
        <div className="space-y-4">
          <div className="p-5 bg-[#22C55E]/10 border border-[#22C55E]/30 rounded-lg">
            <h3 className="font-semibold text-[#22C55E] mb-2 flex items-center gap-2">
              <CheckCircle size={18} />
              Immediate Retraining Required
            </h3>
            <ul className="text-sm text-[#A1A1AA] space-y-1 ml-6 list-disc">
              <li>F1 Score drops below 75% (15%+ degradation from baseline)</li>
              <li>False Negative rate increases significantly (greater than 30%)</li>
              <li>Estimated cost per 1k predictions exceeds $60</li>
              <li>Critical business metrics show negative impact</li>
            </ul>
          </div>

          <div className="p-5 bg-[#F97316]/10 border border-[#F97316]/30 rounded-lg">
            <h3 className="font-semibold text-[#F97316] mb-2 flex items-center gap-2">
              <AlertCircle size={18} />
              Consider Retraining Soon
            </h3>
            <ul className="text-sm text-[#A1A1AA] space-y-1 ml-6 list-disc">
              <li>Drift score exceeds 0.30</li>
              <li>Performance degradation between 10-15%</li>
              <li>User behavior patterns have shifted</li>
              <li>New items/users added to catalog (&gt;20% growth)</li>
            </ul>
          </div>

          <div className="p-5 bg-[#3B82F6]/10 border border-[#3B82F6]/30 rounded-lg">
            <h3 className="font-semibold text-[#3B82F6] mb-2 flex items-center gap-2">
              <Activity size={18} />
              Monitor Closely
            </h3>
            <ul className="text-sm text-[#A1A1AA] space-y-1 ml-6 list-disc">
              <li>Performance stable but approaching threshold</li>
              <li>Seasonal patterns affecting recommendations</li>
              <li>A/B test results show marginal improvement potential</li>
              <li>Regular scheduled maintenance period approaching</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-4">
        <button
          onClick={checkRetrainingNeeded}
          disabled={loading}
          className="px-6 py-3 bg-[#3B82F6] hover:bg-[#3B82F6]/90 text-white rounded-lg font-semibold flex items-center gap-2 transition-all duration-200 disabled:opacity-50 shadow-[0_0_15px_-3px_rgba(59,130,246,0.4)]"
          data-testid="btn-check-retraining"
        >
          <RefreshCw size={18} className={loading ? 'animate-spin' : ''} />
          Refresh Status
        </button>
      </div>
    </div>
  );
};

export default RetrainingManagement;
