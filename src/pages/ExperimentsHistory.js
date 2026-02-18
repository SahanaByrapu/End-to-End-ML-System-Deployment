import { useState, useEffect } from 'react';
import { Clock, CheckCircle, XCircle, Loader2 } from 'lucide-react';
import axios from 'axios';
import { format } from 'date-fns';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

const ExperimentsHistory = () => {
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchExperiments();
  }, []);

  const fetchExperiments = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/experiments`);
      setExperiments(response.data);
    } catch (error) {
      console.error('Failed to fetch experiments:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle size={18} className="text-[#22C55E]" />;
      case 'failed':
        return <XCircle size={18} className="text-[#EF4444]" />;
      case 'running':
        return <Loader2 size={18} className="text-[#3B82F6] animate-spin" />;
      default:
        return <Clock size={18} className="text-[#A1A1AA]" />;
    }
  };

  return (
    <div className="p-6 md:p-8 space-y-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-glow" data-testid="experiments-history-title">
          Experiments History
        </h1>
        <p className="text-sm text-[#A1A1AA] mt-2 leading-relaxed">
          View all training experiments with metrics and configurations
        </p>
      </div>

      {/* Experiments Table */}
      <div className="bg-[#0A0A0A] border border-[#27272A] rounded-lg overflow-hidden" data-testid="experiments-table">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 size={32} className="animate-spin text-[#3B82F6]" />
          </div>
        ) : experiments.length === 0 ? (
          <div className="text-center py-12 text-[#A1A1AA]" data-testid="no-experiments-message">
            <p>No experiments yet. Start training to see results here.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-[#18181B] border-b border-[#27272A]">
                <tr>
                  <th className="px-6 py-4 text-left text-xs font-medium text-[#A1A1AA] uppercase tracking-wider">
                    Experiment
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-[#A1A1AA] uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-[#A1A1AA] uppercase tracking-wider">
                    Precision@10
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-[#A1A1AA] uppercase tracking-wider">
                    Recall@10
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-[#A1A1AA] uppercase tracking-wider">
                    F1 Score
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-[#A1A1AA] uppercase tracking-wider">
                    RMSE
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-[#A1A1AA] uppercase tracking-wider">
                    Timestamp
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#27272A]">
                {experiments.map((exp) => (
                  <tr 
                    key={exp.id} 
                    className="hover:bg-[#18181B] transition-colors cursor-pointer group"
                    data-testid={`experiment-row-${exp.id}`}
                  >
                    <td className="px-6 py-4">
                      <div>
                        <p className="text-sm font-medium text-white">{exp.experiment_name}</p>
                        <p className="text-xs text-[#71717A] font-mono mt-1">{exp.id.substring(0, 12)}...</p>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        {getStatusIcon(exp.status)}
                        <span className="text-sm capitalize">{exp.status}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-sm font-mono text-[#22C55E]" data-testid="precision-value">
                      {exp.metrics?.precision_at_10?.toFixed(4) || '—'}
                    </td>
                    <td className="px-6 py-4 text-sm font-mono text-[#22C55E]">
                      {exp.metrics?.recall_at_10?.toFixed(4) || '—'}
                    </td>
                    <td className="px-6 py-4 text-sm font-mono text-[#3B82F6]">
                      {exp.metrics?.f1_score?.toFixed(4) || '—'}
                    </td>
                    <td className="px-6 py-4 text-sm font-mono text-[#F97316]">
                      {exp.metrics?.rmse?.toFixed(4) || '—'}
                    </td>
                    <td className="px-6 py-4 text-sm text-[#A1A1AA]">
                      {exp.timestamp ? format(new Date(exp.timestamp), 'MMM dd, HH:mm') : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default ExperimentsHistory;
