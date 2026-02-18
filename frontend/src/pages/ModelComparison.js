import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Loader2 } from 'lucide-react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

const ModelComparison = () => {
  const [experiments, setExperiments] = useState([]);
  const [selectedIds, setSelectedIds] = useState([]);
  const [comparisonData, setComparisonData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchExperiments();
  }, []);

  const fetchExperiments = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/experiments`);
      const completed = response.data.filter(exp => exp.status === 'completed');
      setExperiments(completed);
      if (completed.length > 0) {
        const defaultSelection = completed.slice(0, 3).map(exp => exp.id);
        setSelectedIds(defaultSelection);
      }
    } catch (error) {
      console.error('Failed to fetch experiments:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedIds.length > 0) {
      fetchComparison();
    }
  }, [selectedIds]);

  const fetchComparison = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/metrics/comparison?experiment_ids=${selectedIds.join(',')}`);
      setComparisonData(response.data);
    } catch (error) {
      console.error('Failed to fetch comparison:', error);
    }
  };

  const toggleSelection = (id) => {
    setSelectedIds(prev => 
      prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]
    );
  };

  const getChartData = () => {
    if (!comparisonData?.experiments) return [];
    return comparisonData.experiments.map(exp => ({
      name: exp.experiment_name.substring(0, 15),
      Precision: exp.metrics?.precision_at_10 || 0,
      Recall: exp.metrics?.recall_at_10 || 0,
      F1: exp.metrics?.f1_score || 0,
      RMSE: exp.metrics?.rmse || 0,
    }));
  };

  return (
    <div className="p-6 md:p-8 space-y-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-glow" data-testid="model-comparison-title">
          Model Comparison
        </h1>
        <p className="text-sm text-[#A1A1AA] mt-2 leading-relaxed">
          Compare metrics across different experiments side-by-side
        </p>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 size={32} className="animate-spin text-[#3B82F6]" />
        </div>
      ) : (
        <>
          {/* Experiment Selection */}
          <div className="bg-[#0A0A0A] border border-[#27272A] rounded-lg p-6" data-testid="experiment-selector">
            <h2 className="text-lg font-semibold mb-4">Select Experiments to Compare</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {experiments.map(exp => (
                <label 
                  key={exp.id} 
                  className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${
                    selectedIds.includes(exp.id)
                      ? 'bg-[#3B82F6]/10 border-[#3B82F6]'
                      : 'bg-[#18181B] border-[#27272A] hover:border-[#3B82F6]/50'
                  }`}
                  data-testid={`experiment-checkbox-${exp.id}`}
                >
                  <input
                    type="checkbox"
                    checked={selectedIds.includes(exp.id)}
                    onChange={() => toggleSelection(exp.id)}
                    className="w-4 h-4 accent-[#3B82F6]"
                  />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">{exp.experiment_name}</p>
                    <p className="text-xs text-[#71717A] font-mono truncate">{exp.id.substring(0, 12)}...</p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Charts */}
          {selectedIds.length > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Bar Chart - Metrics Comparison */}
              <div className="bg-[#0A0A0A] border border-[#27272A] rounded-lg p-6" data-testid="metrics-bar-chart">
                <h2 className="text-lg font-semibold mb-4">Metrics Comparison</h2>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={getChartData()}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#27272A" />
                    <XAxis dataKey="name" stroke="#A1A1AA" style={{ fontSize: '12px' }} />
                    <YAxis stroke="#A1A1AA" style={{ fontSize: '12px' }} />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#0A0A0A', 
                        border: '1px solid #27272A',
                        borderRadius: '8px',
                        color: '#FAFAFA'
                      }} 
                    />
                    <Legend wrapperStyle={{ color: '#A1A1AA' }} />
                    <Bar dataKey="Precision" fill="#22C55E" />
                    <Bar dataKey="Recall" fill="#3B82F6" />
                    <Bar dataKey="F1" fill="#8B5CF6" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* RMSE Comparison */}
              <div className="bg-[#0A0A0A] border border-[#27272A] rounded-lg p-6" data-testid="rmse-chart">
                <h2 className="text-lg font-semibold mb-4">RMSE Comparison (Lower is Better)</h2>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={getChartData()}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#27272A" />
                    <XAxis dataKey="name" stroke="#A1A1AA" style={{ fontSize: '12px' }} />
                    <YAxis stroke="#A1A1AA" style={{ fontSize: '12px' }} />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#0A0A0A', 
                        border: '1px solid #27272A',
                        borderRadius: '8px',
                        color: '#FAFAFA'
                      }} 
                    />
                    <Legend wrapperStyle={{ color: '#A1A1AA' }} />
                    <Bar dataKey="RMSE" fill="#F97316" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Metrics Table */}
          {comparisonData?.experiments && comparisonData.experiments.length > 0 && (
            <div className="bg-[#0A0A0A] border border-[#27272A] rounded-lg overflow-hidden" data-testid="comparison-table">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-[#18181B] border-b border-[#27272A]">
                    <tr>
                      <th className="px-6 py-4 text-left text-xs font-medium text-[#A1A1AA] uppercase tracking-wider">
                        Experiment
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
                        NDCG@10
                      </th>
                      <th className="px-6 py-4 text-left text-xs font-medium text-[#A1A1AA] uppercase tracking-wider">
                        RMSE
                      </th>
                      <th className="px-6 py-4 text-left text-xs font-medium text-[#A1A1AA] uppercase tracking-wider">
                        MAE
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#27272A]">
                    {comparisonData.experiments.map((exp) => (
                      <tr key={exp.id} className="hover:bg-[#18181B] transition-colors">
                        <td className="px-6 py-4 text-sm font-medium">{exp.experiment_name}</td>
                        <td className="px-6 py-4 text-sm font-mono text-[#22C55E]">
                          {exp.metrics?.precision_at_10?.toFixed(4) || '—'}
                        </td>
                        <td className="px-6 py-4 text-sm font-mono text-[#22C55E]">
                          {exp.metrics?.recall_at_10?.toFixed(4) || '—'}
                        </td>
                        <td className="px-6 py-4 text-sm font-mono text-[#3B82F6]">
                          {exp.metrics?.f1_score?.toFixed(4) || '—'}
                        </td>
                        <td className="px-6 py-4 text-sm font-mono text-[#8B5CF6]">
                          {exp.metrics?.ndcg_at_10?.toFixed(4) || '—'}
                        </td>
                        <td className="px-6 py-4 text-sm font-mono text-[#F97316]">
                          {exp.metrics?.rmse?.toFixed(4) || '—'}
                        </td>
                        <td className="px-6 py-4 text-sm font-mono text-[#F97316]">
                          {exp.metrics?.mae?.toFixed(4) || '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default ModelComparison;
