import { BrowserRouter, Routes, Route } from 'react-router-dom';
import '@/App.css';
import { Toaster } from 'sonner';
import Layout from '@/components/Layout';
import TrainingDashboard from '@/pages/TrainingDashboard';
import ExperimentsHistory from '@/pages/ExperimentsHistory';
import ModelComparison from '@/pages/ModelComparison';
import MonitoringDashboard from '@/pages/MonitoringDashboard';
import RetrainingManagement from '@/pages/RetrainingManagement';

function App() {
  return (
    <div className="App min-h-screen bg-[#09090B] text-[#FAFAFA]">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<TrainingDashboard />} />
            <Route path="experiments" element={<ExperimentsHistory />} />
            <Route path="comparison" element={<ModelComparison />} />
            <Route path="monitoring" element={<MonitoringDashboard />} />
            <Route path="retraining" element={<RetrainingManagement />} />
          </Route>
        </Routes>
      </BrowserRouter>
      <Toaster position="top-right" theme="dark" />
    </div>
  );
}

export default App;
