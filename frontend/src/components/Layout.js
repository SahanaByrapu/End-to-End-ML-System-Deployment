import { Outlet, NavLink } from 'react-router-dom';
import { Activity, GitBranch, BarChart3, Monitor, RefreshCw } from 'lucide-react';

const Layout = () => {
  const navItems = [
    { path: '/', label: 'Training', icon: Activity },
    { path: '/experiments', label: 'Experiments', icon: GitBranch },
    { path: '/comparison', label: 'Comparison', icon: BarChart3 },
    { path: '/monitoring', label: 'Monitoring', icon: Monitor },
    { path: '/retraining', label: 'Retraining', icon: RefreshCw },
  ];

  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <aside className="w-64 bg-[#0A0A0A] border-r border-[#27272A] p-6 flex flex-col">
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-[#3B82F6] text-glow" data-testid="app-title">
            ML Ops Pro
          </h1>
          <p className="text-xs text-[#A1A1AA] mt-1 font-mono">Experiment Tracking</p>
        </div>
        
        <nav className="flex-1 space-y-2" data-testid="main-navigation">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-3 rounded-lg transition-colors duration-200 ${
                  isActive
                    ? 'bg-[#27272A] text-white'
                    : 'text-[#A1A1AA] hover:bg-[#18181B] hover:text-white'
                }`
              }
              data-testid={`nav-${item.label.toLowerCase()}`}
            >
              <item.icon size={20} />
              <span className="font-medium">{item.label}</span>
            </NavLink>
          ))}
        </nav>
        
        <div className="mt-auto pt-4 border-t border-[#27272A]">
          <div className="text-xs text-[#A1A1AA] space-y-1">
            <p className="font-mono">v1.0.0</p>
            <p>ML Recommendation System</p>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
};

export default Layout;
