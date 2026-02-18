const MetricCard = ({ title, value, trend, icon: Icon, subtitle, status = 'neutral' }) => {
  const statusColors = {
    success: 'text-[#22C55E]',
    error: 'text-[#EF4444]',
    warning: 'text-[#F97316]',
    neutral: 'text-[#A1A1AA]',
  };

  return (
    <div 
      className="bg-[#0A0A0A] border border-[#27272A] hover:border-[#3B82F6]/50 rounded-lg p-5 transition-all duration-300"
      data-testid={`metric-card-${title.toLowerCase().replace(/\s+/g, '-')}`}
    >
      <div className="flex items-start justify-between mb-3">
        <div>
          <p className="text-xs text-[#A1A1AA] uppercase tracking-wider font-medium">{title}</p>
          {subtitle && <p className="text-xs text-[#71717A] mt-1">{subtitle}</p>}
        </div>
        {Icon && (
          <div className="p-2 bg-[#18181B] rounded-lg">
            <Icon size={18} className="text-[#3B82F6]" />
          </div>
        )}
      </div>
      <div className="flex items-end justify-between">
        <h3 className={`text-3xl font-bold font-[Space_Grotesk] ${statusColors[status]}`} data-testid="metric-value">
          {value}
        </h3>
        {trend && (
          <span className={`text-sm font-mono ${trend > 0 ? 'text-[#22C55E]' : 'text-[#EF4444]'}`}>
            {trend > 0 ? '+' : ''}{trend}%
          </span>
        )}
      </div>
    </div>
  );
};

export default MetricCard;
