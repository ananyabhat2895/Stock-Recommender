
import React from 'react';
import type { PortfolioDistribution } from '../../types';

interface DonutChartProps {
  data: PortfolioDistribution[];
}

const DonutChart: React.FC<DonutChartProps> = ({ data }) => {
  const colors = ['hsl(var(--color-primary))', 'hsl(var(--color-secondary))', 'hsl(var(--color-accent))', '#34d399', '#f87171', '#fbbf24'];
  const radius = 80;
  const circumference = 2 * Math.PI * radius;
  let accumulatedPercentage = 0;

  return (
    <div className="relative w-48 h-48 md:w-56 md:h-56">
      <svg viewBox="0 0 200 200" className="transform -rotate-90">
        {data.map((item, index) => {
          const strokeDashoffset = circumference - (accumulatedPercentage / 100) * circumference;
          const strokeDasharray = `${(item.percentage / 100) * circumference} ${circumference}`;
          accumulatedPercentage += item.percentage;
          return (
            <circle
              key={index}
              cx="100"
              cy="100"
              r={radius}
              fill="transparent"
              stroke={colors[index % colors.length]}
              strokeWidth="20"
              strokeDasharray={strokeDasharray}
              strokeDashoffset={strokeDashoffset}
              className="transition-all duration-500"
            />
          );
        })}
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-xl font-bold text-foreground">Portfolio</span>
      </div>
    </div>
  );
};

export default DonutChart;
