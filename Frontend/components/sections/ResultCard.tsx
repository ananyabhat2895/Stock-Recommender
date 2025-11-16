
import React, { useState } from 'react';
import type { FormData, GeminiResponse } from '../../types';
import Card from '../ui/Card';
import Button from '../ui/Button';
import DonutChart from '../ui/DonutChart';
import { CopyIcon, CheckIcon } from '../../constants';

interface ResultCardProps {
  formData: FormData;
  result: GeminiResponse;
}

const ResultCard: React.FC<ResultCardProps> = ({ formData, result }) => {
  const [copied, setCopied] = useState(false);

  const riskProfileMap = {
      Low: { value: 25, color: 'bg-green-500', label: 'Conservative' },
      Medium: { value: 60, color: 'bg-yellow-500', label: 'Balanced' },
      High: { value: 95, color: 'bg-red-500', label: 'Aggressive' },
  }

  const riskProfile = riskProfileMap[formData.riskTolerance];

  const handleCopy = () => {
    navigator.clipboard.writeText(JSON.stringify({ profile: formData, recommendation: result }, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  
  return (
    <Card className="p-6 md:p-8">
      <h3 className="text-2xl font-bold text-center mb-2">Your AI-Generated Insights</h3>
      <p className="text-center text-foreground/70 mb-6">{result.summary}</p>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 md:gap-8">
        
        {/* Left Panel: Chart and Risk */}
        <div className="md:col-span-1 flex flex-col items-center gap-6">
            <h4 className="font-semibold text-lg">Recommended Distribution</h4>
            <DonutChart data={result.portfolio} />
            <div className="w-full">
                <h4 className="font-semibold text-lg mb-2 text-center">Risk Profile: <span className="font-bold" style={{color: riskProfile.color.replace('bg-', '').replace('-500', '')}}>{riskProfile.label}</span></h4>
                <div className="w-full bg-foreground/10 rounded-full h-3">
                    <div className={`h-full rounded-full ${riskProfile.color}`} style={{ width: `${riskProfile.value}%` }}></div>
                </div>
            </div>
        </div>

        {/* Right Panel: Details and JSON */}
        <div className="md:col-span-2">
            <ul className="space-y-2 mb-6">
                {result.portfolio.map((item, index) => (
                    <li key={index} className="flex justify-between items-center text-sm p-2 rounded-md bg-background">
                        <span>{item.name}</span>
                        <span className="font-bold text-primary">{item.percentage}%</span>
                    </li>
                ))}
            </ul>

            <div className="relative">
                <Button onClick={handleCopy} variant="ghost" size="sm" className="absolute top-2 right-2">
                    {copied ? <CheckIcon className="h-4 w-4 text-green-500"/> : <CopyIcon className="h-4 w-4" />}
                    <span className="ml-1">{copied ? 'Copied!' : 'Copy JSON'}</span>
                </Button>
                <pre className="text-xs bg-background p-4 rounded-md overflow-x-auto">
                    <code>{JSON.stringify({ profile: formData, recommendation: result }, null, 2)}</code>
                </pre>
            </div>
        </div>
      </div>
    </Card>
  );
};

export default ResultCard;
