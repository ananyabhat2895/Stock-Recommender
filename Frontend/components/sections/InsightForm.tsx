import React, { useState, useCallback, useMemo } from 'react';
import type { FormData, GeminiResponse, MarketCap, RiskTolerance, Asset, TimeHorizonUnit } from '../../types';
import { MARKET_CAP_OPTIONS, RISK_TOLERANCE_OPTIONS, ASSET_PREFERENCES, InfoIcon, ChevronDownIcon, XIcon } from '../../constants';
import { generateInsights } from '../../services/geminiService';
import Button from '../ui/Button';
import Card from '../ui/Card';
import Tooltip from '../ui/Tooltip';
import ResultCard from './ResultCard';

const InsightForm: React.FC = () => {
  const [formData, setFormData] = useState<FormData>({
    initialAmount: 10000,
    timeHorizonValue: 5,
    timeHorizonUnit: 'Years',
    specificDate: '',
    marketCap: 'Mid Cap',
    riskTolerance: 'Medium',
    assets: ['Equity', 'Mutual Funds'],
  });
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<GeminiResponse | null>(null);
  
  const handleInputChange = useCallback(<T extends keyof FormData>(field: T, value: FormData[T]) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  }, []);

  const handleAssetToggle = useCallback((asset: Asset) => {
    setFormData(prev => {
      const newAssets = prev.assets.includes(asset)
        ? prev.assets.filter(a => a !== asset)
        : [...prev.assets, asset];
      
      if (newAssets.length > 3) return prev; // Limit to 3
      return { ...prev, assets: newAssets };
    });
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (formData.initialAmount < 1000) {
      setError("Initial amount must be at least ₹1,000.");
      return;
    }
    setError(null);
    setLoading(true);
    setResult(null);
    try {
      const apiResult = await generateInsights(formData);
      setResult(apiResult);
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred.');
    } finally {
      setLoading(false);
    }
  };

  const riskMeterStyle = useMemo(() => {
    const riskMap = { Low: 25, Medium: 60, High: 95 };
    const percentage = riskMap[formData.riskTolerance];
    return {
      width: `${percentage}%`,
      background: `linear-gradient(90deg, hsl(100, 70%, 50%), hsl(50, 80%, 50%) ${percentage}%, hsl(0, 80%, 50%) 100%)`
    };
  }, [formData.riskTolerance]);

  return (
    <section id="insight-form" className="py-16 md:py-24 animate-fadeIn" style={{ animationDelay: '400ms' }}>
      <div className="text-center mb-12">
        <h2 className="text-3xl md:text-4xl font-bold">Build Your Investment Profile</h2>
        <p className="mt-3 max-w-xl mx-auto text-foreground/70">
          Enter your details below, and our system will generate a personalized portfolio recommendation and stock insight report.
        </p>
      </div>
      <Card className="max-w-4xl mx-auto p-6 md:p-8">
        <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-6">
          
          {/* Initial Net Amount */}
          <div className="md:col-span-1">
            <label className="flex items-center text-sm font-medium text-foreground/80 mb-2">
              Initial Net Amount (₹)
              <Tooltip text="Amount invested initially. Should be greater than ₹1,000.">
                <InfoIcon className="h-4 w-4 ml-1 text-foreground/50" />
              </Tooltip>
            </label>
            <div className="relative">
              <span className="absolute inset-y-0 left-3 flex items-center text-foreground/50">₹</span>
              <input 
                type="text"
                value={formData.initialAmount.toLocaleString('en-IN')}
                onChange={e => {
                    const value = parseInt(e.target.value.replace(/,/g, ''), 10);
                    if (!isNaN(value)) handleInputChange('initialAmount', value);
                    else handleInputChange('initialAmount', 0);
                }}
                className="w-full pl-7 pr-4 py-2 bg-background border border-foreground/20 rounded-md focus:ring-2 focus:ring-primary focus:border-primary transition-all"
                placeholder="10,000"
              />
            </div>
             {formData.initialAmount < 1000 && <p className="text-red-500 text-xs mt-1">Amount must be at least ₹1,000</p>}
          </div>

          {/* Time Horizon */}
          <div className="md:col-span-1">
            <label className="text-sm font-medium text-foreground/80 mb-2 block">Time Horizon</label>
            <div className="flex gap-2">
                <div className="relative flex-grow">
                    <select
                        value={formData.timeHorizonValue}
                        onChange={e => handleInputChange('timeHorizonValue', parseInt(e.target.value))}
                        disabled={formData.timeHorizonUnit === 'Specific Date'}
                        className="w-full appearance-none pr-8 py-2 pl-3 bg-background border border-foreground/20 rounded-md focus:ring-2 focus:ring-primary focus:border-primary transition-all disabled:opacity-50"
                    >
                        {Array.from({ length: 50 }, (_, i) => i + 1).map(v => <option key={v} value={v}>{v}</option>)}
                    </select>
                    <ChevronDownIcon className="absolute right-2 top-1/2 -translate-y-1/2 h-5 w-5 text-foreground/50 pointer-events-none"/>
                </div>
                <div className="relative">
                    <select
                        value={formData.timeHorizonUnit}
                        onChange={e => handleInputChange('timeHorizonUnit', e.target.value as TimeHorizonUnit)}
                        className="w-full appearance-none pr-8 py-2 pl-3 bg-background border border-foreground/20 rounded-md focus:ring-2 focus:ring-primary focus:border-primary transition-all"
                    >
                        <option>Months</option>
                        <option>Years</option>
                        <option>Specific Date</option>
                    </select>
                    <ChevronDownIcon className="absolute right-2 top-1/2 -translate-y-1/2 h-5 w-5 text-foreground/50 pointer-events-none"/>
                </div>
            </div>
            {formData.timeHorizonUnit === 'Specific Date' && (
                <div className="mt-2 animate-slideDown">
                    <input 
                        type="date"
                        value={formData.specificDate}
                        onChange={e => handleInputChange('specificDate', e.target.value)}
                        className="w-full p-2 bg-background border border-foreground/20 rounded-md focus:ring-2 focus:ring-primary focus:border-primary"
                        min={new Date().toISOString().split('T')[0]}
                    />
                    <p className="text-xs text-foreground/60 mt-1">Selecting a date overrides the numeric time horizon.</p>
                </div>
            )}
          </div>
          
          {/* Market Cap */}
          <div className="md:col-span-1">
            <label className="text-sm font-medium text-foreground/80 mb-2 block">Market Cap Preference</label>
            <div className="relative">
                <select
                    value={formData.marketCap}
                    onChange={e => handleInputChange('marketCap', e.target.value as MarketCap)}
                    className="w-full appearance-none pr-8 py-2 pl-3 bg-background border border-foreground/20 rounded-md focus:ring-2 focus:ring-primary focus:border-primary"
                >
                    {MARKET_CAP_OPTIONS.map(opt => <option key={opt.value} value={opt.value}>{opt.value}</option>)}
                </select>
                <ChevronDownIcon className="absolute right-2 top-1/2 -translate-y-1/2 h-5 w-5 text-foreground/50 pointer-events-none"/>
            </div>
            <div className="flex items-center mt-2 text-xs text-foreground/60">
              {MARKET_CAP_OPTIONS.find(o => o.value === formData.marketCap)?.icon}
              <span className="ml-2">{MARKET_CAP_OPTIONS.find(o => o.value === formData.marketCap)?.description}</span>
            </div>
          </div>

          {/* Risk Tolerance */}
          <div className="md:col-span-1">
             <label className="flex items-center text-sm font-medium text-foreground/80 mb-2">
              Risk Tolerance
              <Tooltip text="Your comfort level with potential investment losses in exchange for potential gains.">
                <InfoIcon className="h-4 w-4 ml-1 text-foreground/50" />
              </Tooltip>
            </label>
             <div className="flex gap-2 mb-2">
                {RISK_TOLERANCE_OPTIONS.map(opt => (
                    <button
                        key={opt}
                        type="button"
                        onClick={() => handleInputChange('riskTolerance', opt)}
                        className={`flex-1 py-2 text-sm rounded-md transition-all ${formData.riskTolerance === opt ? 'bg-primary text-white font-semibold' : 'bg-background hover:bg-foreground/10 border border-foreground/20'}`}
                    >
                        {opt}
                    </button>
                ))}
             </div>
             <div className="w-full bg-foreground/10 rounded-full h-2.5 overflow-hidden">
                <div className="h-full rounded-full transition-all duration-500 ease-out" style={riskMeterStyle}></div>
             </div>
          </div>

          {/* Asset Preferences */}
          <div className="md:col-span-2">
            <label className="text-sm font-medium text-foreground/80 mb-2 block">Asset Preferences (Max 3)</label>
            <div className="flex flex-wrap gap-2 mb-2">
                {formData.assets.map(asset => (
                    <span key={asset} className="flex items-center bg-primary/10 text-primary text-sm font-medium px-2 py-1 rounded-full">
                        {asset}
                        <button type="button" onClick={() => handleAssetToggle(asset)} className="ml-1.5">
                            <XIcon className="h-3 w-3" />
                        </button>
                    </span>
                ))}
            </div>
            <div className="flex flex-wrap gap-2">
                {ASSET_PREFERENCES.map(opt => (
                    !formData.assets.includes(opt) && (
                        <button
                            key={opt}
                            type="button"
                            onClick={() => handleAssetToggle(opt)}
                            className="px-3 py-1.5 text-sm rounded-md transition-all bg-background hover:bg-foreground/10 border border-foreground/20 disabled:opacity-50 disabled:cursor-not-allowed"
                            disabled={formData.assets.length >= 3}
                        >
                           + {opt}
                        </button>
                    )
                ))}
            </div>
          </div>
          
          <div className="md:col-span-2 text-center mt-4">
            <Button type="submit" size="lg" className="w-full md:w-auto" disabled={loading}>
              {loading ? 'Generating...' : 'Generate Insights'}
            </Button>
            {error && <p className="text-red-500 text-sm mt-4">{error}</p>}
          </div>
        </form>
      </Card>

      {loading && (
        <div className="text-center mt-8">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          <p className="mt-2 text-foreground/70">Our AI is analyzing your profile...</p>
        </div>
      )}

      {result && !loading && (
        <div className="mt-12 animate-fadeIn">
            <ResultCard formData={formData} result={result} />
        </div>
      )}
    </section>
  );
};

export default InsightForm;