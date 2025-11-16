import React from 'react';
import Button from '../ui/Button';

const HeroIllustration = () => (
    <svg viewBox="0 0 500 300" className="w-full h-auto max-w-lg">
        <defs>
            <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="hsl(var(--color-primary))" />
                <stop offset="100%" stopColor="hsl(var(--color-accent))" />
            </linearGradient>
            <linearGradient id="grad2" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="hsl(var(--color-secondary))" />
                <stop offset="100%" stopColor="hsl(var(--color-primary))" />
            </linearGradient>
        </defs>
        <rect x="50" y="50" width="400" height="200" rx="20" fill="url(#grad1)" opacity="0.1" />
        <path d="M100 200 C 150 120, 200 220, 250 150 S 350 50, 400 130" stroke="url(#grad2)" strokeWidth="4" fill="none" strokeLinecap="round" />
        <circle cx="100" cy="200" r="8" fill="hsl(var(--color-secondary))" />
        <circle cx="400" cy="130" r="8" fill="hsl(var(--color-primary))" />
        <g transform="translate(350, 60)">
            <circle cx="0" cy="0" r="30" fill="url(#grad1)" opacity="0.3" />
            <path d="M -10 -5 L 0 10 L 10 -5" stroke="hsl(var(--color-foreground))" strokeWidth="2" fill="none" />
        </g>
        <rect x="80" y="80" width="100" height="10" rx="5" fill="hsl(var(--color-foreground))" opacity="0.2" />
        <rect x="80" y="100" width="150" height="10" rx="5" fill="hsl(var(--color-foreground))" opacity="0.2" />
        <rect x="80" y="120" width="120" height="10" rx="5" fill="hsl(var(--color-foreground))" opacity="0.2" />
    </svg>
);


const Hero: React.FC = () => {
  const scrollToForm = () => {
    document.getElementById('insight-form')?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <section className="text-center py-16 md:py-24 animate-fadeIn">
      <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight">
        User-Centric Stock Insights & <span className="text-primary">Portfolio Recommendation System</span>
      </h1>
      <p className="mt-4 max-w-2xl mx-auto text-lg md:text-xl text-foreground/70">
        Leverage intelligent analytics to tailor your investment journey. Our system evaluates your financial goals, risk behavior, and preferences to generate actionable, data-driven stock insights.
      </p>
      <div className="mt-8 flex justify-center gap-4">
        <Button onClick={scrollToForm} size="lg">
          Generate Insights
        </Button>
        <Button onClick={scrollToForm} variant="outline" size="lg">
          Learn More
        </Button>
      </div>
      <div className="mt-12 flex justify-center">
        <HeroIllustration />
      </div>
    </section>
  );
};

export default Hero;