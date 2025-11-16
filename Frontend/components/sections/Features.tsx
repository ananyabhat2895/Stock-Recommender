import React from 'react';
import Card from '../ui/Card';

const features = [
  {
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
    title: 'Data-Driven Stock Intelligence',
    description: 'Harness advanced algorithms to evaluate market conditions, stock sentiment, and financial indicators for accurate recommendations.',
  },
  {
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 16v-2m0-8v-2m0 16V4m6 6h2m-16 0h2m14 0h2M4 12H2m18 0h2M12 2v2m0 16v2m-6-6H4m16 0h-2m-2-8l-2-2m-8 8l-2-2m10 10l2 2m-12-2l-2 2" />
      </svg>
    ),
    title: 'Personalized Portfolio Alignment',
    description: 'Your risk profile, time horizon, and investment preferences drive the portfolio suggestions we generate for you.',
  },
  {
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
      </svg>
    ),
    title: 'Fast, Secure, and Reliable',
    description: 'Receive insights in seconds. Your data is processed securely and never stored, ensuring complete privacy.',
  },
];

const Features: React.FC = () => {
  return (
    <section className="py-16 md:py-24 animate-fadeIn" style={{ animationDelay: '200ms' }}>
      <div className="text-center">
        <h2 className="text-3xl md:text-4xl font-bold">Smarter Insights for Confident Investing</h2>
        <p className="mt-3 max-w-xl mx-auto text-foreground/70">
          Our platform simplifies decision-making through advanced analytics, personalized portfolio suggestions, and fast, secure processing.
        </p>
      </div>
      <div className="mt-12 grid gap-8 md:grid-cols-3">
        {features.map((feature, index) => (
          <Card key={index} className="text-center p-8">
            <div className="inline-block bg-primary/10 p-4 rounded-full">
              {feature.icon}
            </div>
            <h3 className="mt-4 text-xl font-semibold">{feature.title}</h3>
            <p className="mt-2 text-foreground/70">{feature.description}</p>
          </Card>
        ))}
      </div>
    </section>
  );
};

export default Features;