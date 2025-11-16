import React, { useState, useEffect } from 'react';
import Header from './components/layout/Header';
import Hero from './components/sections/Hero';
import Features from './components/sections/Features';
import InsightForm from './components/sections/InsightForm';
import Footer from './components/layout/Footer';
import { AppLogo } from './constants';

const App: React.FC = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [showApp, setShowApp] = useState(false);

  useEffect(() => {
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    setIsDarkMode(prefersDark);
  }, []);

  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);
  
  // Fake loader to improve perceived performance and initial aesthetic
  useEffect(() => {
    const timer = setTimeout(() => setShowApp(true), 1200);
    return () => clearTimeout(timer);
  }, []);

  const toggleDarkMode = () => {
    setIsDarkMode(prev => !prev);
  };
  
  if (!showApp) {
    return (
      <div className="w-full h-screen flex flex-col items-center justify-center bg-background text-foreground transition-colors duration-500">
        <div className="animate-pulse">
            <AppLogo className="h-16 w-16 text-primary" />
        </div>
        <p className="mt-4 text-sm text-foreground/70">Crafting your financial future...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background font-sans text-foreground transition-colors duration-500">
      <div className="absolute top-0 left-0 w-full h-full z-0">
        <div className="absolute inset-0 bg-grid-slate-300/[0.2] dark:bg-grid-slate-700/[0.2] [mask-image:linear-gradient(to_bottom,white_20%,transparent_100%)]"></div>
      </div>
      <div className="relative z-10">
        <Header isDarkMode={isDarkMode} toggleDarkMode={toggleDarkMode} />
        <main className="container mx-auto px-4 py-8 md:py-16">
          <Hero />
          <Features />
          <InsightForm />
        </main>
        <Footer />
      </div>
    </div>
  );
};

export default App;