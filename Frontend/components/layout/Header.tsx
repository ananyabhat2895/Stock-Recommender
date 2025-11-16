import React from 'react';
import { AppLogo, SunIcon, MoonIcon } from '../../constants';
import Button from '../ui/Button';

interface HeaderProps {
  isDarkMode: boolean;
  toggleDarkMode: () => void;
}

const Header: React.FC<HeaderProps> = ({ isDarkMode, toggleDarkMode }) => {
  return (
    <header className="py-4 border-b border-foreground/10 bg-card/50 backdrop-blur-lg sticky top-0 z-50">
      <div className="container mx-auto px-4 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <AppLogo className="h-8 w-8 text-primary" />
        </div>
        <Button onClick={toggleDarkMode} variant="ghost" size="icon" aria-label="Toggle dark mode">
          {isDarkMode ? <SunIcon className="h-5 w-5" /> : <MoonIcon className="h-5 w-5" />}
        </Button>
      </div>
    </header>
  );
};

export default Header;