import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  Upload, Eye, Brain, Activity, Menu, X, 
  User, Settings, LogOut, HelpCircle
} from 'lucide-react';

const Navigation = ({ systemHealth, currentUser }) => {
  const location = useLocation();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false);

  const navigationItems = [
    {
      name: 'Upload',
      path: '/upload',
      icon: Upload,
      description: 'Upload and process images'
    },
    {
      name: 'Results',
      path: '/results',
      icon: Eye,
      description: 'View processing results'
    },
    {
      name: 'Training',
      path: '/training',
      icon: Brain,
      description: 'Model training dashboard'
    },
    {
      name: 'Monitoring',
      path: '/monitoring',
      icon: Activity,
      description: 'System performance monitoring'
    }
  ];

  const isActivePath = (path) => {
    return location.pathname === path || location.pathname.startsWith(path + '/');
  };

  const getHealthIndicator = () => {
    const colors = {
      healthy: 'bg-green-500',
      degraded: 'bg-yellow-500',
      unhealthy: 'bg-red-500',
      unknown: 'bg-gray-500'
    };
    
    return (
      <div className={`w-2 h-2 rounded-full ${colors[systemHealth] || colors.unknown}`}></div>
    );
  };

  const closeMobileMenu = () => {
    setIsMobileMenuOpen(false);
  };

  const closeUserMenu = () => {
    setIsUserMenuOpen(false);
  };

  return (
    <nav className="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo and Brand */}
          <div className="flex items-center">
            <Link 
              to="/" 
              className="flex items-center space-x-3"
              onClick={closeMobileMenu}
            >
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <Eye className="w-5 h-5 text-white" />
              </div>
              <div className="hidden sm:block">
                <span className="text-xl font-bold text-gray-900">VisionFlow</span>
                <span className="text-sm text-gray-500 ml-2">AI</span>
              </div>
            </Link>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-1">
            {navigationItems.map((item) => {
              const Icon = item.icon;
              const isActive = isActivePath(item.path);
              
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`
                    group relative px-4 py-2 rounded-lg transition-all duration-200
                    ${isActive 
                      ? 'bg-blue-50 text-blue-700' 
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                    }
                  `}
                  title={item.description}
                >
                  <div className="flex items-center space-x-2">
                    <Icon className={`w-4 h-4 ${isActive ? 'text-blue-600' : ''}`} />
                    <span className="font-medium">{item.name}</span>
                  </div>
                  
                  {/* Active indicator */}
                  {isActive && (
                    <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 w-1 h-1 bg-blue-600 rounded-full"></div>
                  )}
                </Link>
              );
            })}
          </div>

          {/* Right side items */}
          <div className="flex items-center space-x-4">
            {/* System Health Indicator */}
            <div className="hidden sm:flex items-center space-x-2">
              {getHealthIndicator()}
              <span className="text-sm text-gray-600 capitalize">
                {systemHealth}
              </span>
            </div>

            {/* User Menu */}
            <div className="relative">
              <button
                onClick={() => setIsUserMenuOpen(!isUserMenuOpen)}
                className="flex items-center space-x-2 p-2 rounded-lg hover:bg-gray-50 transition-colors"
              >
                {currentUser ? (
                  <>
                    <div className="w-8 h-8 bg-gray-300 rounded-full flex items-center justify-center">
                      <span className="text-sm font-medium text-gray-700">
                        {currentUser.name?.charAt(0) || 'U'}
                      </span>
                    </div>
                    <span className="hidden sm:block text-sm font-medium text-gray-700">
                      {currentUser.name || 'User'}
                    </span>
                  </>
                ) : (
                  <User className="w-5 h-5 text-gray-600" />
                )}
              </button>

              {/* User Dropdown Menu */}
              {isUserMenuOpen && (
                <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg border border-gray-200 py-1 z-50">
                  <div className="px-4 py-2 border-b border-gray-100">
                    <p className="text-sm font-medium text-gray-900">
                      {currentUser?.name || 'Guest User'}
                    </p>
                    <p className="text-xs text-gray-500">
                      {currentUser?.email || 'guest@visionflow.ai'}
                    </p>
                  </div>
                  
                  <a
                    href="#settings"
                    className="flex items-center space-x-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                    onClick={closeUserMenu}
                  >
                    <Settings className="w-4 h-4" />
                    <span>Settings</span>
                  </a>
                  
                  <a
                    href="/api/v1/docs"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center space-x-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                    onClick={closeUserMenu}
                  >
                    <HelpCircle className="w-4 h-4" />
                    <span>API Docs</span>
                  </a>
                  
                  <div className="border-t border-gray-100">
                    <button
                      className="flex items-center space-x-2 w-full px-4 py-2 text-sm text-red-600 hover:bg-red-50"
                      onClick={() => {
                        closeUserMenu();
                        // Handle logout if implemented
                        console.log('Logout clicked');
                      }}
                    >
                      <LogOut className="w-4 h-4" />
                      <span>Logout</span>
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Mobile menu button */}
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="md:hidden p-2 rounded-lg hover:bg-gray-50 transition-colors"
            >
              {isMobileMenuOpen ? (
                <X className="w-5 h-5 text-gray-600" />
              ) : (
                <Menu className="w-5 h-5 text-gray-600" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Navigation Menu */}
        {isMobileMenuOpen && (
          <div className="md:hidden border-t border-gray-200 py-2">
            <div className="space-y-1">
              {navigationItems.map((item) => {
                const Icon = item.icon;
                const isActive = isActivePath(item.path);
                
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`
                      flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors
                      ${isActive 
                        ? 'bg-blue-50 text-blue-700 border-l-4 border-blue-600' 
                        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                      }
                    `}
                    onClick={closeMobileMenu}
                  >
                    <Icon className={`w-5 h-5 ${isActive ? 'text-blue-600' : ''}`} />
                    <div>
                      <div className="font-medium">{item.name}</div>
                      <div className="text-xs text-gray-500">{item.description}</div>
                    </div>
                  </Link>
                );
              })}
            </div>

            {/* Mobile System Status */}
            <div className="mt-4 px-4 py-3 border-t border-gray-200">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-700">System Status</span>
                <div className="flex items-center space-x-2">
                  {getHealthIndicator()}
                  <span className="text-sm text-gray-600 capitalize">
                    {systemHealth}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Mobile menu overlay */}
      {isMobileMenuOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-25 z-30 md:hidden"
          onClick={closeMobileMenu}
        ></div>
      )}

      {/* User menu overlay */}
      {isUserMenuOpen && (
        <div 
          className="fixed inset-0 z-30"
          onClick={closeUserMenu}
        ></div>
      )}
    </nav>
  );
};

export default Navigation;