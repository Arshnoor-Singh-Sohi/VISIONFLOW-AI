import React from 'react';
import { Loader2 } from 'lucide-react';

/**
 * LoadingSpinner Component
 * 
 * A reusable loading spinner component that can be sized and customized.
 * Uses Lucide React's Loader2 icon with CSS animations for smooth rotation.
 * 
 * This component is used throughout the application whenever we need to show
 * loading states - during API calls, image processing, model training, etc.
 */
const LoadingSpinner = ({ 
  size = 'md', 
  color = 'blue', 
  text = null, 
  className = '',
  overlay = false 
}) => {
  // Define size variants for consistent spacing throughout the app
  const sizeClasses = {
    'xs': 'w-3 h-3',
    'sm': 'w-4 h-4', 
    'md': 'w-6 h-6',
    'lg': 'w-8 h-8',
    'xl': 'w-12 h-12'
  };

  // Define color variants that match our design system
  const colorClasses = {
    'blue': 'text-blue-600',
    'green': 'text-green-600',
    'red': 'text-red-600',
    'yellow': 'text-yellow-600',
    'purple': 'text-purple-600',
    'gray': 'text-gray-600',
    'white': 'text-white'
  };

  const spinnerSize = sizeClasses[size] || sizeClasses.md;
  const spinnerColor = colorClasses[color] || colorClasses.blue;

  // Create the spinner element with animation
  const spinner = (
    <div className={`flex items-center justify-center ${className}`}>
      <div className="flex items-center space-x-2">
        <Loader2 
          className={`${spinnerSize} ${spinnerColor} animate-spin`}
          aria-hidden="true"
        />
        {text && (
          <span className={`text-sm font-medium ${spinnerColor}`}>
            {text}
          </span>
        )}
      </div>
    </div>
  );

  // If overlay is requested, wrap in a full-screen overlay
  if (overlay) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-6 shadow-lg">
          {spinner}
        </div>
      </div>
    );
  }

  return spinner;
};

/**
 * Specialized loading components for common use cases
 * These provide consistent loading experiences across the app
 */

// Page-level loading spinner for full page loads
export const PageLoadingSpinner = ({ message = "Loading..." }) => (
  <div className="flex items-center justify-center min-h-screen">
    <LoadingSpinner 
      size="xl" 
      color="blue" 
      text={message}
      className="flex-col space-y-4"
    />
  </div>
);

// Card-level loading spinner for sections within pages
export const CardLoadingSpinner = ({ message = "Loading..." }) => (
  <div className="flex items-center justify-center py-12">
    <LoadingSpinner 
      size="lg" 
      color="blue" 
      text={message}
    />
  </div>
);

// Button loading spinner for form submissions
export const ButtonLoadingSpinner = ({ size = 'sm' }) => (
  <LoadingSpinner 
    size={size} 
    color="white" 
    className="mr-2"
  />
);

// Inline loading spinner for small spaces
export const InlineLoadingSpinner = ({ color = 'blue' }) => (
  <LoadingSpinner 
    size="xs" 
    color={color}
  />
);

// Processing status spinner with different states
export const ProcessingSpinner = ({ status = 'processing', size = 'md' }) => {
  const statusConfig = {
    'processing': { color: 'blue', text: 'Processing...' },
    'uploading': { color: 'green', text: 'Uploading...' },
    'analyzing': { color: 'purple', text: 'Analyzing...' },
    'training': { color: 'yellow', text: 'Training...' },
    'completing': { color: 'green', text: 'Completing...' }
  };

  const config = statusConfig[status] || statusConfig.processing;

  return (
    <LoadingSpinner 
      size={size}
      color={config.color}
      text={config.text}
    />
  );
};

export default LoadingSpinner;