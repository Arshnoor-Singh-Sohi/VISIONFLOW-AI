import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

/**
 * React Application Entry Point
 * 
 * This is where the React application bootstraps and mounts to the DOM.
 * We use React 18's createRoot API for better performance and future compatibility.
 * 
 * The application structure flows like this:
 * index.js (this file) → App.js → Individual page components
 */

// Get the root DOM element where React will mount
const container = document.getElementById('root');

// Create the React root using the new React 18 API
// This provides better performance and enables concurrent features
const root = ReactDOM.createRoot(container);

// Render the main App component
// StrictMode helps catch common bugs and deprecated patterns during development
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// Performance monitoring (optional)
// This measures app performance and can send data to analytics
// Learn more: https://bit.ly/CRA-vitals
const reportWebVitals = (metric) => {
  // You can send metrics to your analytics service here
  // For example: analytics.track('Web Vital', metric);
  
  if (process.env.NODE_ENV === 'development') {
    console.log('Web Vital:', metric);
  }
};

// Start measuring performance
// This is useful for understanding app performance in production
if (typeof window !== 'undefined') {
  import('web-vitals').then(({ getCLS, getFID, getFCP, getLCP, getTTFB }) => {
    getCLS(reportWebVitals);
    getFID(reportWebVitals);
    getFCP(reportWebVitals);
    getLCP(reportWebVitals);
    getTTFB(reportWebVitals);
  }).catch((error) => {
    console.warn('Web Vitals could not be loaded:', error);
  });
}

// Handle errors globally in production
if (process.env.NODE_ENV === 'production') {
  window.addEventListener('error', (event) => {
    console.error('Global error caught:', event.error);
    // You could send this to an error tracking service like Sentry
  });

  window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    // You could send this to an error tracking service
  });
}