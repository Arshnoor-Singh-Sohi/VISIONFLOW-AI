import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Toaster } from 'react-hot-toast';
import Navigation from './components/Navigation';
import ImageUpload from './components/ImageUpload';
import ResultsViewer from './components/ResultsViewer';
import TrainingDashboard from './components/TrainingDashboard';
import SystemMonitoring from './components/SystemMonitoring';
import { apiService } from './services/api';
import { websocketService } from './services/websocket';
import LoadingSpinner from './components/LoadingSpinner';
import './index.css';

// Create a client for React Query
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 2,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

function App() {
  const [systemHealth, setSystemHealth] = useState('unknown');
  const [notifications, setNotifications] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);

  // Initialize WebSocket connection and system monitoring
  useEffect(() => {
    const initializeApp = async () => {
      try {
        // Check system health
        const health = await apiService.getSystemHealth();
        setSystemHealth(health.overall_status || 'unknown');

        // Initialize WebSocket for real-time updates
        websocketService.connect({
          onConnect: () => {
            setIsConnected(true);
            console.log('Connected to VisionFlow AI');
          },
          onDisconnect: () => {
            setIsConnected(false);
            console.log('Disconnected from VisionFlow AI');
          },
          onMessage: (message) => {
            handleWebSocketMessage(message);
          },
          onError: (error) => {
            console.error('WebSocket error:', error);
          }
        });

        // Get user info (if authentication is implemented)
        // const userInfo = await apiService.getCurrentUser();
        // setCurrentUser(userInfo);

      } catch (error) {
        console.error('Failed to initialize app:', error);
        setSystemHealth('unhealthy');
      }
    };

    initializeApp();

    // Cleanup on unmount
    return () => {
      websocketService.disconnect();
    };
  }, []);

  const handleWebSocketMessage = (message) => {
    switch (message.type) {
      case 'processing_update':
        // Handle processing status updates
        setNotifications(prev => [...prev, {
          id: Date.now(),
          type: 'info',
          title: 'Processing Update',
          message: `Image ${message.image_id}: ${message.status}`,
          timestamp: new Date()
        }]);
        break;

      case 'training_update':
        // Handle training status updates
        setNotifications(prev => [...prev, {
          id: Date.now(),
          type: 'success',
          title: 'Training Update',
          message: `Training run ${message.training_run_id}: ${message.status}`,
          timestamp: new Date()
        }]);
        break;

      case 'health_alert':
        // Handle system health alerts
        setNotifications(prev => [...prev, {
          id: Date.now(),
          type: 'error',
          title: 'System Alert',
          message: `System status: ${message.status}`,
          timestamp: new Date()
        }]);
        setSystemHealth(message.status);
        break;

      case 'connection':
        // Handle connection confirmation
        console.log('WebSocket connection confirmed:', message.message);
        break;

      default:
        console.log('Unknown WebSocket message:', message);
    }
  };

  const clearNotification = (id) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  // System health indicator component
  const SystemHealthIndicator = () => {
    const getHealthColor = (status) => {
      switch (status) {
        case 'healthy': return 'bg-green-500';
        case 'degraded': return 'bg-yellow-500';
        case 'unhealthy': return 'bg-red-500';
        default: return 'bg-gray-500';
      }
    };

    const getHealthText = (status) => {
      switch (status) {
        case 'healthy': return 'All systems operational';
        case 'degraded': return 'Some services degraded';
        case 'unhealthy': return 'System issues detected';
        default: return 'Status unknown';
      }
    };

    return (
      <div className="flex items-center space-x-2 text-sm">
        <div className={`w-2 h-2 rounded-full ${getHealthColor(systemHealth)}`}></div>
        <span className="text-gray-600">
          {getHealthText(systemHealth)}
        </span>
        {isConnected ? (
          <div className="flex items-center space-x-1 text-green-600">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span>Live</span>
          </div>
        ) : (
          <div className="flex items-center space-x-1 text-red-600">
            <div className="w-2 h-2 bg-red-500 rounded-full"></div>
            <span>Offline</span>
          </div>
        )}
      </div>
    );
  };

  // Notification component
  const NotificationCenter = () => {
    if (notifications.length === 0) return null;

    return (
      <div className="fixed top-4 right-4 z-50 space-y-2 max-w-md">
        {notifications.slice(-5).map((notification) => (
          <div
            key={notification.id}
            className={`p-4 rounded-lg shadow-lg transition-all duration-300 ${
              notification.type === 'error'
                ? 'bg-red-50 border border-red-200 text-red-800'
                : notification.type === 'success'
                ? 'bg-green-50 border border-green-200 text-green-800'
                : 'bg-blue-50 border border-blue-200 text-blue-800'
            }`}
          >
            <div className="flex justify-between items-start">
              <div className="flex-1">
                <h4 className="font-semibold text-sm">{notification.title}</h4>
                <p className="text-sm mt-1">{notification.message}</p>
                <p className="text-xs mt-2 opacity-75">
                  {notification.timestamp.toLocaleTimeString()}
                </p>
              </div>
              <button
                onClick={() => clearNotification(notification.id)}
                className="ml-2 text-gray-400 hover:text-gray-600"
              >
                ×
              </button>
            </div>
          </div>
        ))}
      </div>
    );
  };

  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="min-h-screen bg-gray-50">
          {/* Navigation */}
          <Navigation systemHealth={systemHealth} currentUser={currentUser} />
          
          {/* System Status Bar */}
          <div className="bg-white border-b border-gray-200 px-6 py-2">
            <div className="max-w-7xl mx-auto">
              <SystemHealthIndicator />
            </div>
          </div>

          {/* Main Content */}
          <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <Routes>
              {/* Default route - redirect to upload */}
              <Route path="/" element={<Navigate to="/upload" replace />} />
              
              {/* Image Upload Page */}
              <Route 
                path="/upload" 
                element={
                  <div className="space-y-6">
                    <div className="text-center">
                      <h1 className="text-3xl font-bold text-gray-900">
                        VisionFlow AI
                      </h1>
                      <p className="mt-2 text-lg text-gray-600">
                        Advanced computer vision pipeline with SAM segmentation and OpenAI classification
                      </p>
                    </div>
                    <ImageUpload />
                  </div>
                } 
              />
              
              {/* Results Page */}
              <Route 
                path="/results" 
                element={<ResultsViewer />} 
              />
              
              {/* Specific Result Page */}
              <Route 
                path="/results/:imageId" 
                element={<ResultsViewer />} 
              />
              
              {/* Training Dashboard */}
              <Route 
                path="/training" 
                element={<TrainingDashboard />} 
              />
              
              {/* System Monitoring */}
              <Route 
                path="/monitoring" 
                element={<SystemMonitoring />} 
              />
              
              {/* 404 Page */}
              <Route 
                path="*" 
                element={
                  <div className="text-center py-12">
                    <h2 className="text-2xl font-bold text-gray-900">Page Not Found</h2>
                    <p className="mt-2 text-gray-600">
                      The page you're looking for doesn't exist.
                    </p>
                    <button
                      onClick={() => window.history.back()}
                      className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                    >
                      Go Back
                    </button>
                  </div>
                } 
              />
            </Routes>
          </main>

          {/* Footer */}
          <footer className="bg-white border-t border-gray-200 mt-12">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
              <div className="flex justify-between items-center">
                <div className="text-sm text-gray-500">
                  © 2024 VisionFlow AI. Built with React, FastAPI, and SAM.
                </div>
                <div className="flex space-x-4 text-sm text-gray-500">
                  <a href="/api/v1/docs" target="_blank" rel="noopener noreferrer" 
                     className="hover:text-gray-700">
                    API Docs
                  </a>
                  <a href="https://github.com/yourusername/visionflow-ai" 
                     target="_blank" rel="noopener noreferrer"
                     className="hover:text-gray-700">
                    GitHub
                  </a>
                </div>
              </div>
            </div>
          </footer>

          {/* Notifications */}
          <NotificationCenter />
          
          {/* React Hot Toast Container */}
          <Toaster
            position="bottom-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#363636',
                color: '#fff',
              },
            }}
          />
        </div>
      </Router>
    </QueryClientProvider>
  );
}

export default App;