import React, { useState, useEffect } from 'react';
import { useQuery } from 'react-query';
import { 
  Activity, Server, Database, Cpu, MemoryStick, 
  HardDrive, AlertTriangle, CheckCircle, Clock
} from 'lucide-react';
import { apiService } from '../services/api';
import LoadingSpinner from './LoadingSpinner';

const SystemMonitoring = () => {
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h');

  const { data: systemHealth, isLoading: healthLoading } = useQuery(
    'system-health',
    () => apiService.getSystemHealth(),
    { refetchInterval: 30000 }
  );

  const { data: systemMetrics, isLoading: metricsLoading } = useQuery(
    'system-metrics',
    () => apiService.getSystemMetrics(),
    { refetchInterval: 10000 }
  );

  const { data: processingMetrics } = useQuery(
    ['processing-metrics', selectedTimeRange],
    () => apiService.getProcessingMetrics(selectedTimeRange === '24h' ? 24 : selectedTimeRange === '7d' ? 168 : 720),
    { refetchInterval: 60000 }
  );

  const { data: recentErrors } = useQuery(
    'recent-errors',
    () => apiService.getRecentErrors(24, 10),
    { refetchInterval: 30000 }
  );

  const getHealthStatusColor = (status) => {
    switch (status) {
      case 'healthy': return 'text-green-600 bg-green-100';
      case 'degraded': return 'text-yellow-600 bg-yellow-100';
      case 'unhealthy': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getHealthIcon = (status) => {
    switch (status) {
      case 'healthy': return CheckCircle;
      case 'degraded': return AlertTriangle;
      case 'unhealthy': return AlertTriangle;
      default: return Activity;
    }
  };

  const MetricCard = ({ title, value, unit, icon: Icon, color = 'blue' }) => (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className={`text-2xl font-bold text-${color}-600`}>
            {value}<span className="text-lg text-gray-500 ml-1">{unit}</span>
          </p>
        </div>
        <Icon className={`w-8 h-8 text-${color}-600`} />
      </div>
    </div>
  );

  if (healthLoading || metricsLoading) {
    return <LoadingSpinner text="Loading system monitoring..." />;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">System Monitoring</h1>
        <p className="text-gray-600">Monitor system health, performance, and processing metrics</p>
      </div>

      {/* Overall Health Status */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">System Health Overview</h3>
        
        {systemHealth ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {Object.entries(systemHealth.components || {}).map(([component, status]) => {
              const StatusIcon = getHealthIcon(status.status);
              return (
                <div key={component} className="flex items-center space-x-3 p-3 rounded-lg border border-gray-200">
                  <StatusIcon className={`w-5 h-5 ${getHealthStatusColor(status.status).split(' ')[0]}`} />
                  <div>
                    <p className="font-medium text-gray-900 capitalize">{component.replace('_', ' ')}</p>
                    <p className={`text-sm px-2 py-1 rounded-full ${getHealthStatusColor(status.status)}`}>
                      {status.status}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="text-center text-gray-500">Unable to load system health data</div>
        )}
      </div>

      {/* System Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {systemMetrics && (
          <>
            <MetricCard
              title="CPU Usage"
              value={systemMetrics.cpu?.percent_used?.toFixed(1) || 0}
              unit="%"
              icon={Cpu}
              color={systemMetrics.cpu?.percent_used > 80 ? 'red' : systemMetrics.cpu?.percent_used > 60 ? 'yellow' : 'green'}
            />
            <MetricCard
              title="Memory Usage"
              value={systemMetrics.memory?.percent_used?.toFixed(1) || 0}
              unit="%"
              icon={MemoryStick}
              color={systemMetrics.memory?.percent_used > 80 ? 'red' : systemMetrics.memory?.percent_used > 60 ? 'yellow' : 'green'}
            />
            <MetricCard
              title="Disk Usage"
              value={systemMetrics.disk?.percent_used?.toFixed(1) || 0}
              unit="%"
              icon={HardDrive}
              color={systemMetrics.disk?.percent_used > 80 ? 'red' : systemMetrics.disk?.percent_used > 60 ? 'yellow' : 'green'}
            />
            <MetricCard
              title="Process Memory"
              value={systemMetrics.process?.memory_mb?.toFixed(0) || 0}
              unit="MB"
              icon={Server}
              color="blue"
            />
          </>
        )}
      </div>

      {/* Processing Metrics */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-gray-900">Processing Performance</h3>
          <select
            value={selectedTimeRange}
            onChange={(e) => setSelectedTimeRange(e.target.value)}
            className="px-3 py-1 border border-gray-300 rounded-md text-sm"
          >
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>
        </div>

        {processingMetrics ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600">{processingMetrics.total_processed}</div>
              <div className="text-sm text-gray-600">Images Processed</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-green-600">
                {processingMetrics.success_rate_percent?.toFixed(1) || 0}%
              </div>
              <div className="text-sm text-gray-600">Success Rate</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-600">
                {processingMetrics.average_processing_time_seconds?.toFixed(1) || 0}s
              </div>
              <div className="text-sm text-gray-600">Avg Processing Time</div>
            </div>
          </div>
        ) : (
          <div className="text-center text-gray-500">No processing metrics available</div>
        )}
      </div>

      {/* Recent Errors */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Recent Errors</h3>
        
        {recentErrors?.errors?.length > 0 ? (
          <div className="space-y-3">
            {recentErrors.errors.slice(0, 5).map((error, index) => (
              <div key={index} className="flex items-start space-x-3 p-3 bg-red-50 rounded-lg border border-red-200">
                <AlertTriangle className="w-5 h-5 text-red-600 mt-0.5" />
                <div className="flex-1">
                  <p className="font-medium text-red-800">{error.category}</p>
                  <p className="text-sm text-red-700 mt-1">{error.message}</p>
                  <p className="text-xs text-red-600 mt-2">
                    <Clock className="w-3 h-3 inline mr-1" />
                    {new Date(error.timestamp).toLocaleString()}
                  </p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <CheckCircle className="w-12 h-12 mx-auto mb-4 text-green-500" />
            <p>No recent errors detected!</p>
          </div>
        )}
      </div>

      {/* Service Status Details */}
      {systemHealth?.components && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Service Details</h3>
          
          <div className="space-y-4">
            {Object.entries(systemHealth.components).map(([service, details]) => (
              <div key={service} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-gray-900 capitalize">{service.replace('_', ' ')}</h4>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getHealthStatusColor(details.status)}`}>
                    {details.status}
                  </span>
                </div>
                
                {details.error && (
                  <p className="text-sm text-red-600 mt-1">{details.error}</p>
                )}
                
                {details.response_time_ms && (
                  <p className="text-sm text-gray-600 mt-1">
                    Response time: {details.response_time_ms}ms
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default SystemMonitoring;