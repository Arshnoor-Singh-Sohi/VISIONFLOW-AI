import React, { useState, useEffect } from 'react';
import { useQuery, useQueryClient } from 'react-query';
import { 
  Play, Pause, Square, Brain, BarChart3, Clock, 
  TrendingUp, AlertCircle, CheckCircle, Settings
} from 'lucide-react';
import toast from 'react-hot-toast';
import { apiService } from '../services/api';
import LoadingSpinner from './LoadingSpinner';

const TrainingDashboard = () => {
  const queryClient = useQueryClient();
  const [selectedRun, setSelectedRun] = useState(null);
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [trainingConfig, setTrainingConfig] = useState({
    run_name: `Training_${new Date().toISOString().split('T')[0]}`,
    model_type: 'random_forest',
    batch_size: 32,
    learning_rate: 0.001,
    num_epochs: 10,
    train_test_split: 0.8,
    min_samples_per_class: 5,
    use_human_labels_only: false,
    augmentation_enabled: true
  });

  // Fetch training runs
  const { data: trainingRuns, isLoading: runsLoading } = useQuery(
    'training-runs',
    () => apiService.getTrainingRuns(),
    { refetchInterval: 5000 }
  );

  // Fetch dataset info
  const { data: datasetInfo, isLoading: datasetLoading } = useQuery(
    'dataset-info',
    () => apiService.getDatasetInfo(),
    { refetchInterval: 10000 }
  );

  const handleStartTraining = async () => {
    try {
      await apiService.startTraining(trainingConfig);
      queryClient.invalidateQueries('training-runs');
      setShowConfigModal(false);
      toast.success('Training started successfully');
    } catch (error) {
      toast.error('Failed to start training');
    }
  };

  const handleControlAction = async (runId, action) => {
    try {
      switch (action) {
        case 'pause':
          await apiService.pauseTraining(runId);
          break;
        case 'resume':
          await apiService.resumeTraining(runId);
          break;
        case 'stop':
          await apiService.stopTraining(runId);
          break;
      }
      queryClient.invalidateQueries('training-runs');
      toast.success(`Training ${action}ed successfully`);
    } catch (error) {
      toast.error(`Failed to ${action} training`);
    }
  };

  const getStatusColor = (status) => {
    const colors = {
      'pending': 'gray',
      'in_progress': 'blue',
      'completed': 'green',
      'failed': 'red',
      'paused': 'yellow'
    };
    return colors[status] || 'gray';
  };

  const StatusBadge = ({ status }) => {
    const color = getStatusColor(status);
    const Icon = status === 'in_progress' ? Brain : status === 'completed' ? CheckCircle : AlertCircle;
    
    return (
      <span className={`inline-flex items-center space-x-1 px-2 py-1 rounded-full text-xs font-medium bg-${color}-100 text-${color}-800`}>
        <Icon className="w-3 h-3" />
        <span className="capitalize">{status.replace('_', ' ')}</span>
      </span>
    );
  };

  if (runsLoading || datasetLoading) {
    return <LoadingSpinner text="Loading training dashboard..." />;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Training Dashboard</h1>
          <p className="text-gray-600">Manage model training and monitor progress</p>
        </div>
        <button
          onClick={() => setShowConfigModal(true)}
          disabled={!datasetInfo?.ready_for_training}
          className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300"
        >
          <Play className="w-4 h-4" />
          <span>Start Training</span>
        </button>
      </div>

      {/* Dataset Status */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Dataset Status</h3>
        {datasetInfo ? (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{datasetInfo.total_samples}</div>
              <div className="text-sm text-gray-600">Total Samples</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{datasetInfo.human_verified_samples}</div>
              <div className="text-sm text-gray-600">Human Verified</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">{Object.keys(datasetInfo.samples_by_label || {}).length}</div>
              <div className="text-sm text-gray-600">Unique Labels</div>
            </div>
            <div className="text-center">
              <div className={`text-2xl font-bold ${datasetInfo.ready_for_training ? 'text-green-600' : 'text-red-600'}`}>
                {datasetInfo.ready_for_training ? 'Ready' : 'Not Ready'}
              </div>
              <div className="text-sm text-gray-600">Training Status</div>
            </div>
          </div>
        ) : (
          <div className="text-center text-gray-500">No dataset information available</div>
        )}
      </div>

      {/* Training Runs */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Recent Training Runs</h3>
        
        {trainingRuns?.length > 0 ? (
          <div className="space-y-4">
            {trainingRuns.map((run) => (
              <div key={run.id} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3">
                      <h4 className="font-medium text-gray-900">{run.run_name}</h4>
                      <StatusBadge status={run.status} />
                    </div>
                    <div className="mt-1 text-sm text-gray-600">
                      {run.num_samples} samples • {run.current_epoch}/{run.total_epochs} epochs
                      {run.validation_accuracy && (
                        <span> • {(run.validation_accuracy * 100).toFixed(1)}% accuracy</span>
                      )}
                    </div>
                    <div className="mt-2">
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${run.progress_percentage}%` }}
                        />
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2 ml-4">
                    {run.status === 'in_progress' && (
                      <>
                        <button
                          onClick={() => handleControlAction(run.id, 'pause')}
                          className="p-2 text-gray-600 hover:text-gray-800"
                          title="Pause"
                        >
                          <Pause className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => handleControlAction(run.id, 'stop')}
                          className="p-2 text-red-600 hover:text-red-800"
                          title="Stop"
                        >
                          <Square className="w-4 h-4" />
                        </button>
                      </>
                    )}
                    
                    {run.status === 'paused' && (
                      <button
                        onClick={() => handleControlAction(run.id, 'resume')}
                        className="p-2 text-green-600 hover:text-green-800"
                        title="Resume"
                      >
                        <Play className="w-4 h-4" />
                      </button>
                    )}
                    
                    <button
                      onClick={() => setSelectedRun(run)}
                      className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
                    >
                      Details
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <Brain className="w-12 h-12 mx-auto mb-4 text-gray-400" />
            <p>No training runs yet. Start your first training session!</p>
          </div>
        )}
      </div>

      {/* Training Configuration Modal */}
      {showConfigModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Training Configuration</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Run Name</label>
                <input
                  type="text"
                  value={trainingConfig.run_name}
                  onChange={(e) => setTrainingConfig(prev => ({ ...prev, run_name: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Number of Epochs</label>
                <input
                  type="number"
                  min="1"
                  max="100"
                  value={trainingConfig.num_epochs}
                  onChange={(e) => setTrainingConfig(prev => ({ ...prev, num_epochs: parseInt(e.target.value) }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              
              <div>
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={trainingConfig.use_human_labels_only}
                    onChange={(e) => setTrainingConfig(prev => ({ ...prev, use_human_labels_only: e.target.checked }))}
                    className="w-4 h-4 text-blue-600"
                  />
                  <span className="text-sm text-gray-700">Use only human-verified labels</span>
                </label>
              </div>
            </div>
            
            <div className="flex justify-end space-x-3 mt-6">
              <button
                onClick={() => setShowConfigModal(false)}
                className="px-4 py-2 text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200"
              >
                Cancel
              </button>
              <button
                onClick={handleStartTraining}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                Start Training
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TrainingDashboard;