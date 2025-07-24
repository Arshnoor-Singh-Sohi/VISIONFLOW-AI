import axios from 'axios';
import toast from 'react-hot-toast';

// Create axios instance with default configuration
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for authentication and logging
api.interceptors.request.use(
  (config) => {
    // Add authentication token if available
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    // Add request ID for tracing
    config.headers['X-Request-ID'] = generateRequestId();

    // Log request in development
    if (process.env.NODE_ENV === 'development') {
      console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    }

    return config;
  },
  (error) => {
    console.error('Request interceptor error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    // Log successful responses in development
    if (process.env.NODE_ENV === 'development') {
      console.log(`API Response: ${response.status} ${response.config.url}`);
    }
    return response;
  },
  (error) => {
    // Handle different types of errors
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response;
      
      switch (status) {
        case 401:
          toast.error('Authentication required');
          // Redirect to login if implemented
          break;
        case 403:
          toast.error('Access denied');
          break;
        case 404:
          toast.error('Resource not found');
          break;
        case 422:
          toast.error(data.detail || 'Validation error');
          break;
        case 429:
          toast.error('Too many requests. Please wait.');
          break;
        case 500:
          toast.error('Server error. Please try again.');
          break;
        case 503:
          toast.error('Service temporarily unavailable');
          break;
        default:
          toast.error(data.detail || `Error: ${status}`);
      }
    } else if (error.request) {
      // Network error
      toast.error('Network error. Please check your connection.');
    } else {
      // Something else happened
      toast.error('An unexpected error occurred');
    }

    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

// Helper function to generate request IDs
function generateRequestId() {
  return Math.random().toString(36).substring(2) + Date.now().toString(36);
}

// Helper function to handle file uploads
function createFormData(file, additionalData = {}) {
  const formData = new FormData();
  formData.append('image', file);
  
  Object.entries(additionalData).forEach(([key, value]) => {
    if (typeof value === 'object') {
      formData.append(key, JSON.stringify(value));
    } else {
      formData.append(key, value);
    }
  });
  
  return formData;
}

// Main API service object
export const apiService = {
  // =========================================================================
  // SYSTEM HEALTH AND INFO
  // =========================================================================
  
  async getSystemHealth() {
    const response = await api.get('/health');
    return response.data;
  },

  async getSystemInfo() {
    const response = await api.get('/');
    return response.data;
  },

  async getComponentHealth(component) {
    const response = await api.get(`/api/v1/health/${component}`);
    return response.data;
  },

  // =========================================================================
  // IMAGE MANAGEMENT
  // =========================================================================
  
  async uploadImage(file, config = {}) {
    const formData = createFormData(file, {
      config: JSON.stringify(config),
      user_id: config.user_id || null
    });

    const response = await api.post('/api/v1/images/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (config.onProgress) {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          config.onProgress(percentCompleted);
        }
      },
    });

    return response.data;
  },

  async getImageStatus(imageId) {
    const response = await api.get(`/api/v1/images/status/${imageId}`);
    return response.data;
  },

  async listImages(params = {}) {
    const {
      page = 1,
      pageSize = 20,
      status = null,
      userId = null
    } = params;

    const queryParams = new URLSearchParams({
      page: page.toString(),
      page_size: pageSize.toString()
    });

    if (status) queryParams.append('status', status);
    if (userId) queryParams.append('user_id', userId);

    const response = await api.get(`/api/v1/images/list?${queryParams}`);
    return response.data;
  },

  async downloadImage(imageId) {
    const response = await api.get(`/api/v1/images/download/${imageId}`, {
      responseType: 'blob'
    });
    return response.data;
  },

  async getThumbnail(imageId, size = 300) {
    const response = await api.get(`/api/v1/images/thumbnail/${imageId}?size=${size}`, {
      responseType: 'blob'
    });
    return response.data;
  },

  async deleteImage(imageId) {
    const response = await api.delete(`/api/v1/images/${imageId}`);
    return response.data;
  },

  async cancelProcessing(imageId) {
    const response = await api.post(`/api/v1/images/${imageId}/cancel`);
    return response.data;
  },

  async reprocessImage(imageId, config = {}) {
    const response = await api.post(`/api/v1/images/process/${imageId}`, config);
    return response.data;
  },

  // =========================================================================
  // RESULTS AND ANALYSIS
  // =========================================================================
  
  async getResultSummary(imageId) {
    const response = await api.get(`/api/v1/results/summary/${imageId}`);
    return response.data;
  },

  async getDetailedResults(imageId) {
    const response = await api.get(`/api/v1/results/detailed/${imageId}`);
    return response.data;
  },

  async getAnnotatedImage(imageId, options = {}) {
    const {
      showLabels = true,
      showConfidence = true,
      showBbox = true
    } = options;

    const queryParams = new URLSearchParams({
      show_labels: showLabels.toString(),
      show_confidence: showConfidence.toString(),
      show_bbox: showBbox.toString()
    });

    const response = await api.get(
      `/api/v1/results/annotated/${imageId}?${queryParams}`,
      { responseType: 'blob' }
    );
    return response.data;
  },

  async getSegmentImage(segmentId) {
    const response = await api.get(`/api/v1/results/segment-image/${segmentId}`, {
      responseType: 'blob'
    });
    return response.data;
  },

  async exportResults(imageId, format = 'json') {
    const response = await api.get(
      `/api/v1/results/export/${imageId}?format=${format}`,
      { responseType: 'blob' }
    );
    return response.data;
  },

  async exportBatchResults(imageIds, format = 'json') {
    const queryParams = imageIds.map(id => `image_ids=${id}`).join('&');
    const response = await api.get(
      `/api/v1/results/export/batch?${queryParams}&format=${format}`,
      { responseType: 'blob' }
    );
    return response.data;
  },

  async submitFeedback(classificationId, correctLabel, confidence, notes = null) {
    const response = await api.post('/api/v1/results/feedback', {
      classification_id: classificationId,
      correct_label: correctLabel,
      confidence: confidence,
      notes: notes
    });
    return response.data;
  },

  async getAccuracyMetrics() {
    const response = await api.get('/api/v1/results/feedback/accuracy');
    return response.data;
  },

  // =========================================================================
  // ANALYTICS AND REPORTING
  // =========================================================================
  
  async getAnalyticsOverview(days = 30) {
    const response = await api.get(`/api/v1/results/analytics/overview?days=${days}`);
    return response.data;
  },

  async getPerformanceAnalytics() {
    const response = await api.get('/api/v1/results/analytics/performance');
    return response.data;
  },

  async getProcessingMetrics(hours = 24) {
    const response = await api.get(`/api/v1/health/metrics/processing?hours=${hours}`);
    return response.data;
  },

  async getSystemMetrics() {
    const response = await api.get('/api/v1/health/metrics/system');
    return response.data;
  },

  // =========================================================================
  // TRAINING MANAGEMENT
  // =========================================================================
  
  async startTraining(config) {
    const response = await api.post('/api/v1/training/start', config);
    return response.data;
  },

  async getTrainingRuns(params = {}) {
    const {
      status = null,
      limit = 20,
      offset = 0
    } = params;

    const queryParams = new URLSearchParams({
      limit: limit.toString(),
      offset: offset.toString()
    });

    if (status) queryParams.append('status', status);

    const response = await api.get(`/api/v1/training/runs?${queryParams}`);
    return response.data;
  },

  async getTrainingRun(runId) {
    const response = await api.get(`/api/v1/training/runs/${runId}`);
    return response.data;
  },

  async getTrainingProgress(runId) {
    const response = await api.get(`/api/v1/training/progress/${runId}`);
    return response.data;
  },

  async getTrainingMetrics(runId) {
    const response = await api.get(`/api/v1/training/metrics/${runId}`);
    return response.data;
  },

  async pauseTraining(runId) {
    const response = await api.post(`/api/v1/training/runs/${runId}/pause`);
    return response.data;
  },

  async resumeTraining(runId) {
    const response = await api.post(`/api/v1/training/runs/${runId}/resume`);
    return response.data;
  },

  async stopTraining(runId) {
    const response = await api.post(`/api/v1/training/runs/${runId}/stop`);
    return response.data;
  },

  async getDatasetInfo() {
    const response = await api.get('/api/v1/training/dataset');
    return response.data;
  },

  async getTrainingSamples(params = {}) {
    const {
      label = null,
      source = null,
      verifiedOnly = false,
      limit = 50,
      offset = 0
    } = params;

    const queryParams = new URLSearchParams({
      limit: limit.toString(),
      offset: offset.toString(),
      verified_only: verifiedOnly.toString()
    });

    if (label) queryParams.append('label', label);
    if (source) queryParams.append('source', source);

    const response = await api.get(`/api/v1/training/dataset/samples?${queryParams}`);
    return response.data;
  },

  async downloadModel(runId) {
    const response = await api.get(`/api/v1/training/models/${runId}/download`, {
      responseType: 'blob'
    });
    return response.data;
  },

  async listTrainedModels() {
    const response = await api.get('/api/v1/training/models');
    return response.data;
  },

  // =========================================================================
  // ERROR AND LOG MONITORING
  // =========================================================================
  
  async getRecentErrors(hours = 24, limit = 50) {
    const response = await api.get(
      `/api/v1/health/errors/recent?hours=${hours}&limit=${limit}`
    );
    return response.data;
  },

  async getLogSummary(hours = 24) {
    const response = await api.get(`/api/v1/health/logs/summary?hours=${hours}`);
    return response.data;
  },

  // =========================================================================
  // UTILITY FUNCTIONS
  // =========================================================================
  
  // Create a blob URL for downloaded images
  createBlobUrl(blob) {
    return URL.createObjectURL(blob);
  },

  // Download a file with a given filename
  downloadFile(blob, filename) {
    const url = this.createBlobUrl(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  },

  // Format file size for display
  formatFileSize(bytes) {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  },

  // Format duration for display
  formatDuration(seconds) {
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    if (minutes < 60) return `${minutes}m ${remainingSeconds}s`;
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;
    return `${hours}h ${remainingMinutes}m`;
  },

  // Get status color for UI
  getStatusColor(status) {
    const colors = {
      'uploaded': 'blue',
      'segmenting': 'yellow',
      'classifying': 'purple',
      'training': 'indigo',
      'completed': 'green',
      'failed': 'red',
      'cancelled': 'gray'
    };
    return colors[status] || 'gray';
  },

  // Get training status color
  getTrainingStatusColor(status) {
    const colors = {
      'pending': 'gray',
      'in_progress': 'blue',
      'completed': 'green',
      'failed': 'red',
      'paused': 'yellow'
    };
    return colors[status] || 'gray';
  }
};

// Export the configured axios instance for custom requests
export { api };

export default apiService;