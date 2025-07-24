import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useNavigate } from 'react-router-dom';
import { Upload, Settings, Image as ImageIcon, AlertCircle, CheckCircle, X } from 'lucide-react';
import toast from 'react-hot-toast';
import { apiService } from '../services/api';
import LoadingSpinner from './LoadingSpinner';

const ImageUpload = () => {
  const navigate = useNavigate();
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [processingConfig, setProcessingConfig] = useState({
    min_area: 1000,
    max_segments: 60,
    confidence_threshold: 0.7,
    classification_context: 'food identification',
    enable_training: true
  });
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);

  // Handle file drop and selection
  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    // Handle rejected files
    rejectedFiles.forEach(file => {
      toast.error(`${file.file.name}: ${file.errors[0].message}`);
    });

    // Process accepted files
    const newFiles = acceptedFiles.map(file => ({
      id: Math.random().toString(36).substring(2),
      file,
      status: 'ready',
      progress: 0,
      imageId: null,
      preview: URL.createObjectURL(file)
    }));

    setUploadedFiles(prev => [...prev, ...newFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp']
    },
    maxSize: 10 * 1024 * 1024, // 10MB
    multiple: true
  });

  // Remove file from upload list
  const removeFile = (fileId) => {
    setUploadedFiles(prev => {
      const file = prev.find(f => f.id === fileId);
      if (file && file.preview) {
        URL.revokeObjectURL(file.preview);
      }
      return prev.filter(f => f.id !== fileId);
    });
  };

  // Upload and process files
  const uploadFiles = async () => {
    const readyFiles = uploadedFiles.filter(f => f.status === 'ready');
    
    if (readyFiles.length === 0) {
      toast.error('No files ready for upload');
      return;
    }

    setIsUploading(true);

    try {
      for (const fileData of readyFiles) {
        // Update file status
        setUploadedFiles(prev => 
          prev.map(f => 
            f.id === fileData.id 
              ? { ...f, status: 'uploading', progress: 0 }
              : f
          )
        );

        try {
          // Upload file with progress tracking
          const result = await apiService.uploadImage(fileData.file, {
            ...processingConfig,
            onProgress: (progress) => {
              setUploadedFiles(prev => 
                prev.map(f => 
                  f.id === fileData.id 
                    ? { ...f, progress }
                    : f
                )
              );
            }
          });

          // Update with success
          setUploadedFiles(prev => 
            prev.map(f => 
              f.id === fileData.id 
                ? { 
                    ...f, 
                    status: 'uploaded', 
                    progress: 100,
                    imageId: result.image_id
                  }
                : f
            )
          );

          toast.success(`${fileData.file.name} uploaded successfully`);

        } catch (error) {
          // Update with error
          setUploadedFiles(prev => 
            prev.map(f => 
              f.id === fileData.id 
                ? { 
                    ...f, 
                    status: 'error', 
                    error: error.response?.data?.detail || error.message
                  }
                : f
            )
          );

          toast.error(`Failed to upload ${fileData.file.name}`);
        }
      }

      // Navigate to results if any files were uploaded successfully
      const successfulUploads = uploadedFiles.filter(f => f.status === 'uploaded');
      if (successfulUploads.length > 0) {
        setTimeout(() => {
          navigate('/results');
        }, 1500);
      }

    } finally {
      setIsUploading(false);
    }
  };

  // Clear all files
  const clearAllFiles = () => {
    uploadedFiles.forEach(file => {
      if (file.preview) {
        URL.revokeObjectURL(file.preview);
      }
    });
    setUploadedFiles([]);
  };

  // Retry failed upload
  const retryUpload = (fileId) => {
    setUploadedFiles(prev => 
      prev.map(f => 
        f.id === fileId 
          ? { ...f, status: 'ready', error: null, progress: 0 }
          : f
      )
    );
  };

  // View results for uploaded image
  const viewResults = (imageId) => {
    navigate(`/results/${imageId}`);
  };

  // File status icon
  const getStatusIcon = (status) => {
    switch (status) {
      case 'ready':
        return <ImageIcon className="w-4 h-4 text-blue-500" />;
      case 'uploading':
        return <LoadingSpinner size="sm" />;
      case 'uploaded':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      default:
        return <ImageIcon className="w-4 h-4 text-gray-400" />;
    }
  };

  // Get status color
  const getStatusColor = (status) => {
    switch (status) {
      case 'ready': return 'border-blue-200 bg-blue-50';
      case 'uploading': return 'border-yellow-200 bg-yellow-50';
      case 'uploaded': return 'border-green-200 bg-green-50';
      case 'error': return 'border-red-200 bg-red-50';
      default: return 'border-gray-200 bg-gray-50';
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Upload Area */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div
          {...getRootProps()}
          className={`
            border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
            ${isDragActive 
              ? 'border-blue-400 bg-blue-50' 
              : 'border-gray-300 hover:border-gray-400'
            }
          `}
        >
          <input {...getInputProps()} />
          <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            {isDragActive ? 'Drop images here' : 'Upload Images'}
          </h3>
          <p className="text-gray-600 mb-4">
            Drag and drop images here, or click to select files
          </p>
          <p className="text-sm text-gray-500">
            Supported formats: JPEG, PNG, WebP • Max size: 10MB per file
          </p>
        </div>
      </div>

      {/* Processing Configuration */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-gray-900">Processing Settings</h3>
          <button
            onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}
            className="flex items-center space-x-2 text-blue-600 hover:text-blue-700"
          >
            <Settings className="w-4 h-4" />
            <span>{showAdvancedSettings ? 'Hide' : 'Show'} Advanced</span>
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Basic Settings */}
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Classification Context
              </label>
              <select
                value={processingConfig.classification_context}
                onChange={(e) => setProcessingConfig(prev => ({
                  ...prev,
                  classification_context: e.target.value
                }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="food identification">Food Identification</option>
                <option value="general objects">General Objects</option>
                <option value="kitchen items">Kitchen Items</option>
              </select>
            </div>

            <div>
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={processingConfig.enable_training}
                  onChange={(e) => setProcessingConfig(prev => ({
                    ...prev,
                    enable_training: e.target.checked
                  }))}
                  className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                <span className="text-sm font-medium text-gray-700">
                  Use for Model Training
                </span>
              </label>
              <p className="text-xs text-gray-500 mt-1">
                Allow this data to improve the AI model
              </p>
            </div>
          </div>

          {/* Advanced Settings */}
          {showAdvancedSettings && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Min Segment Area
                </label>
                <input
                  type="number"
                  min="100"
                  max="10000"
                  value={processingConfig.min_area}
                  onChange={(e) => setProcessingConfig(prev => ({
                    ...prev,
                    min_area: parseInt(e.target.value) || 1000
                  }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Minimum area in pixels for detected segments
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Max Segments
                </label>
                <input
                  type="number"
                  min="10"
                  max="100"
                  value={processingConfig.max_segments}
                  onChange={(e) => setProcessingConfig(prev => ({
                    ...prev,
                    max_segments: parseInt(e.target.value) || 60
                  }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Maximum number of segments to process
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Confidence Threshold
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.1"
                  value={processingConfig.confidence_threshold}
                  onChange={(e) => setProcessingConfig(prev => ({
                    ...prev,
                    confidence_threshold: parseFloat(e.target.value)
                  }))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Less selective</span>
                  <span>{processingConfig.confidence_threshold}</span>
                  <span>More selective</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* File List */}
      {uploadedFiles.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900">
              Files ({uploadedFiles.length})
            </h3>
            <button
              onClick={clearAllFiles}
              className="text-sm text-gray-500 hover:text-gray-700"
            >
              Clear All
            </button>
          </div>

          <div className="space-y-3">
            {uploadedFiles.map((fileData) => (
              <div
                key={fileData.id}
                className={`p-4 rounded-lg border-2 ${getStatusColor(fileData.status)}`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(fileData.status)}
                    <div className="flex-1">
                      <p className="text-sm font-medium text-gray-900">
                        {fileData.file.name}
                      </p>
                      <p className="text-xs text-gray-500">
                        {apiService.formatFileSize(fileData.file.size)}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center space-x-2">
                    {fileData.status === 'uploading' && (
                      <div className="text-xs text-gray-600">
                        {fileData.progress}%
                      </div>
                    )}
                    
                    {fileData.status === 'uploaded' && fileData.imageId && (
                      <button
                        onClick={() => viewResults(fileData.imageId)}
                        className="text-xs bg-blue-600 text-white px-2 py-1 rounded hover:bg-blue-700"
                      >
                        View Results
                      </button>
                    )}
                    
                    {fileData.status === 'error' && (
                      <button
                        onClick={() => retryUpload(fileData.id)}
                        className="text-xs bg-red-600 text-white px-2 py-1 rounded hover:bg-red-700"
                      >
                        Retry
                      </button>
                    )}
                    
                    <button
                      onClick={() => removeFile(fileData.id)}
                      className="text-gray-400 hover:text-gray-600"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {/* Progress Bar */}
                {fileData.status === 'uploading' && (
                  <div className="mt-2">
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${fileData.progress}%` }}
                      ></div>
                    </div>
                  </div>
                )}

                {/* Error Message */}
                {fileData.status === 'error' && fileData.error && (
                  <div className="mt-2 text-xs text-red-600">
                    Error: {fileData.error}
                  </div>
                )}

                {/* Preview */}
                {fileData.preview && (
                  <div className="mt-3">
                    <img
                      src={fileData.preview}
                      alt="Preview"
                      className="w-20 h-20 object-cover rounded border"
                    />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Upload Button */}
      {uploadedFiles.length > 0 && (
        <div className="flex justify-center">
          <button
            onClick={uploadFiles}
            disabled={isUploading || uploadedFiles.filter(f => f.status === 'ready').length === 0}
            className={`
              px-8 py-3 rounded-lg font-medium transition-colors
              ${isUploading || uploadedFiles.filter(f => f.status === 'ready').length === 0
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-blue-600 text-white hover:bg-blue-700'
              }
            `}
          >
            {isUploading ? (
              <div className="flex items-center space-x-2">
                <LoadingSpinner size="sm" />
                <span>Processing...</span>
              </div>
            ) : (
              `Upload & Process ${uploadedFiles.filter(f => f.status === 'ready').length} Files`
            )}
          </button>
        </div>
      )}
    </div>
  );
};

export default ImageUpload;