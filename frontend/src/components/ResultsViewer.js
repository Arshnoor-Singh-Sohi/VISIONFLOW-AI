import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useQueryClient } from 'react-query';
import { 
  Eye, Download, Share2, RefreshCw, Edit3, MessageSquare, 
  Image as ImageIcon, Clock, CheckCircle, AlertCircle,
  BarChart3, Grid3X3, Layers, ZoomIn
} from 'lucide-react';
import toast from 'react-hot-toast';
import { apiService } from '../services/api';
import LoadingSpinner from './LoadingSpinner';

const ResultsViewer = () => {
  const { imageId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [selectedImage, setSelectedImage] = useState(imageId || null);
  const [viewMode, setViewMode] = useState('grid'); // 'grid', 'detail'
  const [selectedSegment, setSelectedSegment] = useState(null);
  const [showFeedbackModal, setShowFeedbackModal] = useState(false);
  const [feedbackData, setFeedbackData] = useState(null);
  const [annotationSettings, setAnnotationSettings] = useState({
    showLabels: true,
    showConfidence: true,
    showBbox: true
  });

  // Fetch images list
  const { data: imagesList, isLoading: imagesLoading, error: imagesError } = useQuery(
    'images-list',
    () => apiService.listImages({ page: 1, pageSize: 50 }),
    { refetchInterval: 5000 } // Refresh every 5 seconds for status updates
  );

  // Fetch selected image details
  const { data: imageDetails, isLoading: detailsLoading, error: detailsError } = useQuery(
    ['image-details', selectedImage],
    () => selectedImage ? apiService.getDetailedResults(selectedImage) : null,
    { 
      enabled: !!selectedImage,
      refetchInterval: (data) => {
        // Keep refreshing if processing is not complete
        if (data && ['uploaded', 'segmenting', 'classifying', 'training'].includes(data.status)) {
          return 3000; // 3 seconds
        }
        return false; // Don't refetch if completed
      }
    }
  );

  // Auto-select first image if none selected
  useEffect(() => {
    if (!selectedImage && imagesList?.images?.length > 0) {
      setSelectedImage(imagesList.images[0].id);
    }
  }, [imagesList, selectedImage]);

  // Handle image selection
  const handleImageSelect = (id) => {
    setSelectedImage(id);
    setSelectedSegment(null);
    navigate(`/results/${id}`);
  };

  // Handle image refresh
  const handleRefresh = () => {
    queryClient.invalidateQueries(['image-details', selectedImage]);
    queryClient.invalidateQueries('images-list');
    toast.success('Results refreshed');
  };

  // Handle reprocessing
  const handleReprocess = async () => {
    if (!selectedImage) return;

    try {
      await apiService.reprocessImage(selectedImage);
      queryClient.invalidateQueries(['image-details', selectedImage]);
      toast.success('Reprocessing started');
    } catch (error) {
      toast.error('Failed to start reprocessing');
    }
  };

  // Handle image download
  const handleDownload = async (type = 'original') => {
    if (!selectedImage) return;

    try {
      let blob;
      let filename;

      switch (type) {
        case 'original':
          blob = await apiService.downloadImage(selectedImage);
          filename = `${imageDetails?.filename || 'image'}_original.jpg`;
          break;
        case 'annotated':
          blob = await apiService.getAnnotatedImage(selectedImage, annotationSettings);
          filename = `${imageDetails?.filename || 'image'}_annotated.jpg`;
          break;
        case 'results':
          blob = await apiService.exportResults(selectedImage, 'json');
          filename = `${imageDetails?.filename || 'image'}_results.json`;
          break;
        default:
          return;
      }

      apiService.downloadFile(blob, filename);
      toast.success('Download started');
    } catch (error) {
      toast.error('Download failed');
    }
  };

  // Handle feedback submission
  const handleFeedbackSubmit = async (data) => {
    try {
      await apiService.submitFeedback(
        data.classificationId,
        data.correctLabel,
        data.confidence,
        data.notes
      );
      
      queryClient.invalidateQueries(['image-details', selectedImage]);
      setShowFeedbackModal(false);
      setFeedbackData(null);
      toast.success('Feedback submitted successfully');
    } catch (error) {
      toast.error('Failed to submit feedback');
    }
  };

  // Open feedback modal
  const openFeedbackModal = (classification, segment) => {
    setFeedbackData({
      classificationId: classification.id,
      currentLabel: classification.primary_label,
      confidence: classification.confidence_score,
      segment: segment
    });
    setShowFeedbackModal(true);
  };

  // Status badge component
  const StatusBadge = ({ status }) => {
    const getStatusConfig = (status) => {
      const configs = {
        'uploaded': { color: 'blue', icon: Clock, text: 'Uploaded' },
        'segmenting': { color: 'yellow', icon: RefreshCw, text: 'Segmenting' },
        'classifying': { color: 'purple', icon: RefreshCw, text: 'Classifying' },
        'training': { color: 'indigo', icon: RefreshCw, text: 'Training' },
        'completed': { color: 'green', icon: CheckCircle, text: 'Completed' },
        'failed': { color: 'red', icon: AlertCircle, text: 'Failed' },
        'cancelled': { color: 'gray', icon: AlertCircle, text: 'Cancelled' }
      };
      return configs[status] || configs['uploaded'];
    };

    const config = getStatusConfig(status);
    const Icon = config.icon;

    return (
      <span className={`
        inline-flex items-center space-x-1 px-2 py-1 rounded-full text-xs font-medium
        bg-${config.color}-100 text-${config.color}-800
      `}>
        <Icon className={`w-3 h-3 ${status === 'segmenting' || status === 'classifying' || status === 'training' ? 'animate-spin' : ''}`} />
        <span>{config.text}</span>
      </span>
    );
  };

  // Segment grid component
  const SegmentGrid = ({ segments }) => (
    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
      {segments.map((segment, index) => (
        <div
          key={segment.id}
          className={`
            relative border-2 rounded-lg p-2 cursor-pointer transition-colors
            ${selectedSegment?.id === segment.id 
              ? 'border-blue-500 bg-blue-50' 
              : 'border-gray-200 hover:border-gray-300'
            }
          `}
          onClick={() => setSelectedSegment(segment)}
        >
          {segment.segment_image_url ? (
            <img
              src={segment.segment_image_url}
              alt={`Segment ${index + 1}`}
              className="w-full h-20 object-cover rounded"
            />
          ) : (
            <div className="w-full h-20 bg-gray-100 rounded flex items-center justify-center">
              <ImageIcon className="w-6 h-6 text-gray-400" />
            </div>
          )}
          
          <div className="mt-2 text-xs">
            <div className="font-medium text-gray-900 truncate">
              {segment.classification?.primary_label || 'Unknown'}
            </div>
            {segment.classification?.confidence_score && (
              <div className="text-gray-500">
                {(segment.classification.confidence_score * 100).toFixed(0)}%
              </div>
            )}
          </div>

          {segment.classification?.human_verified && (
            <div className="absolute top-1 right-1">
              <CheckCircle className="w-4 h-4 text-green-500" />
            </div>
          )}
        </div>
      ))}
    </div>
  );

  // Segment detail component
  const SegmentDetail = ({ segment }) => (
    <div className="bg-gray-50 rounded-lg p-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          {segment.segment_image_url ? (
            <img
              src={segment.segment_image_url}
              alt="Segment detail"
              className="w-full h-48 object-cover rounded-lg"
            />
          ) : (
            <div className="w-full h-48 bg-gray-200 rounded-lg flex items-center justify-center">
              <ImageIcon className="w-12 h-12 text-gray-400" />
            </div>
          )}
        </div>

        <div className="space-y-4">
          <div>
            <h4 className="font-medium text-gray-900 mb-2">Classification</h4>
            {segment.classification ? (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-medium">
                    {segment.classification.primary_label}
                  </span>
                  <button
                    onClick={() => openFeedbackModal(segment.classification, segment)}
                    className="text-blue-600 hover:text-blue-700 text-sm"
                  >
                    <Edit3 className="w-4 h-4" />
                  </button>
                </div>
                <div className="text-sm text-gray-600">
                  Confidence: {(segment.classification.confidence_score * 100).toFixed(1)}%
                </div>
                {segment.classification.human_verified && (
                  <div className="flex items-center space-x-1 text-green-600 text-sm">
                    <CheckCircle className="w-4 h-4" />
                    <span>Human verified</span>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-gray-500">No classification available</div>
            )}
          </div>

          <div>
            <h4 className="font-medium text-gray-900 mb-2">Segment Info</h4>
            <div className="space-y-1 text-sm text-gray-600">
              <div>Area: {segment.area?.toLocaleString()} pixels</div>
              <div>
                Bounding Box: {segment.bbox[2]} × {segment.bbox[3]}
              </div>
              <div>
                Confidence: {(segment.confidence_score * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  // Feedback modal component
  const FeedbackModal = ({ isOpen, onClose, data, onSubmit }) => {
    const [correctLabel, setCorrectLabel] = useState(data?.currentLabel || '');
    const [confidence, setConfidence] = useState(5);
    const [notes, setNotes] = useState('');

    if (!isOpen) return null;

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
          <h3 className="text-lg font-medium text-gray-900 mb-4">
            Provide Feedback
          </h3>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Correct Label
              </label>
              <input
                type="text"
                value={correctLabel}
                onChange={(e) => setCorrectLabel(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Enter the correct classification"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Confidence (1-10)
              </label>
              <input
                type="range"
                min="1"
                max="10"
                value={confidence}
                onChange={(e) => setConfidence(parseInt(e.target.value))}
                className="w-full"
              />
              <div className="text-center text-sm text-gray-600">{confidence}</div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Notes (optional)
              </label>
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                rows={3}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Additional notes about this classification"
              />
            </div>
          </div>

          <div className="flex justify-end space-x-3 mt-6">
            <button
              onClick={onClose}
              className="px-4 py-2 text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200"
            >
              Cancel
            </button>
            <button
              onClick={() => onSubmit({
                classificationId: data.classificationId,
                correctLabel,
                confidence: confidence / 10,
                notes: notes.trim() || null
              })}
              disabled={!correctLabel.trim()}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-300"
            >
              Submit Feedback
            </button>
          </div>
        </div>
      </div>
    );
  };

  if (imagesLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner />
      </div>
    );
  }

  if (imagesError) {
    return (
      <div className="text-center py-12">
        <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
        <h2 className="text-xl font-semibold text-gray-900 mb-2">
          Failed to Load Results
        </h2>
        <p className="text-gray-600 mb-4">
          There was an error loading the results. Please try again.
        </p>
        <button
          onClick={() => queryClient.invalidateQueries('images-list')}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!imagesList?.images?.length) {
    return (
      <div className="text-center py-12">
        <ImageIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
        <h2 className="text-xl font-semibold text-gray-900 mb-2">
          No Images Found
        </h2>
        <p className="text-gray-600 mb-4">
          Upload some images to see processing results here.
        </p>
        <button
          onClick={() => navigate('/upload')}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
        >
          Upload Images
        </button>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
      {/* Images List Sidebar */}
      <div className="lg:col-span-1">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <h3 className="text-lg font-medium text-gray-900 mb-4">
            Recent Images
          </h3>
          
          <div className="space-y-2">
            {imagesList.images.map((image) => (
              <div
                key={image.id}
                className={`
                  p-3 rounded-lg cursor-pointer transition-colors
                  ${selectedImage === image.id 
                    ? 'bg-blue-50 border border-blue-200' 
                    : 'hover:bg-gray-50'
                  }
                `}
                onClick={() => handleImageSelect(image.id)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {image.filename}
                    </p>
                    <p className="text-xs text-gray-500">
                      {new Date(image.created_at).toLocaleDateString()}
                    </p>
                  </div>
                  <StatusBadge status={image.status} />
                </div>
                
                <div className="mt-2 text-xs text-gray-600">
                  {image.segment_count} segments • {image.classification_count} classified
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="lg:col-span-3">
        {detailsLoading ? (
          <div className="flex items-center justify-center h-64">
            <LoadingSpinner />
          </div>
        ) : detailsError ? (
          <div className="text-center py-12">
            <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
            <h2 className="text-xl font-semibold text-gray-900 mb-2">
              Failed to Load Image Details
            </h2>
            <button
              onClick={handleRefresh}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
            >
              Retry
            </button>
          </div>
        ) : imageDetails ? (
          <div className="space-y-6">
            {/* Header */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-xl font-semibold text-gray-900">
                    {imageDetails.filename}
                  </h2>
                  <p className="text-gray-600">
                    {imageDetails.segments.length} segments • {imageDetails.statistics.total_classifications} classifications
                  </p>
                </div>
                
                <div className="flex items-center space-x-2">
                  <StatusBadge status={imageDetails.status} />
                  
                  <button
                    onClick={handleRefresh}
                    className="p-2 text-gray-600 hover:text-gray-800"
                    title="Refresh"
                  >
                    <RefreshCw className="w-4 h-4" />
                  </button>
                  
                  <div className="relative group">
                    <button className="p-2 text-gray-600 hover:text-gray-800">
                      <Download className="w-4 h-4" />
                    </button>
                    <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg border border-gray-200 py-1 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all">
                      <button
                        onClick={() => handleDownload('original')}
                        className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                      >
                        Original Image
                      </button>
                      <button
                        onClick={() => handleDownload('annotated')}
                        className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                      >
                        Annotated Image
                      </button>
                      <button
                        onClick={() => handleDownload('results')}
                        className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                      >
                        Results (JSON)
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* Statistics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {imageDetails.segments.length}
                  </div>
                  <div className="text-sm text-gray-600">Segments</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">
                    {imageDetails.statistics.unique_labels}
                  </div>
                  <div className="text-sm text-gray-600">Unique Items</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600">
                    {(imageDetails.statistics.average_confidence * 100).toFixed(0)}%
                  </div>
                  <div className="text-sm text-gray-600">Avg Confidence</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-orange-600">
                    {imageDetails.statistics.processing_time_seconds ? 
                      apiService.formatDuration(imageDetails.statistics.processing_time_seconds) : 
                      'N/A'
                    }
                  </div>
                  <div className="text-sm text-gray-600">Processing Time</div>
                </div>
              </div>
            </div>

            {/* Main Image */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium text-gray-900">
                  Annotated Results
                </h3>
                
                <div className="flex items-center space-x-4">
                  {/* Annotation controls */}
                  <div className="flex items-center space-x-2 text-sm">
                    <label className="flex items-center space-x-1">
                      <input
                        type="checkbox"
                        checked={annotationSettings.showLabels}
                        onChange={(e) => setAnnotationSettings(prev => ({
                          ...prev,
                          showLabels: e.target.checked
                        }))}
                        className="w-4 h-4"
                      />
                      <span>Labels</span>
                    </label>
                    
                    <label className="flex items-center space-x-1">
                      <input
                        type="checkbox"
                        checked={annotationSettings.showConfidence}
                        onChange={(e) => setAnnotationSettings(prev => ({
                          ...prev,
                          showConfidence: e.target.checked
                        }))}
                        className="w-4 h-4"
                      />
                      <span>Confidence</span>
                    </label>
                    
                    <label className="flex items-center space-x-1">
                      <input
                        type="checkbox"
                        checked={annotationSettings.showBbox}
                        onChange={(e) => setAnnotationSettings(prev => ({
                          ...prev,
                          showBbox: e.target.checked
                        }))}
                        className="w-4 h-4"
                      />
                      <span>Boxes</span>
                    </label>
                  </div>
                  
                  {/* View mode toggle */}
                  <div className="flex bg-gray-100 rounded-md">
                    <button
                      onClick={() => setViewMode('grid')}
                      className={`px-3 py-1 rounded-l-md text-sm ${
                        viewMode === 'grid' 
                          ? 'bg-white text-gray-900 shadow-sm' 
                          : 'text-gray-600'
                      }`}
                    >
                      <Grid3X3 className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => setViewMode('detail')}
                      className={`px-3 py-1 rounded-r-md text-sm ${
                        viewMode === 'detail' 
                          ? 'bg-white text-gray-900 shadow-sm' 
                          : 'text-gray-600'
                      }`}
                    >
                      <ZoomIn className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>

              <div className="text-center">
                <img
                  src={imageDetails.annotated_image_url}
                  alt="Annotated results"
                  className="max-w-full h-auto rounded-lg border border-gray-200"
                />
              </div>
            </div>

            {/* Segments */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">
                Detected Segments
              </h3>

              {viewMode === 'grid' ? (
                <SegmentGrid segments={imageDetails.segments} />
              ) : selectedSegment ? (
                <SegmentDetail segment={selectedSegment} />
              ) : (
                <div className="text-center py-8 text-gray-500">
                  Select a segment to view details
                </div>
              )}
            </div>

            {/* Selected Segment Detail in Grid Mode */}
            {viewMode === 'grid' && selectedSegment && (
              <SegmentDetail segment={selectedSegment} />
            )}
          </div>
        ) : null}
      </div>

      {/* Feedback Modal */}
      <FeedbackModal
        isOpen={showFeedbackModal}
        onClose={() => setShowFeedbackModal(false)}
        data={feedbackData}
        onSubmit={handleFeedbackSubmit}
      />
    </div>
  );
};

export default ResultsViewer;