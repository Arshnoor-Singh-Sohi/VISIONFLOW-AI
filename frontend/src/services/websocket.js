import io from 'socket.io-client';

class WebSocketService {
  constructor() {
    this.socket = null;
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000; // Start with 1 second
    this.maxReconnectDelay = 30000; // Max 30 seconds
    this.eventHandlers = {
      onConnect: () => {},
      onDisconnect: () => {},
      onMessage: () => {},
      onError: () => {}
    };
    this.subscriptions = new Set();
  }

  /**
   * Connect to the WebSocket server
   * @param {Object} handlers - Event handlers
   */
  connect(handlers = {}) {
    // Store event handlers
    this.eventHandlers = {
      onConnect: handlers.onConnect || (() => {}),
      onDisconnect: handlers.onDisconnect || (() => {}),
      onMessage: handlers.onMessage || (() => {}),
      onError: handlers.onError || (() => {})
    };

    // Get WebSocket URL from environment or default
    const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';
    
    // Create socket connection
    this.socket = io(wsUrl, {
      transports: ['websocket', 'polling'],
      upgrade: true,
      rememberUpgrade: true,
      timeout: 20000,
      forceNew: true
    });

    this.setupEventListeners();
  }

  /**
   * Set up socket event listeners
   */
  setupEventListeners() {
    if (!this.socket) return;

    // Connection established
    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      this.isConnected = true;
      this.reconnectAttempts = 0;
      this.reconnectDelay = 1000;
      
      // Send initial subscription requests
      this.resubscribe();
      
      this.eventHandlers.onConnect();
    });

    // Connection lost
    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      this.isConnected = false;
      this.eventHandlers.onDisconnect(reason);

      // Attempt reconnection for certain disconnect reasons
      if (reason === 'io server disconnect') {
        // Server initiated disconnect, don't reconnect automatically
        console.log('Server disconnected the client');
      } else {
        // Client-side disconnect, attempt reconnection
        this.scheduleReconnect();
      }
    });

    // Connection error
    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      this.isConnected = false;
      this.eventHandlers.onError(error);
      this.scheduleReconnect();
    });

    // Reconnection attempt
    this.socket.on('reconnect_attempt', (attemptNumber) => {
      console.log(`WebSocket reconnection attempt ${attemptNumber}`);
    });

    // Reconnection successful
    this.socket.on('reconnect', (attemptNumber) => {
      console.log(`WebSocket reconnected after ${attemptNumber} attempts`);
      this.isConnected = true;
      this.reconnectAttempts = 0;
      this.reconnectDelay = 1000;
      this.resubscribe();
    });

    // Reconnection failed
    this.socket.on('reconnect_failed', () => {
      console.error('WebSocket reconnection failed');
      this.isConnected = false;
    });

    // Handle incoming messages
    this.socket.on('message', (data) => {
      this.handleMessage(data);
    });

    // Handle specific event types
    this.socket.on('processing_update', (data) => {
      this.handleMessage({ type: 'processing_update', ...data });
    });

    this.socket.on('training_update', (data) => {
      this.handleMessage({ type: 'training_update', ...data });
    });

    this.socket.on('health_alert', (data) => {
      this.handleMessage({ type: 'health_alert', ...data });
    });

    this.socket.on('system_notification', (data) => {
      this.handleMessage({ type: 'system_notification', ...data });
    });
  }

  /**
   * Handle incoming messages
   * @param {Object} message - The received message
   */
  handleMessage(message) {
    try {
      // Add timestamp if not present
      if (!message.timestamp) {
        message.timestamp = new Date().toISOString();
      }

      // Log message in development
      if (process.env.NODE_ENV === 'development') {
        console.log('WebSocket message received:', message);
      }

      // Call the message handler
      this.eventHandlers.onMessage(message);
    } catch (error) {
      console.error('Error handling WebSocket message:', error);
    }
  }

  /**
   * Send a message to the server
   * @param {Object} message - Message to send
   */
  send(message) {
    if (!this.socket || !this.isConnected) {
      console.warn('WebSocket not connected, cannot send message:', message);
      return false;
    }

    try {
      this.socket.emit('message', message);
      return true;
    } catch (error) {
      console.error('Error sending WebSocket message:', error);
      return false;
    }
  }

  /**
   * Subscribe to specific event types
   * @param {Array|string} events - Event types to subscribe to
   */
  subscribe(events) {
    const eventArray = Array.isArray(events) ? events : [events];
    
    eventArray.forEach(event => {
      this.subscriptions.add(event);
    });

    if (this.isConnected) {
      this.send({
        type: 'subscribe',
        events: eventArray
      });
    }
  }

  /**
   * Unsubscribe from specific event types
   * @param {Array|string} events - Event types to unsubscribe from
   */
  unsubscribe(events) {
    const eventArray = Array.isArray(events) ? events : [events];
    
    eventArray.forEach(event => {
      this.subscriptions.delete(event);
    });

    if (this.isConnected) {
      this.send({
        type: 'unsubscribe',
        events: eventArray
      });
    }
  }

  /**
   * Resubscribe to all events after reconnection
   */
  resubscribe() {
    if (this.subscriptions.size > 0) {
      this.send({
        type: 'subscribe',
        events: Array.from(this.subscriptions)
      });
    }
  }

  /**
   * Send a ping to keep connection alive
   */
  ping() {
    if (this.isConnected) {
      this.send({
        type: 'ping',
        timestamp: Date.now()
      });
    }
  }

  /**
   * Schedule reconnection with exponential backoff
   */
  scheduleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    
    setTimeout(() => {
      if (!this.isConnected && this.socket) {
        console.log(`Attempting to reconnect (attempt ${this.reconnectAttempts})`);
        this.socket.connect();
      }
    }, this.reconnectDelay);

    // Exponential backoff
    this.reconnectDelay = Math.min(
      this.reconnectDelay * 2,
      this.maxReconnectDelay
    );
  }

  /**
   * Disconnect from the WebSocket server
   */
  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    this.isConnected = false;
    this.subscriptions.clear();
    this.reconnectAttempts = 0;
  }

  /**
   * Get connection status
   */
  getConnectionStatus() {
    return {
      isConnected: this.isConnected,
      reconnectAttempts: this.reconnectAttempts,
      subscriptions: Array.from(this.subscriptions)
    };
  }

  /**
   * Update event handlers
   * @param {Object} handlers - New event handlers
   */
  updateHandlers(handlers) {
    this.eventHandlers = {
      ...this.eventHandlers,
      ...handlers
    };
  }
}

// Create singleton instance
const websocketService = new WebSocketService();

// Auto-connect functionality with retry
export const connectWithRetry = (handlers, maxAttempts = 3) => {
  let attempts = 0;
  
  const attemptConnection = () => {
    attempts++;
    
    try {
      websocketService.connect({
        ...handlers,
        onError: (error) => {
          console.error(`Connection attempt ${attempts} failed:`, error);
          
          if (attempts < maxAttempts) {
            console.log(`Retrying connection in ${attempts * 2} seconds...`);
            setTimeout(attemptConnection, attempts * 2000);
          } else {
            console.error('All connection attempts failed');
            if (handlers.onError) {
              handlers.onError(error);
            }
          }
        }
      });
    } catch (error) {
      console.error('Failed to initiate connection:', error);
      if (attempts < maxAttempts) {
        setTimeout(attemptConnection, attempts * 2000);
      }
    }
  };
  
  attemptConnection();
};

// Utility functions for common operations
export const websocketUtils = {
  /**
   * Subscribe to processing updates for a specific image
   * @param {string} imageId - Image ID to monitor
   */
  subscribeToImageProcessing(imageId) {
    websocketService.subscribe([
      'processing_update',
      'image_completed',
      'image_failed'
    ]);
    
    // Send specific image subscription
    websocketService.send({
      type: 'monitor_image',
      image_id: imageId
    });
  },

  /**
   * Subscribe to training updates
   */
  subscribeToTraining() {
    websocketService.subscribe([
      'training_started',
      'training_progress',
      'training_completed',
      'training_failed'
    ]);
  },

  /**
   * Subscribe to system health alerts
   */
  subscribeToSystemHealth() {
    websocketService.subscribe([
      'health_alert',
      'service_status_change',
      'system_notification'
    ]);
  },

  /**
   * Start periodic ping to keep connection alive
   * @param {number} interval - Ping interval in milliseconds
   */
  startHeartbeat(interval = 30000) {
    return setInterval(() => {
      websocketService.ping();
    }, interval);
  },

  /**
   * Format WebSocket messages for UI display
   * @param {Object} message - WebSocket message
   */
  formatMessageForUI(message) {
    const formatters = {
      processing_update: (msg) => ({
        title: 'Processing Update',
        content: `Image ${msg.image_id}: ${msg.status}`,
        type: 'info',
        timestamp: msg.timestamp
      }),
      
      training_update: (msg) => ({
        title: 'Training Update',
        content: `Training run ${msg.training_run_id}: ${msg.status}`,
        type: msg.status === 'completed' ? 'success' : 'info',
        timestamp: msg.timestamp
      }),
      
      health_alert: (msg) => ({
        title: 'System Alert',
        content: `System status: ${msg.status}`,
        type: msg.status === 'healthy' ? 'success' : 'warning',
        timestamp: msg.timestamp
      }),
      
      system_notification: (msg) => ({
        title: msg.title || 'System Notification',
        content: msg.message,
        type: msg.level || 'info',
        timestamp: msg.timestamp
      })
    };

    const formatter = formatters[message.type];
    return formatter ? formatter(message) : {
      title: 'Notification',
      content: message.message || JSON.stringify(message),
      type: 'info',
      timestamp: message.timestamp || new Date().toISOString()
    };
  }
};

export { websocketService };
export default websocketService;