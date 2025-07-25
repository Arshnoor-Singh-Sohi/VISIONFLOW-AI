#!/usr/bin/env python3
"""
VisionFlow AI - Main Scheduler Service
======================================

This is the central scheduling service that orchestrates all automated tasks
in VisionFlow AI. Think of this as the "conductor" of an orchestra, ensuring
that each task runs at the right time, in the right order, and with the right
resources available.

The scheduler handles:
- Task registration and management
- Cron-like scheduling with flexible timing
- Resource monitoring and task throttling
- Error handling and retry logic
- Health monitoring and alerting
- Task history and performance tracking

This service is designed to run as a long-lived daemon process, either as
a standalone service or as part of a containerized deployment.

Usage:
    python scheduler/scheduler.py [--config config.json] [--daemon] [--debug]
"""

import os
import sys
import asyncio
import signal
import logging
import json
import traceback
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import argparse

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scheduler.config import (
        get_scheduler_settings, SchedulerSettings, TaskDefinition,
        get_default_tasks, validate_schedule_format, get_next_run_time,
        is_resource_available
    )
    from scheduler.daily_processor import (
        process_daily_images, check_training_trigger, cleanup_old_logs,
        cleanup_temp_files, maintain_database, system_health_check,
        generate_daily_report
    )
    from backend.utils.logging import setup_logging, get_logger
except ImportError as e:
    print(f"Error importing VisionFlow modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Set up logging for this module
logger = get_logger(__name__)


# =============================================================================
# TASK EXECUTION ENGINE
# =============================================================================

class TaskExecution:
    """
    Represents a single execution of a scheduled task.
    
    This class tracks everything about a task run - when it started,
    how long it took, whether it succeeded, what errors occurred,
    and what resources it used. Think of it as a detailed work log
    for each task execution.
    """
    
    def __init__(self, task_def: TaskDefinition):
        self.task_def = task_def
        self.execution_id = f"{task_def.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.status = "pending"  # pending, running, completed, failed, timeout
        self.result: Optional[Dict[str, Any]] = None
        self.error_message: Optional[str] = None
        self.retry_count = 0
        self.resource_usage: Dict[str, Any] = {}
    
    @property
    def duration_seconds(self) -> float:
        """Calculate execution duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert execution record to dictionary for logging/storage."""
        return {
            'execution_id': self.execution_id,
            'task_name': self.task_def.name,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'status': self.status,
            'retry_count': self.retry_count,
            'error_message': self.error_message,
            'resource_usage': self.resource_usage,
            'result_summary': str(self.result)[:200] if self.result else None
        }


class TaskRunner:
    """
    Handles the execution of individual tasks with proper resource management,
    timeout handling, and error recovery.
    
    This class is like a careful task supervisor who makes sure each job
    is done properly, monitors resource usage, handles errors gracefully,
    and provides detailed reports on what happened.
    """
    
    def __init__(self, settings: SchedulerSettings):
        self.settings = settings
        self.running_tasks: Dict[str, TaskExecution] = {}
        
        # Map function names to actual callable functions
        self.function_registry = {
            'daily_processor.process_daily_images': process_daily_images,
            'daily_processor.check_training_trigger': check_training_trigger,
            'daily_processor.cleanup_old_logs': cleanup_old_logs,
            'daily_processor.cleanup_temp_files': cleanup_temp_files,
            'daily_processor.maintain_database': maintain_database,
            'daily_processor.system_health_check': system_health_check,
            'daily_processor.generate_daily_report': generate_daily_report,
        }
    
    async def execute_task(self, task_def: TaskDefinition) -> TaskExecution:
        """
        Execute a single task with full monitoring and error handling.
        
        This method acts like a task execution supervisor, managing the
        entire lifecycle of a task from pre-execution checks through
        completion and cleanup.
        """
        execution = TaskExecution(task_def)
        
        try:
            logger.info(f"Starting task execution: {task_def.name}")
            
            # Check if task is enabled
            if not task_def.enabled:
                execution.status = "skipped"
                execution.error_message = "Task is disabled"
                logger.info(f"Task {task_def.name} is disabled, skipping")
                return execution
            
            # Check resource availability
            resource_available, resource_message = is_resource_available(
                self.settings, task_def.required_resources
            )
            
            if not resource_available:
                execution.status = "failed"
                execution.error_message = f"Insufficient resources: {resource_message}"
                logger.warning(f"Task {task_def.name} skipped: {resource_message}")
                return execution
            
            # Check concurrent task limit
            if len(self.running_tasks) >= self.settings.max_concurrent_tasks:
                execution.status = "failed"
                execution.error_message = "Maximum concurrent tasks reached"
                logger.warning(f"Task {task_def.name} skipped: too many concurrent tasks")
                return execution
            
            # Get the function to execute
            func = self.function_registry.get(task_def.function)
            if not func:
                execution.status = "failed"
                execution.error_message = f"Unknown function: {task_def.function}"
                logger.error(f"Task {task_def.name} failed: function not found")
                return execution
            
            # Record task start
            execution.start_time = datetime.now(timezone.utc)
            execution.status = "running"
            self.running_tasks[execution.execution_id] = execution
            
            # Record initial resource usage
            execution.resource_usage = await self._get_resource_snapshot()
            
            # Execute the task with timeout
            try:
                result = await asyncio.wait_for(
                    func(**task_def.parameters),
                    timeout=task_def.timeout_minutes * 60
                )
                
                execution.result = result
                execution.status = "completed"
                logger.info(f"Task {task_def.name} completed successfully")
                
            except asyncio.TimeoutError:
                execution.status = "timeout"
                execution.error_message = f"Task timed out after {task_def.timeout_minutes} minutes"
                logger.error(f"Task {task_def.name} timed out")
                
            except Exception as e:
                execution.status = "failed"
                execution.error_message = str(e)
                logger.error(f"Task {task_def.name} failed: {e}")
                logger.debug(f"Task {task_def.name} traceback: {traceback.format_exc()}")
            
            # Record final resource usage
            final_resources = await self._get_resource_snapshot()
            execution.resource_usage.update({
                'memory_peak_mb': final_resources.get('memory_mb', 0),
                'cpu_time_seconds': final_resources.get('cpu_percent', 0) * execution.duration_seconds / 100
            })
            
        except Exception as e:
            # Catch any unexpected errors in the execution framework itself
            execution.status = "failed"
            execution.error_message = f"Execution framework error: {str(e)}"
            logger.error(f"Execution framework error for task {task_def.name}: {e}")
            
        finally:
            # Clean up and record completion
            execution.end_time = datetime.now(timezone.utc)
            
            if execution.execution_id in self.running_tasks:
                del self.running_tasks[execution.execution_id]
            
            logger.info(f"Task {task_def.name} finished: {execution.status} "
                       f"(duration: {execution.duration_seconds:.1f}s)")
        
        return execution
    
    async def _get_resource_snapshot(self) -> Dict[str, Any]:
        """Get current system resource usage for monitoring."""
        try:
            import psutil
            
            # Get current process info
            process = psutil.Process()
            
            return {
                'memory_mb': process.memory_info().rss / (1024 * 1024),
                'cpu_percent': process.cpu_percent(),
                'open_files': len(process.open_files()),
                'threads': process.num_threads()
            }
            
        except ImportError:
            return {}
        except Exception as e:
            logger.debug(f"Failed to get resource snapshot: {e}")
            return {}
    
    def get_running_tasks(self) -> List[Dict[str, Any]]:
        """Get information about currently running tasks."""
        return [execution.to_dict() for execution in self.running_tasks.values()]


# =============================================================================
# TASK SCHEDULER
# =============================================================================

class VisionFlowScheduler:
    """
    Main scheduler class that manages all scheduled tasks for VisionFlow AI.
    
    This is the "mission control" of the automation system. It keeps track
    of all scheduled tasks, determines when they should run, launches them
    at the right time, and monitors their progress. Think of it as a very
    sophisticated alarm clock that can handle complex schedules and manage
    multiple simultaneous activities.
    """
    
    def __init__(self, settings: SchedulerSettings):
        self.settings = settings
        self.task_runner = TaskRunner(settings)
        self.tasks: Dict[str, TaskDefinition] = {}
        self.execution_history: List[TaskExecution] = []
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Performance tracking
        self.stats = {
            'scheduler_started_at': datetime.now(timezone.utc),
            'total_tasks_executed': 0,
            'total_tasks_succeeded': 0,
            'total_tasks_failed': 0,
            'last_health_check': None
        }
        
        logger.info("VisionFlow Scheduler initialized")
    
    def register_task(self, task_def: TaskDefinition):
        """
        Register a new task with the scheduler.
        
        This method adds a new task to the scheduler's task list and
        validates that it's properly configured before accepting it.
        """
        # Validate task definition
        if not validate_schedule_format(task_def.schedule):
            raise ValueError(f"Invalid schedule format for task {task_def.name}: {task_def.schedule}")
        
        if task_def.function not in self.task_runner.function_registry:
            raise ValueError(f"Unknown function for task {task_def.name}: {task_def.function}")
        
        self.tasks[task_def.name] = task_def
        logger.info(f"Registered task: {task_def.name} ({task_def.schedule})")
    
    def register_default_tasks(self):
        """Register all default tasks based on current settings."""
        default_tasks = get_default_tasks(self.settings)
        
        for task_def in default_tasks:
            try:
                self.register_task(task_def)
            except Exception as e:
                logger.error(f"Failed to register default task {task_def.name}: {e}")
    
    async def start(self):
        """
        Start the scheduler's main execution loop.
        
        This method begins the scheduler's main loop, which continuously
        checks for tasks that need to be executed and launches them at
        the appropriate times.
        """
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        logger.info("Starting VisionFlow Scheduler")
        self.is_running = True
        
        # Register default tasks
        self.register_default_tasks()
        
        # Start the main scheduling loop
        try:
            await self._main_loop()
        except Exception as e:
            logger.error(f"Scheduler main loop failed: {e}")
            raise
        finally:
            self.is_running = False
            logger.info("VisionFlow Scheduler stopped")
    
    async def stop(self):
        """
        Gracefully stop the scheduler.
        
        This method signals the scheduler to shut down gracefully,
        allowing currently running tasks to complete before stopping.
        """
        logger.info("Stopping VisionFlow Scheduler...")
        self.shutdown_event.set()
        
        # Wait for running tasks to complete (with timeout)
        if self.task_runner.running_tasks:
            logger.info(f"Waiting for {len(self.task_runner.running_tasks)} running tasks to complete...")
            
            # Wait up to 5 minutes for tasks to complete
            for _ in range(300):  # 5 minutes in seconds
                if not self.task_runner.running_tasks:
                    break
                await asyncio.sleep(1)
            
            if self.task_runner.running_tasks:
                logger.warning(f"Forcing shutdown with {len(self.task_runner.running_tasks)} tasks still running")
    
    async def _main_loop(self):
        """
        Main scheduler loop that checks for tasks to execute.
        
        This is the heart of the scheduler - it continuously checks
        what time it is, compares that to the schedule of each task,
        and launches tasks when their time comes.
        """
        logger.info("Scheduler main loop started")
        
        # Track next execution times for all tasks
        next_executions: Dict[str, datetime] = {}
        
        # Calculate initial next execution times
        for task_name, task_def in self.tasks.items():
            try:
                next_executions[task_name] = get_next_run_time(task_def.schedule)
                logger.debug(f"Task {task_name} next scheduled: {next_executions[task_name]}")
            except Exception as e:
                logger.error(f"Failed to calculate next run time for {task_name}: {e}")
        
        # Main loop
        while not self.shutdown_event.is_set():
            try:
                current_time = datetime.now(timezone.utc)
                
                # Check each task to see if it should run
                for task_name, task_def in self.tasks.items():
                    if task_name not in next_executions:
                        continue
                    
                    next_run_time = next_executions[task_name]
                    
                    # If it's time to run this task
                    if current_time >= next_run_time:
                        logger.debug(f"Task {task_name} is due to run")
                        
                        # Launch the task
                        asyncio.create_task(self._execute_task_with_tracking(task_def))
                        
                        # Calculate next execution time
                        try:
                            next_executions[task_name] = get_next_run_time(task_def.schedule)
                            logger.debug(f"Task {task_name} rescheduled for: {next_executions[task_name]}")
                        except Exception as e:
                            logger.error(f"Failed to reschedule task {task_name}: {e}")
                            # Remove from schedule if we can't calculate next run time
                            del next_executions[task_name]
                
                # Cleanup old execution history (keep last 100 executions)
                if len(self.execution_history) > 100:
                    self.execution_history = self.execution_history[-100:]
                
                # Sleep for a short time before checking again
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in scheduler main loop: {e}")
                await asyncio.sleep(60)  # Wait longer after errors
    
    async def _execute_task_with_tracking(self, task_def: TaskDefinition):
        """
        Execute a task with full tracking and statistics updating.
        
        This method wraps task execution with additional tracking,
        retry logic, and statistics updates.
        """
        max_retries = task_def.retry_attempts
        retry_delay = task_def.retry_delay_minutes * 60  # Convert to seconds
        
        for attempt in range(max_retries + 1):
            try:
                # Execute the task
                execution = await self.task_runner.execute_task(task_def)
                execution.retry_count = attempt
                
                # Update statistics
                self.stats['total_tasks_executed'] += 1
                if execution.status == "completed":
                    self.stats['total_tasks_succeeded'] += 1
                else:
                    self.stats['total_tasks_failed'] += 1
                
                # Store execution history
                self.execution_history.append(execution)
                
                # Log execution details
                self._log_task_execution(execution)
                
                # If task succeeded or reached max retries, we're done
                if execution.status == "completed" or attempt >= max_retries:
                    break
                
                # If task failed and we have retries left, wait and try again
                if attempt < max_retries:
                    logger.info(f"Task {task_def.name} failed (attempt {attempt + 1}/{max_retries + 1}), "
                               f"retrying in {retry_delay}s")
                    await asyncio.sleep(retry_delay)
                
            except Exception as e:
                logger.error(f"Unexpected error executing task {task_def.name} (attempt {attempt + 1}): {e}")
                
                if attempt >= max_retries:
                    # Create a failed execution record
                    execution = TaskExecution(task_def)
                    execution.status = "failed"
                    execution.error_message = f"Execution framework error: {str(e)}"
                    execution.retry_count = attempt
                    self.execution_history.append(execution)
                    self.stats['total_tasks_failed'] += 1
                    break
                
                await asyncio.sleep(retry_delay)
    
    def _log_task_execution(self, execution: TaskExecution):
        """Log detailed information about a task execution."""
        if execution.status == "completed":
            logger.info(f"✓ Task completed: {execution.task_def.name} "
                       f"(duration: {execution.duration_seconds:.1f}s, "
                       f"retries: {execution.retry_count})")
        else:
            logger.warning(f"✗ Task failed: {execution.task_def.name} "
                          f"(status: {execution.status}, "
                          f"retries: {execution.retry_count}, "
                          f"error: {execution.error_message})")
        
        # Log resource usage if available
        if execution.resource_usage:
            memory_mb = execution.resource_usage.get('memory_peak_mb', 0)
            if memory_mb > 0:
                logger.debug(f"Task {execution.task_def.name} resource usage: "
                           f"{memory_mb:.1f}MB memory")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive scheduler status information."""
        current_time = datetime.now(timezone.utc)
        uptime_seconds = (current_time - self.stats['scheduler_started_at']).total_seconds()
        
        # Calculate success rate
        total_executed = self.stats['total_tasks_executed']
        success_rate = (self.stats['total_tasks_succeeded'] / max(total_executed, 1)) * 100
        
        # Get information about recent executions
        recent_executions = [
            exec.to_dict() for exec in self.execution_history[-10:]
        ]
        
        return {
            'scheduler_status': 'running' if self.is_running else 'stopped',
            'uptime_seconds': uptime_seconds,
            'registered_tasks': len(self.tasks),
            'running_tasks': len(self.task_runner.running_tasks),
            'statistics': {
                **self.stats,
                'success_rate_percent': success_rate,
                'uptime_hours': uptime_seconds / 3600
            },
            'recent_executions': recent_executions,
            'task_summary': [
                {
                    'name': task.name,
                    'schedule': task.schedule,
                    'enabled': task.enabled,
                    'description': task.description
                }
                for task in self.tasks.values()
            ]
        }
    
    def get_task_history(self, task_name: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get execution history for a specific task."""
        task_executions = [
            exec.to_dict() for exec in self.execution_history
            if exec.task_def.name == task_name
        ]
        
        # Sort by start time (most recent first) and limit
        task_executions.sort(key=lambda x: x['start_time'] or '', reverse=True)
        return task_executions[:limit]


# =============================================================================
# SCHEDULER DAEMON
# =============================================================================

class SchedulerDaemon:
    """
    Daemon wrapper for the VisionFlow Scheduler.
    
    This class handles the setup and lifecycle management of the scheduler
    when running as a daemon process. It manages signal handling, logging
    setup, configuration loading, and graceful shutdown procedures.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.scheduler: Optional[VisionFlowScheduler] = None
        self.settings: Optional[SchedulerSettings] = None
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        if hasattr(signal, 'SIGHUP'):  # Unix only
            signal.signal(signal.SIGHUP, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        
        if self.scheduler:
            asyncio.create_task(self.scheduler.stop())
    
    async def load_configuration(self) -> SchedulerSettings:
        """Load scheduler configuration from various sources."""
        if self.config_path and os.path.exists(self.config_path):
            logger.info(f"Loading configuration from: {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            # Create settings with override values
            self.settings = SchedulerSettings(**config_data)
        else:
            # Load from environment and defaults
            logger.info("Loading configuration from environment variables and defaults")
            self.settings = get_scheduler_settings()
        
        return self.settings
    
    async def run(self):
        """
        Run the scheduler daemon.
        
        This is the main entry point for running the scheduler as a daemon.
        It handles all setup, configuration loading, and cleanup.
        """
        try:
            # Load configuration
            await self.load_configuration()
            
            logger.info("Starting VisionFlow AI Scheduler Daemon")
            logger.info(f"Configuration: {self.settings.enabled} enabled, "
                       f"timezone: {self.settings.timezone}")
            
            # Check if scheduler is enabled
            if not self.settings.enabled:
                logger.info("Scheduler is disabled in configuration")
                return
            
            # Create and start scheduler
            self.scheduler = VisionFlowScheduler(self.settings)
            
            # Start the scheduler
            await self.scheduler.start()
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Scheduler daemon failed: {e}")
            logger.debug(f"Scheduler daemon traceback: {traceback.format_exc()}")
            raise
        finally:
            # Cleanup
            if self.scheduler:
                await self.scheduler.stop()
            
            logger.info("VisionFlow AI Scheduler Daemon stopped")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main():
    """
    Main entry point for the scheduler service.
    
    This function handles command-line argument parsing, logging setup,
    and starts the appropriate scheduler mode (daemon or one-time).
    """
    parser = argparse.ArgumentParser(
        description="VisionFlow AI Scheduler Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run with default configuration
  %(prog)s --config config.json     # Run with custom configuration
  %(prog)s --daemon                 # Run as background daemon
  %(prog)s --debug                  # Run with debug logging
  %(prog)s --status                 # Show scheduler status and exit
        """
    )
    
    parser.add_argument(
        '--config',
        help='Path to JSON configuration file'
    )
    parser.add_argument(
        '--daemon',
        action='store_true',
        help='Run as background daemon'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show scheduler status and exit'
    )
    parser.add_argument(
        '--test-task',
        help='Run a single task for testing (task name)'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logging(log_level=log_level)
    
    try:
        if args.status:
            # Show status and exit
            await show_scheduler_status()
            return
        
        if args.test_task:
            # Run a single task for testing
            await test_single_task(args.test_task, args.config)
            return
        
        # Create and run scheduler daemon
        daemon = SchedulerDaemon(config_path=args.config)
        
        if args.daemon:
            # TODO: Implement proper daemon mode with PID file
            logger.info("Daemon mode not fully implemented, running in foreground")
        
        await daemon.run()
        
    except Exception as e:
        logger.error(f"Scheduler failed to start: {e}")
        sys.exit(1)


async def show_scheduler_status():
    """Show current scheduler status (if running)."""
    # This would typically connect to a running scheduler instance
    # For now, just show configuration status
    
    settings = get_scheduler_settings()
    
    print("VisionFlow AI Scheduler Status")
    print("=" * 40)
    print(f"Enabled: {settings.enabled}")
    print(f"Timezone: {settings.timezone}")
    print(f"Daily Processing: {settings.daily_processing_enabled} at {settings.daily_processing_time}")
    print(f"Auto Training: {settings.auto_training_enabled}")
    print(f"Maintenance: {settings.maintenance_enabled}")
    print(f"Health Checks: Every {settings.health_check_interval_minutes} minutes")
    print()
    
    # Show default tasks that would be scheduled
    default_tasks = get_default_tasks(settings)
    print(f"Default Tasks ({len(default_tasks)}):")
    for task in default_tasks:
        status = "✓" if task.enabled else "✗"
        print(f"  {status} {task.name}: {task.schedule}")
        print(f"    {task.description}")
    print()


async def test_single_task(task_name: str, config_path: Optional[str] = None):
    """Run a single task for testing purposes."""
    print(f"Testing task: {task_name}")
    print("=" * 40)
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        settings = SchedulerSettings(**config_data)
    else:
        settings = get_scheduler_settings()
    
    # Get default tasks
    tasks = get_default_tasks(settings)
    task_def = None
    
    for task in tasks:
        if task.name == task_name:
            task_def = task
            break
    
    if not task_def:
        print(f"Task '{task_name}' not found!")
        print("Available tasks:")
        for task in tasks:
            print(f"  - {task.name}")
        return
    
    # Create task runner and execute
    task_runner = TaskRunner(settings)
    
    print(f"Executing task: {task_def.name}")
    print(f"Function: {task_def.function}")
    print(f"Parameters: {task_def.parameters}")
    print()
    
    execution = await task_runner.execute_task(task_def)
    
    print("Execution Results:")
    print(f"Status: {execution.status}")
    print(f"Duration: {execution.duration_seconds:.1f} seconds")
    
    if execution.error_message:
        print(f"Error: {execution.error_message}")
    
    if execution.result:
        print("Result:")
        if isinstance(execution.result, dict):
            for key, value in execution.result.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {execution.result}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScheduler interrupted by user")
    except Exception as e:
        print(f"Scheduler failed: {e}")
        sys.exit(1)