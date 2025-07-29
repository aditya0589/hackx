#!/usr/bin/env python3
"""
Performance monitoring script for the RAG pipeline.
Tracks response times, accuracy, and system metrics.
"""

import time
import psutil
import json
from datetime import datetime
from typing import Dict, List, Any
import os

class PerformanceMonitor:
    def __init__(self, log_file="performance_log.json"):
        self.log_file = log_file
        self.metrics = []
        self.start_time = None
        
    def start_monitoring(self):
        """Start monitoring a request."""
        self.start_time = time.time()
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available': psutil.virtual_memory().available / (1024**3)  # GB
        }
    
    def end_monitoring(self, request_id: str, success: bool, error: str = None):
        """End monitoring and log metrics."""
        if self.start_time is None:
            return
        
        end_time = time.time()
        duration = end_time - self.start_time
        
        metric = {
            'request_id': request_id,
            'duration': duration,
            'success': success,
            'error': error,
            'end_timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available': psutil.virtual_memory().available / (1024**3)
        }
        
        self.metrics.append(metric)
        self._save_metrics()
        
        # Print summary
        print(f"[PERF] Request {request_id}: {duration:.2f}s ({'SUCCESS' if success else 'FAILED'})")
        
    def _save_metrics(self):
        """Save metrics to file."""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to save metrics: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.metrics:
            return {}
        
        successful_requests = [m for m in self.metrics if m['success']]
        failed_requests = [m for m in self.metrics if not m['success']]
        
        if successful_requests:
            durations = [m['duration'] for m in successful_requests]
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)
        else:
            avg_duration = min_duration = max_duration = 0
        
        return {
            'total_requests': len(self.metrics),
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'success_rate': len(successful_requests) / len(self.metrics) * 100,
            'avg_response_time': avg_duration,
            'min_response_time': min_duration,
            'max_response_time': max_duration,
            'total_duration': sum(m['duration'] for m in self.metrics)
        }
    
    def print_summary(self):
        """Print performance summary."""
        summary = self.get_performance_summary()
        if not summary:
            print("No performance data available.")
            return
        
        print("\n=== Performance Summary ===")
        print(f"Total Requests: {summary['total_requests']}")
        print(f"Successful: {summary['successful_requests']}")
        print(f"Failed: {summary['failed_requests']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Average Response Time: {summary['avg_response_time']:.2f}s")
        print(f"Min Response Time: {summary['min_response_time']:.2f}s")
        print(f"Max Response Time: {summary['max_response_time']:.2f}s")
        print(f"Total Duration: {summary['total_duration']:.2f}s")

# Global monitor instance
monitor = PerformanceMonitor()

def monitor_request(request_id: str):
    """Decorator to monitor request performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_metrics = monitor.start_monitoring()
            try:
                result = func(*args, **kwargs)
                monitor.end_monitoring(request_id, True)
                return result
            except Exception as e:
                monitor.end_monitoring(request_id, False, str(e))
                raise
        return wrapper
    return decorator

if __name__ == "__main__":
    # Example usage
    monitor.print_summary() 