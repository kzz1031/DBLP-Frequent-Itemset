import time
import psutil
import os
from typing import Dict, Any
from collections import defaultdict

class PerformanceMonitor:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.memory_usage = []
        self.metrics = defaultdict(int)
    
    def start(self):
        """开始监控"""
        self.start_time = time.time()
        self.memory_usage = []
    
    def record_memory(self):
        """记录当前内存使用"""
        process = psutil.Process(os.getpid())
        self.memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
    
    def end(self):
        """结束监控"""
        self.end_time = time.time()
    
    def add_metric(self, name: str, value: int):
        """添加自定义指标"""
        self.metrics[name] += value
    
    def get_results(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            'execution_time': self.end_time - self.start_time,
            'peak_memory': max(self.memory_usage) if self.memory_usage else 0,
            'metrics': dict(self.metrics)
        }
