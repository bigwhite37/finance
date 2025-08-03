"""
异步回撤计算器

该模块提供异步回撤计算功能，包括：
- 异步回撤计算
- 流式回撤处理
- 并行投资组合分析
- 异步滚动分析
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, AsyncGenerator, Optional
import logging
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class AsyncDrawdownCalculator:
    """异步回撤计算器"""
    
    def __init__(self, max_workers: int = 4):
        """
        初始化异步回撤计算器
        
        Args:
            max_workers: 最大工作线程数
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 流式处理状态
        self.streaming_values = []
        self.streaming_peaks = []
        self.streaming_drawdowns = []
        
        logger.info(f"异步回撤计算器初始化完成，最大工作线程: {max_workers}")
    
    async def calculate_drawdown_async(self, portfolio_values: np.ndarray) -> Dict[str, Any]:
        """
        异步计算回撤指标
        
        Args:
            portfolio_values: 投资组合净值序列
            
        Returns:
            Dict: 回撤指标
        """
        loop = asyncio.get_event_loop()
        
        # 在线程池中执行计算密集型任务
        result = await loop.run_in_executor(
            self.executor, 
            self._calculate_drawdown_sync, 
            portfolio_values
        )
        
        return result
    
    def _calculate_drawdown_sync(self, portfolio_values: np.ndarray) -> Dict[str, Any]:
        """同步回撤计算（在线程池中执行）"""
        if len(portfolio_values) < 2:
            return {
                'current_drawdown': 0.0,
                'max_drawdown': 0.0,
                'drawdown_series': [],
                'peak_values': [],
                'trough_indices': [],
                'underwater_curve': []
            }
        
        # 计算滚动最大值
        running_max = np.maximum.accumulate(portfolio_values)
        
        # 计算回撤序列
        drawdown_series = (portfolio_values - running_max) / running_max
        
        # 当前回撤和最大回撤
        current_drawdown = drawdown_series[-1]
        max_drawdown = np.min(drawdown_series)
        
        # 找到谷值位置
        trough_indices = self._find_trough_indices(drawdown_series)
        
        # 计算水下曲线
        underwater_curve = self._calculate_underwater_curve(drawdown_series)
        
        return {
            'current_drawdown': float(current_drawdown),
            'max_drawdown': float(max_drawdown),
            'drawdown_series': drawdown_series.tolist(),
            'peak_values': running_max.tolist(),
            'trough_indices': trough_indices,
            'underwater_curve': underwater_curve,
            'calculation_time': time.time()
        }
    
    def _find_trough_indices(self, drawdown_series: np.ndarray) -> List[int]:
        """找到回撤谷值的索引"""
        trough_indices = []
        in_drawdown = False
        current_trough_idx = -1
        current_trough_value = 0.0
        
        for i, dd in enumerate(drawdown_series):
            if dd < -0.001 and not in_drawdown:
                # 开始回撤
                in_drawdown = True
                current_trough_idx = i
                current_trough_value = dd
            elif dd < current_trough_value and in_drawdown:
                # 更深的回撤
                current_trough_idx = i
                current_trough_value = dd
            elif dd >= -0.001 and in_drawdown:
                # 回撤结束
                trough_indices.append(current_trough_idx)
                in_drawdown = False
        
        # 如果结束时仍在回撤中
        if in_drawdown:
            trough_indices.append(current_trough_idx)
        
        return trough_indices
    
    def _calculate_underwater_curve(self, drawdown_series: np.ndarray) -> List[int]:
        """计算水下曲线（连续回撤天数）"""
        underwater = []
        current_underwater = 0
        
        for dd in drawdown_series:
            if dd < -0.001:
                current_underwater += 1
            else:
                current_underwater = 0
            underwater.append(current_underwater)
        
        return underwater
    
    async def update_drawdown_streaming(self, new_value: float) -> Dict[str, Any]:
        """
        流式更新回撤计算
        
        Args:
            new_value: 新的投资组合净值
            
        Returns:
            Dict: 更新后的回撤指标
        """
        self.streaming_values.append(new_value)
        
        # 限制历史数据长度
        max_history = 10000
        if len(self.streaming_values) > max_history:
            self.streaming_values = self.streaming_values[-max_history:]
            self.streaming_peaks = self.streaming_peaks[-max_history:]
            self.streaming_drawdowns = self.streaming_drawdowns[-max_history:]
        
        # 更新峰值
        if not self.streaming_peaks or new_value > self.streaming_peaks[-1]:
            current_peak = new_value
        else:
            current_peak = self.streaming_peaks[-1]
        
        self.streaming_peaks.append(current_peak)
        
        # 计算当前回撤
        current_drawdown = (new_value - current_peak) / current_peak
        self.streaming_drawdowns.append(current_drawdown)
        
        # 计算统计指标
        max_drawdown = min(self.streaming_drawdowns) if self.streaming_drawdowns else 0.0
        
        return {
            'current_value': new_value,
            'current_peak': current_peak,
            'current_drawdown': current_drawdown,
            'max_drawdown': max_drawdown,
            'data_points': len(self.streaming_values),
            'update_time': time.time()
        }
    
    async def calculate_multiple_portfolios_async(self, 
                                                portfolios: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """
        并行计算多个投资组合的回撤
        
        Args:
            portfolios: 投资组合字典 {名称: 净值序列}
            
        Returns:
            Dict: 每个投资组合的回撤结果
        """
        # 创建并发任务
        tasks = {}
        for name, values in portfolios.items():
            task = asyncio.create_task(self.calculate_drawdown_async(values))
            tasks[name] = task
        
        # 等待所有任务完成
        results = {}
        for name, task in tasks.items():
            try:
                result = await task
                results[name] = result
            except Exception as e:
                logger.error(f"计算投资组合 {name} 回撤失败: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    async def calculate_rolling_drawdown_async(self, 
                                             portfolio_values: np.ndarray,
                                             window_size: int = 252) -> List[Dict[str, Any]]:
        """
        异步滚动回撤分析
        
        Args:
            portfolio_values: 投资组合净值序列
            window_size: 滚动窗口大小
            
        Returns:
            List: 滚动回撤结果列表
        """
        if len(portfolio_values) < window_size:
            raise ValueError(f"数据长度 {len(portfolio_values)} 小于窗口大小 {window_size}")
        
        # 创建滚动窗口任务
        tasks = []
        for i in range(len(portfolio_values) - window_size + 1):
            window_data = portfolio_values[i:i + window_size]
            task = asyncio.create_task(self._calculate_rolling_window(window_data, i))
            tasks.append(task)
        
        # 批量执行任务以控制并发
        batch_size = self.max_workers * 2
        results = []
        
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"滚动窗口计算失败: {result}")
                else:
                    results.append(result)
        
        return results
    
    async def _calculate_rolling_window(self, window_data: np.ndarray, window_index: int) -> Dict[str, Any]:
        """计算单个滚动窗口的回撤"""
        loop = asyncio.get_event_loop()
        
        # 在线程池中执行计算
        drawdown_result = await loop.run_in_executor(
            self.executor,
            self._calculate_drawdown_sync,
            window_data
        )
        
        # 添加窗口信息
        drawdown_result.update({
            'window_index': window_index,
            'window_start': window_index,
            'window_end': window_index + len(window_data) - 1,
            'window_size': len(window_data)
        })
        
        return drawdown_result
    
    async def analyze_drawdown_patterns_async(self, 
                                            portfolio_values: np.ndarray) -> Dict[str, Any]:
        """
        异步分析回撤模式
        
        Args:
            portfolio_values: 投资组合净值序列
            
        Returns:
            Dict: 回撤模式分析结果
        """
        # 先计算基本回撤指标
        basic_result = await self.calculate_drawdown_async(portfolio_values)
        
        # 在线程池中执行模式分析
        loop = asyncio.get_event_loop()
        pattern_analysis = await loop.run_in_executor(
            self.executor,
            self._analyze_patterns_sync,
            basic_result
        )
        
        return {
            **basic_result,
            'pattern_analysis': pattern_analysis
        }
    
    def _analyze_patterns_sync(self, drawdown_result: Dict[str, Any]) -> Dict[str, Any]:
        """同步分析回撤模式"""
        drawdown_series = np.array(drawdown_result['drawdown_series'])
        underwater_curve = drawdown_result['underwater_curve']
        
        # 回撤事件识别
        drawdown_events = self._identify_drawdown_events(drawdown_series)
        
        # 回撤统计
        if drawdown_events:
            avg_drawdown_magnitude = np.mean([event['magnitude'] for event in drawdown_events])
            avg_drawdown_duration = np.mean([event['duration'] for event in drawdown_events])
            max_underwater_duration = max(underwater_curve) if underwater_curve else 0
        else:
            avg_drawdown_magnitude = 0.0
            avg_drawdown_duration = 0.0
            max_underwater_duration = 0
        
        return {
            'drawdown_events': drawdown_events,
            'event_count': len(drawdown_events),
            'avg_magnitude': float(avg_drawdown_magnitude),
            'avg_duration': float(avg_drawdown_duration),
            'max_underwater_duration': max_underwater_duration,
            'drawdown_frequency': len(drawdown_events) / len(drawdown_series) * 252  # 年化频率
        }
    
    def _identify_drawdown_events(self, drawdown_series: np.ndarray) -> List[Dict[str, Any]]:
        """识别回撤事件"""
        events = []
        in_drawdown = False
        start_idx = 0
        
        for i, dd in enumerate(drawdown_series):
            if dd < -0.01 and not in_drawdown:  # 开始回撤（1%阈值）
                in_drawdown = True
                start_idx = i
            elif dd >= -0.001 and in_drawdown:  # 结束回撤
                end_idx = i
                magnitude = np.min(drawdown_series[start_idx:end_idx+1])
                
                events.append({
                    'start_index': start_idx,
                    'end_index': end_idx,
                    'duration': end_idx - start_idx + 1,
                    'magnitude': float(magnitude),
                    'recovery_time': None  # 可以进一步计算恢复时间
                })
                
                in_drawdown = False
        
        return events
    
    def get_streaming_statistics(self) -> Dict[str, Any]:
        """获取流式处理统计信息"""
        if not self.streaming_values:
            return {'status': 'no_data'}
        
        return {
            'total_values': len(self.streaming_values),
            'current_value': self.streaming_values[-1],
            'current_peak': self.streaming_peaks[-1] if self.streaming_peaks else 0,
            'current_drawdown': self.streaming_drawdowns[-1] if self.streaming_drawdowns else 0,
            'max_drawdown': min(self.streaming_drawdowns) if self.streaming_drawdowns else 0,
            'total_peaks': sum(1 for i in range(1, len(self.streaming_peaks)) 
                             if self.streaming_peaks[i] > self.streaming_peaks[i-1])
        }
    
    def reset_streaming_state(self):
        """重置流式处理状态"""
        self.streaming_values.clear()
        self.streaming_peaks.clear()
        self.streaming_drawdowns.clear()
        logger.info("流式处理状态已重置")
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)