"""
数据处理工具函数
处理股票数据加载中的异常情况
"""
import pandas as pd
import logging
from typing import List, Tuple

from qlib.data import D

logger = logging.getLogger(__name__)


def load_robust_stock_data(instruments: List[str], 
                          start_time: str,
                          end_time: str,
                          freq: str = 'day',
                          fields: List[str] = None,
                          max_missing_ratio: float = 0.1) -> Tuple[pd.DataFrame, List[str]]:
    """
    稳健地加载股票数据，自动过滤数据不完整的股票
    
    Args:
        instruments: 股票代码列表
        start_time: 开始时间
        end_time: 结束时间
        freq: 数据频率
        fields: 字段列表
        max_missing_ratio: 最大缺失比例阈值
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: (有效数据, 有效股票列表)
    """
    if fields is None:
        fields = ["$open", "$high", "$low", "$close", "$volume", "$change", "$factor"]
    
    logger.info(f"开始稳健加载数据: {len(instruments)}只股票")
    
    valid_instruments = []
    failed_instruments = []
    
    # 分批次加载，每次5只股票
    batch_size = 5
    all_data = []
    
    for i in range(0, len(instruments), batch_size):
        batch_instruments = instruments[i:i+batch_size]
        
        try:
            batch_data = D.features(
                instruments=batch_instruments,
                fields=fields,
                start_time=start_time,
                end_time=end_time,
                freq=freq
            )
            
            if batch_data is not None and not batch_data.empty:
                # 检查每只股票的数据完整性
                for instrument in batch_instruments:
                    try:
                        stock_data = batch_data.xs(instrument, level=0)
                        if not stock_data.empty:
                            # 计算缺失比例
                            total_rows = len(stock_data)
                            missing_rows = stock_data.isnull().any(axis=1).sum()
                            missing_ratio = missing_rows / total_rows if total_rows > 0 else 1.0
                            
                            if missing_ratio <= max_missing_ratio:
                                valid_instruments.append(instrument)
                            else:
                                logger.warning(f"股票 {instrument} 缺失比例 {missing_ratio:.2%} 超过阈值，跳过")
                                failed_instruments.append(instrument)
                        else:
                            logger.warning(f"股票 {instrument} 数据为空，跳过")
                            failed_instruments.append(instrument)
                    except KeyError:
                        logger.warning(f"股票 {instrument} 不存在于数据中，跳过")
                        failed_instruments.append(instrument)
                
                all_data.append(batch_data)
                        
        except Exception as e:
            logger.warning(f"批次加载失败 {batch_instruments}: {e}")
            failed_instruments.extend(batch_instruments)
    
    if not valid_instruments:
        raise RuntimeError("没有找到任何有效的股票数据")
    
    # 合并所有有效数据
    if len(all_data) == 1:
        final_data = all_data[0]
    else:
        final_data = pd.concat(all_data, axis=0)
    
    # 只保留有效股票的数据
    final_data = final_data.loc[valid_instruments]
    
    logger.info(f"数据加载完成: {len(valid_instruments)}只有效股票, {len(failed_instruments)}只股票被跳过")
    logger.info(f"最终数据形状: {final_data.shape}")
    
    if failed_instruments:
        logger.warning(f"跳过的股票: {failed_instruments[:10]}{'...' if len(failed_instruments) > 10 else ''}")
    
    return final_data, valid_instruments