#!/usr/bin/env python3
"""
invest.py - A股日频投资组合管理系统
功能：组合账本 + 交易计划器 + 对账器
"""

import json
import pandas as pd
import numpy as np
import argparse
import hashlib
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import akshare for stock data fetching
try:
    import akshare as ak
except ImportError:
    ak = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== 数据结构定义 ====================

@dataclass
class Lot:
    """持仓分笔记录"""
    code: str              # 股票代码
    shares: int           # 股数
    cost: float           # 成本价
    buy_date: str         # 买入日期 YYYY-MM-DD
    sellable_date: str    # 可卖日期 YYYY-MM-DD (T+1)
    fees_accum: float     # 累计费用
    tags: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class TradeOrder:
    """交易指令"""
    trade_id: str
    code: str
    name: str
    side: str              # 'buy' or 'sell'
    qty: int
    limit_price: float
    reason: str            # 'init', 'add', 'trim', 'exit'
    constraints: List[str] # ['t1', 'limit', 'st']
    expected_fees: float
    expected_slippage: float
    notes: str = ""


@dataclass
class Fill:
    """成交记录"""
    trade_id: str
    fill_price: float
    fill_qty: int
    fill_time: str
    actual_fees: float


@dataclass
class ReconcileReport:
    """对账报告"""
    date: str
    total_orders: int
    filled_orders: int
    fill_rate: float
    total_fees: float
    slippage_bps: float
    net_cash_flow: float
    summary: Dict[str, Any]


# ==================== Portfolio 组合账本 ====================

class Portfolio:
    """投资组合账本 - 管理分笔持仓、资金、T+1约束"""
    
    def __init__(self, 
                 cash_free: float = 1000000.0,
                 cash_reserved: float = 0.0,
                 lots: List[Lot] = None,
                 params: Dict[str, Any] = None):
        self.cash_free = cash_free
        self.cash_reserved = cash_reserved
        self.lots = lots or []
        
        # 默认参数配置
        default_params = {
            'commission_rate': 0.0003,      # 佣金费率 0.03%
            'min_commission': 5.0,          # 最低佣金 5元
            'stamp_duty_rate': 0.0005,      # 印花税 0.05% (仅卖出)
            'transfer_fee_rate': 0.00002,   # 过户费 0.002%
            'slippage_bps': 5,              # 滑点 5个bp
            'max_turnover': 0.25,           # 最大换手率 25%
            'max_single_weight': 0.08,      # 单票最大权重 8%
            'min_rebalance_threshold': 0.20, # 最小调仓阈值 20%
            'lot_selection': 'fifo'         # 卖出选择: fifo/lowest_cost
        }
        
        if params:
            default_params.update(params)
        self.params = default_params
        
        self._build_indices()
    
    def _build_indices(self):
        """构建持仓索引"""
        self.positions_by_code = {}
        for lot in self.lots:
            if lot.code not in self.positions_by_code:
                self.positions_by_code[lot.code] = []
            self.positions_by_code[lot.code].append(lot)
    
    @classmethod
    def load(cls, path: str) -> "Portfolio":
        """从文件加载组合状态"""
        if not Path(path).exists():
            logger.info(f"Portfolio file not found, creating new portfolio: {path}")
            return cls()
            
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        lots = [Lot(**lot_data) for lot_data in data.get('lots', [])]
        
        portfolio = cls(
            cash_free=data.get('cash_free', 1000000.0),
            cash_reserved=data.get('cash_reserved', 0.0),
            lots=lots,
            params=data.get('params', {})
        )
        
        logger.info(f"Loaded portfolio: {len(lots)} lots, cash_free={portfolio.cash_free:.2f}")
        return portfolio
    
    def save(self, path: str) -> None:
        """保存组合状态到文件"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'cash_free': self.cash_free,
            'cash_reserved': self.cash_reserved,
            'lots': [asdict(lot) for lot in self.lots],
            'params': self.params,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved portfolio to {path}")
    
    def get_positions(self) -> pd.DataFrame:
        """获取当前持仓汇总"""
        if not self.lots:
            return pd.DataFrame(columns=['code', 'shares', 'avg_cost', 'lots_count'])
            
        positions = []
        for code, lots in self.positions_by_code.items():
            total_shares = sum(lot.shares for lot in lots)
            if total_shares > 0:
                avg_cost = sum(lot.shares * lot.cost for lot in lots) / total_shares
                positions.append({
                    'code': code,
                    'shares': total_shares,
                    'avg_cost': avg_cost,
                    'lots_count': len(lots)
                })
        
        return pd.DataFrame(positions)
    
    def get_sellable_shares(self, code: str, trade_date: str) -> int:
        """获取指定股票在交易日可卖出的股数(T+1约束)"""
        if code not in self.positions_by_code:
            return 0
            
        sellable = 0
        for lot in self.positions_by_code[code]:
            if lot.sellable_date <= trade_date:
                sellable += lot.shares
                
        return sellable
    
    def add_position(self, code: str, shares: int, cost: float, 
                    trade_date: str, fees: float = 0.0) -> None:
        """新增持仓分笔"""
        # T+1: 买入当日不可卖出，次一交易日可卖
        buy_date = datetime.strptime(trade_date, '%Y-%m-%d')
        sellable_date = (buy_date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        lot = Lot(
            code=code,
            shares=shares,
            cost=cost,
            buy_date=trade_date,
            sellable_date=sellable_date,
            fees_accum=fees
        )
        
        self.lots.append(lot)
        self._build_indices()
    
    def reduce_position(self, code: str, shares: int, trade_date: str) -> List[Lot]:
        """减少持仓，返回被减少的分笔记录"""
        if code not in self.positions_by_code:
            raise ValueError(f"No position found for {code}")
        
        lots = self.positions_by_code[code]
        
        # 按策略排序（FIFO或最低成本优先）
        if self.params['lot_selection'] == 'fifo':
            lots.sort(key=lambda x: x.buy_date)
        else:  # lowest_cost
            lots.sort(key=lambda x: x.cost)
        
        remaining_shares = shares
        reduced_lots = []
        
        for lot in lots:
            # 检查T+1约束
            if lot.sellable_date > trade_date:
                continue
                
            if remaining_shares <= 0:
                break
                
            if lot.shares <= remaining_shares:
                # 完全卖出这个分笔
                reduced_lots.append(lot)
                remaining_shares -= lot.shares
                lot.shares = 0
            else:
                # 部分卖出
                reduced_lot = Lot(
                    code=lot.code,
                    shares=remaining_shares,
                    cost=lot.cost,
                    buy_date=lot.buy_date,
                    sellable_date=lot.sellable_date,
                    fees_accum=lot.fees_accum * (remaining_shares / lot.shares),
                    tags=lot.tags.copy()
                )
                reduced_lots.append(reduced_lot)
                
                # 更新原分笔
                lot.shares -= remaining_shares
                lot.fees_accum *= ((lot.shares + remaining_shares - remaining_shares) / 
                                 (lot.shares + remaining_shares))
                remaining_shares = 0
        
        # 清理空分笔
        self.lots = [lot for lot in self.lots if lot.shares > 0]
        self._build_indices()
        
        if remaining_shares > 0:
            raise ValueError(f"Insufficient sellable shares for {code}. "
                           f"Requested: {shares}, Available: {shares - remaining_shares}")
        
        return reduced_lots
    
    @property
    def nav(self) -> float:
        """净值"""
        return self.cash_free + self.cash_reserved


# ==================== Market 市场数据 ====================

class Market:
    """市场数据管理 - EOD行情、涨跌停价格、板块信息"""
    
    @staticmethod
    def load_eod_prices(date: str, data_path: str = "data/market") -> pd.DataFrame:
        """加载EOD行情数据"""
        eod_file = Path(data_path) / f"eod_{date}.parquet"
        
        if not eod_file.exists():
            logger.warning(f"EOD file not found: {eod_file}")
            # 如果没有市场数据文件，尝试从Qlib获取
            return Market._fetch_from_qlib(date)
        
        df = pd.read_parquet(eod_file)
        logger.info(f"Loaded EOD data for {date}: {len(df)} stocks")
        return df
    
    @staticmethod
    def _fetch_from_qlib(date: str) -> pd.DataFrame:
        """从Qlib获取市场数据（fallback）"""
        try:
            import os
            import qlib
            from qlib.data import D
            
            # 初始化Qlib
            qlib_dir = os.path.expanduser('~/.qlib/qlib_data/cn_data')
            qlib.init(provider_uri=qlib_dir, region='cn')
            
            # 转换日期格式：qlib 期望 YYYYMMDD 格式
            if '-' in date:
                qlib_date = date.replace('-', '')
            else:
                qlib_date = date
                
            # 获取基础价格数据
            # 尝试不同的instruments获取方式，避免ParallelExt问题
            try:
                # 方法1：尝试获取信号中的具体股票
                signals_path = f'data/signals/{date}.parquet'
                if os.path.exists(signals_path):
                    signals_df = pd.read_parquet(signals_path)
                    if 'code' in signals_df.columns:
                        # 将股票代码转换为qlib格式
                        instruments = []
                        for code in signals_df['code'].tolist():
                            if len(code) == 6:
                                if code.startswith('6'):
                                    instruments.append(f"SH{code}")
                                else:
                                    instruments.append(f"SZ{code}")
                            else:
                                instruments.append(code)
                        logger.info(f"Using {len(instruments)} instruments from signals")
                    else:
                        instruments = D.instruments('csi500')
                else:
                    instruments = D.instruments('csi500')
            except:
                instruments = D.instruments('csi500')
                
            fields = ['$close', '$volume', '$factor']
            
            # 直接使用标准qlib API
            df = D.features(instruments, fields, start_time=qlib_date, end_time=qlib_date, 
                           disk_cache=0, freq='day')
            
            if df is None or df.empty:
                # 2025-08-13可能在日历中但数据尚未更新，尝试前一个交易日
                logger.warning(f"No qlib data for {date}, trying previous trading day...")
                
                from qlib.data.data import Cal
                cal = Cal.calendar()
                date_ts = pd.Timestamp(qlib_date)
                
                # 查找前一个交易日
                try:
                    cal_list = cal.tolist() if hasattr(cal, 'tolist') else list(cal)
                    cal_index = cal_list.index(date_ts) if date_ts in cal_list else None
                except (ValueError, AttributeError):
                    cal_index = None
                
                if cal_index is not None and cal_index > 0:
                    prev_date = cal[cal_index - 1]
                    prev_date_str = prev_date.strftime('%Y-%m-%d')
                    logger.info(f"Trying previous trading day: {prev_date_str}")
                    
                    df = D.features(instruments, fields, start_time=prev_date_str, end_time=prev_date_str, 
                                   disk_cache=0, freq='day')
                    
                    if df is not None and not df.empty:
                        logger.info(f"Successfully fetched data from {prev_date_str} for {date} trading plan")
                        # 更新数据中的日期字段为实际使用的日期
                        df = df.reset_index()
                        df.columns = ['code', 'date', 'close', 'volume', 'factor']
                        # 标准化股票代码格式：移除SH/SZ前缀
                        df['code'] = df['code'].str.replace('^(SH|SZ)', '', regex=True)
                        df['date'] = prev_date_str  # 记录实际数据日期
                        df['original_date'] = date  # 记录原始请求日期
                    else:
                        raise Exception("No data available for current or previous trading day")
                else:
                    raise Exception("Cannot find previous trading day")
            else:
                # 正常情况下的数据处理
                df = df.reset_index()
                df.columns = ['code', 'date', 'close', 'volume', 'factor']
                # 标准化股票代码格式：移除SH/SZ前缀
                df['code'] = df['code'].str.replace('^(SH|SZ)', '', regex=True)
            
            # 添加板块信息（简化版本）
            df['board'] = df['code'].apply(Market._get_board_from_code)
            df['st_flag'] = df['code'].apply(Market._is_st_stock)
            
            logger.info(f"Fetched {len(df)} stocks from Qlib for {date}")
            return df
            
        except Exception as e:
            logger.info(f"Qlib fetch failed: {str(e)}")
            logger.debug(f"Qlib error details: {type(e).__name__}: {e}")
            
            logger.info("Using AkShare fallback")
            
        # AkShare 回退方案
        try:
            import akshare as ak
            
            # 转换日期格式为 YYYYMMDD
            if '-' in date:
                akshare_date = date.replace('-', '')
            else:
                akshare_date = date
                
            logger.info(f"Using AkShare fallback for date {date}")
            
            # 不再获取全市场股票列表，直接返回空 DataFrame
            # 让后续的 mock 数据逻辑来处理信号股票
            logger.info("AkShare fallback: creating empty DataFrame for signals-based mock data")
            return pd.DataFrame(columns=['code', 'date', 'close', 'volume', 'factor', 'board', 'st_flag'])
                
        except Exception as e:
            logger.error(f"AkShare fallback also failed: {e}")
            
        # 最后的回退：空 DataFrame
        logger.error("All data sources failed")
        return pd.DataFrame(columns=['code', 'date', 'close', 'volume', 'factor', 'board', 'st_flag'])
    
    @staticmethod
    def _get_board_from_code(code: str) -> str:
        """根据股票代码判断板块"""
        if code.startswith('68'):
            return 'STAR'  # 科创板
        elif code.startswith('30'):
            return 'ChiNext'  # 创业板  
        elif code.startswith('8') or code.startswith('4'):
            return 'NEEQ'  # 北交所
        else:
            return 'Main'  # 主板
    
    @staticmethod
    def _is_st_stock(code: str) -> bool:
        """判断是否ST股票（简化版本）"""
        # 实际应该从股票名称或专门的ST列表判断
        return False
    
    
    @staticmethod
    def fetch_stock_names(stock_codes: List[str]) -> Dict[str, str]:
        """获取股票名称映射表
        
        Args:
            stock_codes: 股票代码列表，如 ['600000', '000001', '603955']
            
        Returns:
            Dict[str, str]: 股票代码到名称的映射，如 {'600000': '浦发银行', '000001': '平安银行'}
        """
        if not stock_codes:
            return {}
            
        logger.info(f"Fetching stock names for {len(stock_codes)} stocks...")
        name_mapping = {}
        
        # 方法1: 尝试使用AkShare stock_info_a_code_name
        try:
            if ak is None:
                raise ImportError("AkShare not available")
                
            logger.debug("Trying AkShare stock_info_a_code_name()...")
            stock_info = ak.stock_info_a_code_name()
            
            # 创建代码到名称的映射
            for _, row in stock_info.iterrows():
                code = str(row['code']).zfill(6)  # 确保6位代码
                name = row['name']
                if code in stock_codes:
                    name_mapping[code] = name
                    
            logger.info(f"Successfully fetched {len(name_mapping)} stock names via AkShare stock_info_a_code_name")
            
        except Exception as e:
            logger.debug(f"AkShare stock_info_a_code_name failed: {e}")
            
            # 方法2: 尝试使用AkShare stock_zh_a_spot_em (更轻量级的API)
            try:
                logger.debug("Trying AkShare stock_zh_a_spot_em()...")
                stock_spot = ak.stock_zh_a_spot_em()
                
                # 创建代码到名称的映射
                for _, row in stock_spot.iterrows():
                    code = str(row['代码']).zfill(6)  # 确保6位代码
                    name = row['名称']
                    if code in stock_codes:
                        name_mapping[code] = name
                        
                logger.info(f"Successfully fetched {len(name_mapping)} stock names via AkShare stock_zh_a_spot_em")
                
            except Exception as e2:
                logger.debug(f"AkShare stock_zh_a_spot_em also failed: {e2}")
                
                # 方法3: 使用预定义的常见股票名称映射（简化方案）
                try:
                    logger.debug("Using predefined stock name mappings...")
                    predefined_names = Market._get_predefined_stock_names()
                    
                    for code in stock_codes:
                        if code in predefined_names:
                            name_mapping[code] = predefined_names[code]
                            
                    if name_mapping:
                        logger.info(f"Successfully matched {len(name_mapping)} stock names from predefined mappings")
                    
                except Exception as e3:
                    logger.debug(f"Predefined mappings also failed: {e3}")
        
        # 对于未找到的股票，使用空字符串
        for code in stock_codes:
            if code not in name_mapping:
                name_mapping[code] = ''
                logger.debug(f"Stock name not found for code: {code}")
                
        return name_mapping
    
    @staticmethod
    def _get_predefined_stock_names() -> Dict[str, str]:
        """获取预定义的常见股票名称映射（备用方案）"""
        # 这里包含一些常见股票的映射，可以根据需要扩展
        return {
            '603955': '大千生态',
            '600353': '旭光电子', 
            '603086': '先达股份',
            '603665': '康隆达',
            '600105': '永鼎股份',
            '603680': '今创集团',
            '603040': '新坐标',
            '603657': '春光科技',
            '603586': '金麒麟',
            '600800': '渤海化学',
            '600000': '浦发银行',
            '000001': '平安银行',
            '000002': '万科A',
            '600036': '招商银行',
            '600519': '贵州茅台',
            '000858': '五粮液',
            '002415': '海康威视',
            '600276': '恒瑞医药',
            '000063': '中兴通讯',
            '002594': '比亚迪'
        }
    
    @staticmethod
    def calc_price_limits(df: pd.DataFrame) -> pd.DataFrame:
        """计算次日涨跌停价格"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # 根据板块设置涨跌幅限制
        def get_limit_ratio(row):
            if row.get('st_flag', False):
                return 0.05  # ST股 ±5%
            elif row['board'] == 'STAR' or row['board'] == 'ChiNext':
                return 0.20  # 科创板/创业板 ±20%
            elif row['board'] == 'NEEQ':
                return 0.30  # 北交所 ±30%
            else:
                return 0.10  # 主板 ±10%
        
        df['limit_ratio'] = df.apply(get_limit_ratio, axis=1)
        df['upper_limit'] = (df['close'] * (1 + df['limit_ratio'])).round(2)
        df['lower_limit'] = (df['close'] * (1 - df['limit_ratio'])).round(2)
        
        return df


# ==================== Trading Rules 交易规则 ====================

class TradingRules:
    """A股交易规则工具类"""
    
    @staticmethod
    def round_lot_size(code: str, desired_qty: int, board: str, side: str = 'buy') -> int:
        """整手约束：按板块规则调整股数"""
        if side == 'sell':
            # 卖出允许零股
            return desired_qty
        
        # 买入整手规则
        if board == 'STAR':
            # 科创板：≥200股，步进1股
            return max(200, desired_qty)
        else:
            # 主板/创业板/北交所：100股整数倍，使用四舍五入
            if desired_qty < 50:
                return 0  # 少于50股直接舍弃
            rounded = round(desired_qty / 100) * 100  # 四舍五入到最近的100股
            return max(100, rounded)
    
    @staticmethod
    def estimate_fees(side: str, qty: int, price: float, params: Dict[str, Any]) -> Dict[str, float]:
        """费用估算"""
        amount = qty * price
        
        # 佣金（双边）
        commission = max(amount * params['commission_rate'], params['min_commission'])
        
        # 印花税（仅卖出）
        stamp_duty = amount * params['stamp_duty_rate'] if side == 'sell' else 0.0
        
        # 过户费
        transfer_fee = amount * params.get('transfer_fee_rate', 0.00002)
        
        total_fees = commission + stamp_duty + transfer_fee
        
        return {
            'commission': commission,
            'stamp_duty': stamp_duty, 
            'transfer_fee': transfer_fee,
            'total_fees': total_fees
        }
    
    @staticmethod
    def generate_trade_id(date: str, code: str, side: str, qty: int) -> str:
        """生成确定性的交易ID"""
        content = f"{date}_{code}_{side}_{qty}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


# ==================== Planner 交易计划器 ====================

class Planner:
    """交易计划器 - target_weight → trades_plan"""
    
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
    
    def plan_trades(self, signals: pd.DataFrame, market: pd.DataFrame, 
                   trade_date: str, rules: Dict[str, Any]) -> pd.DataFrame:
        """生成交易计划"""
        if signals.empty or market.empty:
            logger.warning("Empty signals or market data")
            return pd.DataFrame()
        
        # 合并信号和市场数据
        market_cols = ['code', 'close']
        if 'board' in market.columns:
            market_cols.append('board')
        if 'st_flag' in market.columns:
            market_cols.append('st_flag')
        if 'upper_limit' in market.columns:
            market_cols.append('upper_limit')
        if 'lower_limit' in market.columns:
            market_cols.append('lower_limit')
            
        # DEBUG: 检查merge前的数据
        logger.debug(f"Signals codes: {signals['code'].tolist()[:5]}... (total: {len(signals)})")
        logger.debug(f"Market codes: {market['code'].tolist()[:5]}... (total: {len(market)})")
        
        merged = signals.merge(market[market_cols], on='code', how='inner')
        logger.info(f"After merge: {len(merged)} stocks matched between signals and market data")
        
        if len(merged) == 0:
            logger.warning("No stocks matched between signals and market data!")
            logger.warning(f"Signal stocks: {sorted(signals['code'].tolist())}")
            logger.warning(f"Market stocks sample: {sorted(market['code'].tolist()[:10])}")
            # 尝试 left join 来保留信号
            merged = signals.merge(market[market_cols], on='code', how='left')
            logger.info(f"Using left join: {len(merged)} stocks retained from signals")
        
        # 如果market数据中没有board字段，从signals中获取
        if 'board' not in merged.columns and 'board' in signals.columns:
            merged['board'] = signals['board']
        
        # 应用ST股票过滤
        if rules.get('forbid_st', True) and 'st_flag' in merged.columns:
            # 确保 st_flag 是布尔型
            merged['st_flag'] = merged['st_flag'].fillna(False).astype(bool)
            merged = merged[~merged['st_flag']]
        
        # 应用流动性过滤
        merged = self._apply_liquidity_filter(merged, rules)
            
        logger.info(f"Planning trades for {len(merged)} stocks on {trade_date}")
        
        # 获取股票名称
        if not merged.empty:
            stock_codes = merged['code'].tolist()
            stock_names = Market.fetch_stock_names(stock_codes)
            merged['name'] = merged['code'].map(stock_names)
        
        # 计算目标仓位
        total_nav = self.portfolio.nav
        merged['target_value'] = merged['target_weight'] * total_nav
        
        # 防止除零错误和NaN值
        merged['close'] = merged['close'].fillna(10.0)  # 用默认价格填充NaN
        merged['close'] = merged['close'].replace(0, 10.0)  # 用默认价格替换零值
        target_shares = merged['target_value'] / merged['close']
        target_shares = target_shares.fillna(0)  # 填充NaN
        target_shares = target_shares.replace([float('inf'), float('-inf')], 0)  # 处理无穷大值
        merged['target_shares'] = target_shares.astype(int)
        
        # 获取当前持仓
        current_positions = self.portfolio.get_positions()
        current_dict = dict(zip(current_positions['code'], current_positions['shares']))
        
        # 计算交易需求
        trade_orders = []
        
        for _, row in merged.iterrows():
            code = row['code']
            target_shares = row['target_shares']
            current_shares = current_dict.get(code, 0)
            
            # 计算需要调整的股数
            delta_shares = target_shares - current_shares
            
            if abs(delta_shares) < rules.get('min_trade_threshold', 100):
                continue  # 小于最小交易阈值
            
            if delta_shares > 0:
                # 买入
                adjusted_qty = TradingRules.round_lot_size(
                    code, delta_shares, row['board'], 'buy')
                
                if adjusted_qty > 0:
                    order = self._create_buy_order(row, adjusted_qty, trade_date)
                    if order is not None:
                        trade_orders.append(order)
                    
            elif delta_shares < 0:
                # 卖出
                sell_qty = abs(delta_shares)
                sellable_qty = self.portfolio.get_sellable_shares(code, trade_date)
                
                if sellable_qty >= sell_qty:
                    order = self._create_sell_order(row, sell_qty, trade_date)
                    trade_orders.append(order)
                else:
                    logger.warning(f"Insufficient sellable shares for {code}: "
                                 f"need {sell_qty}, available {sellable_qty}")
        
        # 应用现金缓冲和换手率约束
        if trade_orders:
            trade_orders = self._apply_risk_constraints(trade_orders, rules)
            logger.info(f"After risk constraints: {len(trade_orders)} trade orders")
        
        # 转换为DataFrame
        if trade_orders:
            trades_df = pd.DataFrame([asdict(order) for order in trade_orders])
            logger.info(f"Generated {len(trades_df)} trade orders")
            return trades_df
        else:
            return pd.DataFrame()
    
    def _create_buy_order(self, row: pd.Series, qty: int, trade_date: str) -> TradeOrder:
        """创建买入订单"""
        code = row['code']
        
        # 检查是否已经涨停
        close_price = row['close']
        upper_limit = row.get('upper_limit', close_price * 1.1)
        
        # 如果收盘价已经接近涨停（差距小于0.5%），不生成买单
        if close_price >= upper_limit * 0.995:
            logger.warning(f"Stock {code} close to upper limit, skipping buy order")
            return None
        
        limit_price = min(close_price * (1 + self.portfolio.params['slippage_bps'] / 10000),
                         upper_limit)
        
        fees = TradingRules.estimate_fees('buy', qty, limit_price, self.portfolio.params)
        
        constraints = []
        if limit_price >= upper_limit * 0.99:  # 接近涨停
            constraints.append('limit')
        
        return TradeOrder(
            trade_id=TradingRules.generate_trade_id(trade_date, code, 'buy', qty),
            code=code,
            name=row.get('name', ''),
            side='buy',
            qty=qty,
            limit_price=limit_price,
            reason='rebalance',
            constraints=constraints,
            expected_fees=fees['total_fees'],
            expected_slippage=abs(limit_price - close_price) * qty,
            notes=f"target_weight={row['target_weight']:.4f}"
        )
    
    def _create_sell_order(self, row: pd.Series, qty: int, trade_date: str) -> TradeOrder:
        """创建卖出订单"""
        code = row['code']
        limit_price = max(row['close'] * (1 - self.portfolio.params['slippage_bps'] / 10000),
                         row['lower_limit'])
        
        fees = TradingRules.estimate_fees('sell', qty, limit_price, self.portfolio.params)
        
        constraints = []
        if 'lower_limit' in row and pd.notna(row['lower_limit']) and limit_price <= row['lower_limit']:
            constraints.append('limit')
        
        return TradeOrder(
            trade_id=TradingRules.generate_trade_id(trade_date, code, 'sell', qty),
            code=code,
            name=row.get('name', ''),
            side='sell',
            qty=qty,
            limit_price=limit_price,
            reason='rebalance',
            constraints=constraints,
            expected_fees=fees['total_fees'],
            expected_slippage=abs(limit_price - row['close']) * qty,
            notes=f"target_weight={row['target_weight']:.4f}"
        )
    
    def _apply_risk_constraints(self, trade_orders: List[TradeOrder], rules: Dict[str, Any]) -> List[TradeOrder]:
        """应用现金缓冲和换手率约束"""
        if not trade_orders:
            return trade_orders
            
        # 计算买入和卖出总金额
        buy_amount = sum(order.qty * order.limit_price for order in trade_orders if order.side == 'buy')
        sell_amount = sum(order.qty * order.limit_price for order in trade_orders if order.side == 'sell')
        
        # 获取组合净值
        total_nav = self.portfolio.nav
        available_cash = self.portfolio.cash_free
        
        # 应用现金缓冲约束
        cash_buffer_ratio = rules.get('cash_buffer', 0.02)
        reserved_cash = total_nav * cash_buffer_ratio
        usable_cash = available_cash + sell_amount - reserved_cash
        
        logger.info(f"Buy amount: {buy_amount:.2f}, Usable cash: {usable_cash:.2f}")
        
        # 如果买入金额超过可用现金，需要缩放
        cash_scale_factor = 1.0
        if buy_amount > usable_cash and usable_cash > 0:
            cash_scale_factor = usable_cash / buy_amount
            logger.warning(f"Cash constraint: scaling buy orders by {cash_scale_factor:.3f}")
        
        # 应用换手率约束
        max_turnover = rules.get('max_turnover', 0.25)
        total_turnover = (buy_amount + sell_amount) / total_nav if total_nav > 0 else 0
        
        logger.info(f"Total turnover: {total_turnover:.3f}, Max allowed: {max_turnover:.3f}")
        
        turnover_scale_factor = 1.0
        if total_turnover > max_turnover:
            turnover_scale_factor = max_turnover / total_turnover
            logger.warning(f"Turnover constraint: scaling all orders by {turnover_scale_factor:.3f}")
        
        # 取较小的缩放因子
        final_scale_factor = min(cash_scale_factor, turnover_scale_factor)
        
        if final_scale_factor < 1.0:
            logger.info(f"Applying final scale factor: {final_scale_factor:.3f}")
            
            # 缩放订单数量
            scaled_orders = []
            for order in trade_orders:
                scaled_qty = int(order.qty * final_scale_factor)
                
                # 确保符合整手要求
                if order.side == 'buy':
                    scaled_qty = TradingRules.round_lot_size(order.code, scaled_qty, 'Main', 'buy')
                
                if scaled_qty > 0:
                    # 重新计算费用
                    fees = TradingRules.estimate_fees(order.side, scaled_qty, order.limit_price, self.portfolio.params)
                    
                    # 创建新的订单
                    scaled_order = TradeOrder(
                        trade_id=order.trade_id,
                        code=order.code,
                        name=order.name,
                        side=order.side,
                        qty=scaled_qty,
                        limit_price=order.limit_price,
                        reason=order.reason,
                        constraints=order.constraints + (['scaled'] if final_scale_factor < 1.0 else []),
                        expected_fees=fees['total_fees'],
                        expected_slippage=abs(order.limit_price - order.limit_price) * scaled_qty,  # 这里需要实际价格
                        notes=order.notes + f" (scaled={final_scale_factor:.3f})"
                    )
                    scaled_orders.append(scaled_order)
                else:
                    logger.debug(f"Order for {order.code} dropped due to scaling (qty became 0)")
            
            return scaled_orders
        
        return trade_orders
    
    def _apply_liquidity_filter(self, merged: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """应用流动性和执行性过滤"""
        if merged.empty:
            return merged
            
        original_count = len(merged)
        
        # 过滤条件
        min_volume = rules.get('min_volume', 1000)  # 最小成交量
        min_turnover = rules.get('min_turnover', 100000)  # 最小成交额
        
        # 成交量过滤
        if 'volume' in merged.columns:
            merged = merged[merged['volume'] >= min_volume]
            
        # 成交额过滤 (close * volume)
        if 'volume' in merged.columns and 'close' in merged.columns:
            merged['turnover'] = merged['close'] * merged['volume']
            merged = merged[merged['turnover'] >= min_turnover]
            
        # 价格有效性过滤（排除价格为0或异常的股票）
        if 'close' in merged.columns:
            merged = merged[merged['close'] > 0]
            merged = merged[merged['close'] < 1000]  # 排除价格异常高的股票
            
        filtered_count = len(merged)
        if filtered_count < original_count:
            logger.info(f"Liquidity filter: {original_count} → {filtered_count} stocks")
            
        return merged


# ==================== Reconciler 对账器 ====================

class Reconciler:
    """对账器 - 导入实际成交，更新账本"""
    
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
    
    def reconcile(self, fills_df: pd.DataFrame, trade_date: str) -> ReconcileReport:
        """对账处理"""
        if fills_df.empty:
            logger.warning("No fills data provided")
            return ReconcileReport(
                date=trade_date,
                total_orders=0,
                filled_orders=0,
                fill_rate=0.0,
                total_fees=0.0,
                slippage_bps=0.0,
                net_cash_flow=0.0,
                summary={}
            )
        
        logger.info(f"Reconciling {len(fills_df)} fills for {trade_date}")
        
        total_orders = len(fills_df)
        filled_orders = len(fills_df[fills_df['fill_qty'] > 0])
        fill_rate = filled_orders / total_orders if total_orders > 0 else 0.0
        
        total_fees = 0.0
        net_cash_flow = 0.0
        
        # 处理每笔成交
        for _, fill in fills_df.iterrows():
            if fill['fill_qty'] > 0:
                self._process_fill(fill, trade_date)
                
                # 累计费用和资金流
                total_fees += fill.get('actual_fees', 0.0)
                
                if fill['side'] == 'buy':
                    net_cash_flow -= fill['fill_qty'] * fill['fill_price'] + fill.get('actual_fees', 0.0)
                else:  # sell
                    net_cash_flow += fill['fill_qty'] * fill['fill_price'] - fill.get('actual_fees', 0.0)
        
        # 更新现金
        self.portfolio.cash_free += net_cash_flow
        
        return ReconcileReport(
            date=trade_date,
            total_orders=total_orders,
            filled_orders=filled_orders,
            fill_rate=fill_rate,
            total_fees=total_fees,
            slippage_bps=0.0,  # 可计算实际滑点
            net_cash_flow=net_cash_flow,
            summary={
                'buy_orders': len(fills_df[fills_df['side'] == 'buy']),
                'sell_orders': len(fills_df[fills_df['side'] == 'sell']),
                'buy_amount': fills_df[fills_df['side'] == 'buy']['fill_qty'].sum() if 'side' in fills_df.columns else 0,
                'sell_amount': fills_df[fills_df['side'] == 'sell']['fill_qty'].sum() if 'side' in fills_df.columns else 0
            }
        )
    
    def _process_fill(self, fill: pd.Series, trade_date: str) -> None:
        """处理单笔成交"""
        code = fill['code']
        side = fill['side']
        qty = fill['fill_qty']
        price = fill['fill_price']
        fees = fill.get('actual_fees', 0.0)
        
        if side == 'buy':
            # 买入：新增分笔
            self.portfolio.add_position(code, qty, price, trade_date, fees)
            logger.debug(f"Added position: {code} {qty}@{price}")
            
        elif side == 'sell':
            # 卖出：减少分笔
            try:
                reduced_lots = self.portfolio.reduce_position(code, qty, trade_date)
                logger.debug(f"Reduced position: {code} {qty}@{price}")
            except ValueError as e:
                logger.error(f"Failed to reduce position for {code}: {e}")


# ==================== CLI 命令行接口 ====================

def cmd_plan(args):
    """生成次日交易计划"""
    logger.info(f"Planning trades for {args.date}")
    
    # 加载组合
    portfolio = Portfolio.load('data/state/portfolio.json')
    
    # 加载信号
    signals_path = Path(args.signals or f'data/signals/{args.date}.parquet')
    if not signals_path.exists():
        logger.error(f"Signals file not found: {signals_path}")
        return
    
    signals = pd.read_parquet(signals_path)
    # 解析risk_flags JSON字符串
    if 'risk_flags' in signals.columns:
        signals['risk_flags'] = signals['risk_flags'].apply(json.loads)
    logger.info(f"Loaded signals: {len(signals)} stocks")
    
    # 加载市场数据
    market = Market.load_eod_prices(args.date)
    if market.empty:
        logger.error(f"No reliable market data available for {args.date}")
        logger.error("Cannot generate safe trading plan without real market prices")
        print(f"\n❌ 无法生成交易计划")
        print(f"原因: 缺少 {args.date} 的真实市场数据")
        print(f"解决方案:")
        print(f"  1. 检查 data/market/eod_{args.date}.parquet 是否存在")
        print(f"  2. 确保qlib数据源可用")
        print(f"  3. 或使用有市场数据的交易日期")
        return
    
    # 计算涨跌停价格
    market = Market.calc_price_limits(market)
    
    # 生成交易计划
    planner = Planner(portfolio)
    rules = {
        'forbid_st': args.forbid_st,
        'max_turnover': args.max_turnover,
        'cash_buffer': args.cash_buffer
    }
    
    trades = planner.plan_trades(signals, market, args.date, rules)
    
    if not trades.empty:
        # 保存交易计划
        output_path = f'data/logs/trades_plan_{args.date}.parquet'
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        trades.to_parquet(output_path, index=False)
        
        # 生成可读报告
        report_path = f'data/logs/trades_plan_{args.date}.md'
        _generate_trade_report(trades, report_path, args.date)
        
        logger.info(f"Generated {len(trades)} orders, saved to {output_path}")
        
        # 打印摘要
        print(f"\n=== 交易计划摘要 ({args.date}) ===")
        print(f"总订单数: {len(trades)}")
        print(f"买入订单: {len(trades[trades['side'] == 'buy'])}")
        print(f"卖出订单: {len(trades[trades['side'] == 'sell'])}")
        print(f"预估费用: {trades['expected_fees'].sum():.2f}")
        print(f"详细计划已保存至: {report_path}")
    else:
        logger.info("No trades generated")


def cmd_status(args):
    """查看当前组合状态"""
    portfolio = Portfolio.load('data/state/portfolio.json')
    positions = portfolio.get_positions()
    
    print("\n=== 投资组合状态 ===")
    print(f"现金余额: {portfolio.cash_free:,.2f}")
    print(f"预留资金: {portfolio.cash_reserved:,.2f}")
    print(f"净值: {portfolio.nav:,.2f}")
    print(f"持仓品种: {len(positions)}")
    print(f"分笔数量: {len(portfolio.lots)}")
    
    if not positions.empty:
        print(f"\n前10大持仓:")
        top10 = positions.nlargest(10, 'shares')
        for _, pos in top10.iterrows():
            market_value = pos['shares'] * pos['avg_cost']  # 使用成本价估算
            weight = market_value / portfolio.nav
            print(f"{pos['code']}: {pos['shares']:,}股 "
                 f"@{pos['avg_cost']:.2f} "
                 f"权重{weight:.2%}")


def cmd_reconcile(args):
    """盘后对账"""
    logger.info(f"Reconciling fills from {args.fills}")
    
    # 加载组合
    portfolio = Portfolio.load('data/state/portfolio.json')
    
    # 加载成交数据
    if not Path(args.fills).exists():
        logger.error(f"Fills file not found: {args.fills}")
        return
    
    fills = pd.read_csv(args.fills)
    logger.info(f"Loaded {len(fills)} fills")
    
    # 执行对账
    reconciler = Reconciler(portfolio)
    trade_date = datetime.now().strftime('%Y-%m-%d')  # 或从参数获取
    
    report = reconciler.reconcile(fills, trade_date)
    
    # 保存更新后的组合
    portfolio.save('data/state/portfolio.json')
    
    # 生成对账报告
    print(f"\n=== 对账报告 ({trade_date}) ===")
    print(f"总订单数: {report.total_orders}")
    print(f"成交订单数: {report.filled_orders}")
    print(f"成交率: {report.fill_rate:.1%}")
    print(f"总费用: {report.total_fees:.2f}")
    print(f"净资金流: {report.net_cash_flow:,.2f}")
    
    logger.info("Reconciliation completed")


def cmd_generate_fills(args):
    """根据交易计划生成模拟成交文件"""
    logger.info(f"Generating fills for trade plan: {args.plan}")
    
    # 加载交易计划
    if not Path(args.plan).exists():
        logger.error(f"Trade plan file not found: {args.plan}")
        return
    
    if args.plan.endswith('.parquet'):
        trades = pd.read_parquet(args.plan)
    else:
        trades = pd.read_csv(args.plan)
    
    logger.info(f"Loaded {len(trades)} trade orders")
    
    if trades.empty:
        logger.warning("No trade orders to process")
        return
    
    # 获取股票代码列表
    stock_codes = trades['code'].tolist()
    
    # 获取最新股价
    logger.info("Fetching latest stock prices from AkShare...")
    fills_data = []
    
    for _, trade in trades.iterrows():
        code = trade['code']
        
        # 获取实时股价
        try:
            current_price = _get_stock_current_price(code)
            if current_price is None:
                logger.warning(f"Failed to get price for {code}, using simulated price")
                # 使用限价附近的模拟价格（±1%波动）
                import random
                random.seed(hash(code) % 2**32)  # 基于代码的确定性随机
                fluctuation = random.uniform(-0.01, 0.01)  # ±1%波动
                current_price = trade['limit_price'] * (1 + fluctuation)
        except Exception as e:
            logger.warning(f"Error getting price for {code}: {e}, using simulated price")
            import random
            random.seed(hash(code) % 2**32)
            fluctuation = random.uniform(-0.01, 0.01)
            current_price = trade['limit_price'] * (1 + fluctuation)
        
        # 计算实际费用
        actual_fees = TradingRules.estimate_fees(
            trade['side'], 
            trade['qty'], 
            current_price, 
            {'commission_rate': 0.0003, 'min_commission': 5.0, 
             'stamp_duty_rate': 0.0005, 'transfer_fee_rate': 0.00002}
        )['total_fees']
        
        # 生成成交记录
        fill = {
            'trade_id': trade['trade_id'],
            'code': code,
            'name': trade.get('name', ''),
            'side': trade['side'],
            'qty': trade['qty'],
            'fill_qty': trade['qty'],  # 假设全部成交
            'limit_price': trade['limit_price'],
            'fill_price': current_price,
            'fill_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'expected_fees': trade['expected_fees'],
            'actual_fees': actual_fees,
            'slippage': abs(current_price - trade['limit_price']),
            'reason': trade.get('reason', 'rebalance'),
            'notes': trade.get('notes', '')
        }
        
        fills_data.append(fill)
        logger.debug(f"Generated fill for {code}: {trade['qty']}@{current_price:.2f}")
    
    # 转换为DataFrame
    fills_df = pd.DataFrame(fills_data)
    
    # 保存成交文件
    output_path = args.output or f"data/fills/fills_{datetime.now().strftime('%Y-%m-%d')}.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fills_df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"Generated fills saved to: {output_path}")
    
    # 打印摘要
    total_amount = (fills_df['fill_qty'] * fills_df['fill_price']).sum()
    total_fees = fills_df['actual_fees'].sum()
    avg_slippage = fills_df['slippage'].mean()
    
    print(f"\n=== 模拟成交摘要 ===")
    print(f"成交订单数: {len(fills_df)}")
    print(f"买入订单: {len(fills_df[fills_df['side'] == 'buy'])}")
    print(f"卖出订单: {len(fills_df[fills_df['side'] == 'sell'])}")
    print(f"总成交金额: {total_amount:,.2f}")
    print(f"总费用: {total_fees:.2f}")
    print(f"平均滑点: {avg_slippage:.4f}")
    print(f"成交文件已保存至: {output_path}")
    
    return output_path


# 全局缓存实时行情数据，避免重复请求
_PRICE_CACHE = {}
_CACHE_TIME = None

def _get_stock_current_price(code: str) -> Optional[float]:
    """获取股票当前价格（带缓存优化）"""
    if ak is None:
        logger.warning("AkShare not available, cannot fetch real-time prices")
        return None
    
    global _PRICE_CACHE, _CACHE_TIME
    
    try:
        # 检查缓存是否需要刷新（超过5分钟）
        current_time = datetime.now()
        if (_CACHE_TIME is None or 
            (current_time - _CACHE_TIME).seconds > 300 or 
            not _PRICE_CACHE):
            
            logger.info("Refreshing price cache from AkShare...")
            _PRICE_CACHE = {}
            _CACHE_TIME = current_time
            
            # 方法1: 批量获取实时行情数据
            try:
                df = ak.stock_zh_a_spot_em()
                for _, row in df.iterrows():
                    stock_code = str(row['代码']).zfill(6)
                    current_price = float(row['最新价'])
                    if current_price > 0:  # 只缓存有效价格
                        _PRICE_CACHE[stock_code] = current_price
                        
                logger.info(f"Cached {len(_PRICE_CACHE)} stock prices")
            except Exception as e:
                logger.warning(f"Failed to refresh price cache: {e}")
        
        # 从缓存获取价格
        code_6digit = code.zfill(6)
        if code_6digit in _PRICE_CACHE:
            return _PRICE_CACHE[code_6digit]
        
        # 缓存中没有，尝试单独获取
        logger.debug(f"Price for {code} not in cache, trying individual fetch...")
        
        # 方法2: 使用股票历史数据获取最新收盘价
        try:
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=3)).strftime('%Y%m%d')
            
            df = ak.stock_zh_a_hist(symbol=code, period="daily", 
                                   start_date=start_date, end_date=end_date, 
                                   adjust="")
            
            if not df.empty:
                current_price = float(df.iloc[-1]['收盘'])
                logger.debug(f"Got price for {code} via stock_zh_a_hist: {current_price}")
                # 更新缓存
                _PRICE_CACHE[code_6digit] = current_price
                return current_price
        except Exception as e:
            logger.debug(f"stock_zh_a_hist failed for {code}: {e}")
        
        logger.warning(f"Failed to get price for {code}")
        return None
        
    except Exception as e:
        logger.warning(f"Error fetching price for {code}: {e}")
        return None


def cmd_audit(args):
    """审计导出"""
    logger.info(f"Generating audit report from {args.from_date} to {args.to}")
    
    # 获取日期范围内的所有数据
    from_date = pd.to_datetime(args.from_date)
    to_date = pd.to_datetime(args.to)
    
    print(f"\n=== 审计报告 ({args.from_date} ~ {args.to}) ===")
    
    # 1. 收集填充数据（实际成交）
    fills_data = _load_fills_data(from_date, to_date)
    
    if fills_data.empty:
        print("在指定日期范围内未找到任何交易记录")
        return
    
    # 2. 收集计划数据
    plans_data = _load_plans_data(from_date, to_date)
    
    # 3. 生成审计分析
    _generate_audit_analysis(fills_data, plans_data, args.from_date, args.to)


def _load_fills_data(from_date: pd.Timestamp, to_date: pd.Timestamp) -> pd.DataFrame:
    """加载指定日期范围内的成交数据"""
    fills_dir = Path('data/fills')
    all_fills = []
    
    if not fills_dir.exists():
        return pd.DataFrame()
    
    for file_path in fills_dir.glob('*.csv'):
        try:
            df = pd.read_csv(file_path, dtype={'code': str})  # 确保code列为字符串
            if not df.empty:
                # 解析填充时间
                df['fill_time'] = pd.to_datetime(df['fill_time'])
                df['fill_date'] = df['fill_time'].dt.date
                
                # 过滤日期
                mask = (pd.to_datetime(df['fill_date']) >= from_date) & (pd.to_datetime(df['fill_date']) <= to_date)
                filtered_df = df[mask]
                
                if not filtered_df.empty:
                    all_fills.append(filtered_df)
                    logger.info(f"Loaded {len(filtered_df)} fills from {file_path.name}")
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
    
    if all_fills:
        return pd.concat(all_fills, ignore_index=True)
    else:
        return pd.DataFrame()


def _load_plans_data(from_date: pd.Timestamp, to_date: pd.Timestamp) -> pd.DataFrame:
    """加载指定日期范围内的计划数据"""
    logs_dir = Path('data/logs')
    all_plans = []
    
    if not logs_dir.exists():
        return pd.DataFrame()
    
    for file_path in logs_dir.glob('trades_plan_*.parquet'):
        try:
            df = pd.read_parquet(file_path)
            if not df.empty:
                # 从文件名解析日期
                date_str = file_path.stem.replace('trades_plan_', '')
                plan_date = pd.to_datetime(date_str).date()
                
                # 过滤日期
                if from_date.date() <= plan_date <= to_date.date():
                    df['plan_date'] = plan_date
                    all_plans.append(df)
                    logger.info(f"Loaded {len(df)} planned trades from {file_path.name}")
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
    
    if all_plans:
        return pd.concat(all_plans, ignore_index=True)
    else:
        return pd.DataFrame()


def _generate_audit_analysis(fills_data: pd.DataFrame, plans_data: pd.DataFrame, from_date: str, to_date: str):
    """生成审计分析报告"""
    print(f"\n📊 **交易概览**")
    print(f"- 成交订单数: {len(fills_data)}")
    print(f"- 计划订单数: {len(plans_data)}")
    
    # 交易方向分析
    if not fills_data.empty:
        buy_orders = len(fills_data[fills_data['side'] == 'buy'])
        sell_orders = len(fills_data[fills_data['side'] == 'sell'])
        print(f"- 买入订单: {buy_orders}")
        print(f"- 卖出订单: {sell_orders}")
        
        # 成交金额统计
        fills_data['fill_amount'] = fills_data['fill_qty'] * fills_data['fill_price']
        total_amount = fills_data['fill_amount'].sum()
        buy_amount = fills_data[fills_data['side'] == 'buy']['fill_amount'].sum()
        sell_amount = fills_data[fills_data['side'] == 'sell']['fill_amount'].sum()
        
        print(f"\n💰 **资金流动**")
        print(f"- 总成交金额: ¥{total_amount:,.2f}")
        print(f"- 买入金额: ¥{buy_amount:,.2f}")
        print(f"- 卖出金额: ¥{sell_amount:,.2f}")
        print(f"- 净流入: ¥{sell_amount - buy_amount:,.2f}")
        
        # 费用分析
        total_expected_fees = fills_data['expected_fees'].sum()
        total_actual_fees = fills_data['actual_fees'].sum()
        fee_variance = total_actual_fees - total_expected_fees
        
        print(f"\n💸 **费用分析**")
        print(f"- 预期费用: ¥{total_expected_fees:,.2f}")
        print(f"- 实际费用: ¥{total_actual_fees:,.2f}")
        print(f"- 费用偏差: ¥{fee_variance:,.2f} ({fee_variance/total_expected_fees*100:+.1f}%)")
        
        # 滑点分析
        avg_slippage = fills_data['slippage'].mean()
        max_slippage = fills_data['slippage'].max()
        total_slippage_cost = fills_data['slippage'].sum()
        
        print(f"\n📉 **滑点分析**")
        print(f"- 平均滑点: ¥{avg_slippage:.4f}")
        print(f"- 最大滑点: ¥{max_slippage:.4f}")
        print(f"- 滑点成本: ¥{total_slippage_cost:,.2f}")
        
        # 执行效率分析
        if not plans_data.empty:
            print(f"\n⚡ **执行效率**")
            plan_codes = set(plans_data['code'].unique()) if 'code' in plans_data.columns else set()
            filled_codes = set(fills_data['code'].unique())
            
            execution_rate = len(filled_codes & plan_codes) / len(plan_codes) if plan_codes else 0
            print(f"- 计划执行率: {execution_rate:.1%}")
            
            unexecuted_codes = plan_codes - filled_codes
            if unexecuted_codes:
                print(f"- 未执行代码: {', '.join(sorted(unexecuted_codes)[:5])}{'...' if len(unexecuted_codes) > 5 else ''}")
        
        # 持仓分析
        print(f"\n📈 **持仓分析**")
        position_changes = fills_data.groupby(['code', 'name']).agg({
            'fill_qty': lambda x: (x * fills_data.loc[x.index, 'side'].map({'buy': 1, 'sell': -1})).sum(),
            'fill_amount': lambda x: (x * fills_data.loc[x.index, 'side'].map({'buy': 1, 'sell': -1})).sum(),
        }).round(2)
        
        position_changes = position_changes[position_changes['fill_qty'] != 0]
        position_changes = position_changes.sort_values('fill_amount', key=abs, ascending=False)
        
        print(f"- 仓位变动股票数: {len(position_changes)}")
        if not position_changes.empty:
            print(f"\n**主要仓位变动 (Top 10):**")
            for (code, name), row in position_changes.head(10).iterrows():
                direction = "增持" if row['fill_qty'] > 0 else "减持"
                print(f"  • {code} {name}: {direction} {abs(row['fill_qty']):,}股, ¥{abs(row['fill_amount']):,.2f}")
        
        # 板块分布分析
        print(f"\n🏢 **板块分布**")
        def get_board(code):
            code = str(code).zfill(6)  # 确保6位数字
            if code.startswith('688'):
                return '科创板'
            elif code.startswith(('300', '301')):  # 创业板包括300和301开头
                return '创业板'
            elif code.startswith('002'):
                return '深市中小板'
            elif code.startswith(('000', '003')):  # 深市主板包括000和003开头
                return '深市主板'
            elif code.startswith(('600', '601', '603', '605')):
                return '沪市主板'
            elif code.startswith(('8', '4')):
                return '北交所'
            else:
                return f'其他({code[:3]})'
        
        fills_data['code'] = fills_data['code'].astype(str).str.zfill(6)  # 确保代码格式正确
        fills_data['board'] = fills_data['code'].apply(get_board)
        board_stats = fills_data.groupby('board').agg({
            'fill_qty': 'count',
            'fill_amount': 'sum'
        }).round(2)
        
        for board, stats in board_stats.iterrows():
            print(f"  • {board}: {stats['fill_qty']}单, ¥{stats['fill_amount']:,.2f}")
    
    else:
        print("暂无成交数据进行分析")


def _generate_trade_report(trades: pd.DataFrame, output_path: str, date: str):
    """生成交易计划markdown报告"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# 交易计划 - {date}\n\n")
        
        # 汇总信息
        f.write("## 汇总\n")
        f.write(f"- 总订单数: {len(trades)}\n")
        f.write(f"- 买入订单: {len(trades[trades['side'] == 'buy'])}\n")
        f.write(f"- 卖出订单: {len(trades[trades['side'] == 'sell'])}\n")
        f.write(f"- 预估费用: {trades['expected_fees'].sum():.2f}\n\n")
        
        # 详细订单
        f.write("## 详细订单\n\n")
        f.write("| 代码 | 名称 | 方向 | 数量 | 限价 | 预估费用 | 约束 | 备注 |\n")
        f.write("|------|------|------|------|------|----------|------|------|\n")
        
        for _, trade in trades.iterrows():
            constraints_str = ','.join(trade['constraints']) if trade['constraints'] else '-'
            # 获取股票名称（如果有的话）
            stock_name = trade.get('name', '') or ''
            f.write(f"| {trade['code']} | {stock_name} | {trade['side']} | {trade['qty']:,} | "
                   f"{trade['limit_price']:.2f} | {trade['expected_fees']:.2f} | "
                   f"{constraints_str} | {trade['notes']} |\n")


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description='A股投资组合管理系统')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # plan 命令
    plan_parser = subparsers.add_parser('plan', help='生成次日交易计划')
    plan_parser.add_argument('--date', required=True, help='交易日期 YYYY-MM-DD')
    plan_parser.add_argument('--signals', help='信号文件路径')
    plan_parser.add_argument('--max-turnover', type=float, default=0.25, help='最大换手率')
    plan_parser.add_argument('--cash-buffer', type=float, default=0.02, help='现金缓冲比例')
    plan_parser.add_argument('--forbid-st', type=bool, default=True, help='禁止ST股票')
    
    # status 命令
    status_parser = subparsers.add_parser('status', help='查看当前组合状态')
    
    # generate-fills 命令
    generate_fills_parser = subparsers.add_parser('generate-fills', help='根据交易计划生成模拟成交文件')
    generate_fills_parser.add_argument('--plan', required=True, help='交易计划文件路径(.parquet或.csv)')
    generate_fills_parser.add_argument('--output', help='输出成交文件路径(默认为data/fills/fills_YYYY-MM-DD.csv)')
    
    # reconcile 命令
    reconcile_parser = subparsers.add_parser('reconcile', help='盘后对账')
    reconcile_parser.add_argument('--fills', required=True, help='成交文件路径')
    
    # audit 命令
    audit_parser = subparsers.add_parser('audit', help='审计导出')
    audit_parser.add_argument('--from', dest='from_date', required=True, help='开始日期')
    audit_parser.add_argument('--to', required=True, help='结束日期')
    
    args = parser.parse_args()
    
    if args.command == 'plan':
        cmd_plan(args)
    elif args.command == 'status':
        cmd_status(args)
    elif args.command == 'generate-fills':
        cmd_generate_fills(args)
    elif args.command == 'reconcile':
        cmd_reconcile(args)
    elif args.command == 'audit':
        cmd_audit(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()