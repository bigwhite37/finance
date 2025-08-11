"""
A股趋势跟踪 + 相对强度策略 (风险优化版)
增强风险管理：ATR止损、最大回撤控制、波动率过滤、仓位管理
"""

import akshare as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import qlib
from qlib.data import D
import os
import argparse
warnings.filterwarnings('ignore')


class RiskSensitiveTrendStrategy:
    """风险敏感型趋势跟踪 + 相对强度策略"""

    def __init__(self, start_date='20230101', end_date=None, qlib_dir="~/.qlib/qlib_data/cn_data",
                 stock_pool_mode='auto', custom_stocks=None, index_code='000300'):
        """
        初始化策略

        Parameters:
        -----------
        start_date : str
            开始日期，格式'YYYYMMDD'
        end_date : str
            结束日期，默认为今天
        qlib_dir : str
            qlib数据目录
        stock_pool_mode : str
            股票池模式：'auto'(自动), 'index'(指数成分股), 'custom'(自定义)
        custom_stocks : list
            自定义股票列表
        index_code : str
            指数代码(当stock_pool_mode='index'时使用)
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y%m%d')
        self.qlib_dir = os.path.expanduser(qlib_dir)
        self.stock_pool_mode = stock_pool_mode
        self.custom_stocks = custom_stocks or []
        self.index_code = index_code
        self.stock_pool = []
        self.price_data = {}
        self.rs_scores = pd.DataFrame()
        self.risk_metrics = {}
        self._qlib_initialized = False

        # 风险参数
        self.max_drawdown_threshold = 0.15  # 最大回撤阈值15%
        self.volatility_threshold = 0.35    # 年化波动率阈值35%
        self.atr_multiplier = 2.0          # ATR止损倍数
        self.risk_per_trade = 0.02         # 每笔交易风险2%
        self.max_correlation = 0.7         # 最大相关性阈值

        # A股交易制度参数
        self.t_plus_1 = True               # T+1交易制度
        self.price_limit_pct = 0.10        # 涨跌停幅度（10%）
        self.st_limit_pct = 0.05           # ST股涨跌停幅度（5%）
        self.transaction_cost = 0.003      # 双边交易成本（0.3%）
        self.slippage_bps = 5              # 滑点（5个基点）

        # 初始化qlib
        self._init_qlib()

    def _init_qlib(self):
        """初始化qlib"""
        if self._qlib_initialized:
            return
        try:
            if os.path.exists(self.qlib_dir):
                qlib.init(provider_uri=self.qlib_dir, region="cn")
                print(f"Qlib初始化成功，数据路径: {self.qlib_dir}")
                self._qlib_initialized = True
            else:
                print(f"警告：Qlib数据目录不存在 {self.qlib_dir}，部分功能可能受影响")
        except Exception as e:
            print(f"Qlib初始化失败: {e}")

    def _normalize_instrument(self, code: str) -> str:
        """规范股票代码为 Qlib 标准格式"""
        c = str(code).strip().upper()
        if len(c) == 6 and c.isdigit():
            if c[0] == '6':
                return 'SH' + c
            elif c[0] in ('0', '3'):
                return 'SZ' + c
        return c

    def _convert_date_format(self, date_str: str) -> str:
        """转换日期格式从YYYYMMDD到YYYY-MM-DD"""
        if len(date_str) == 8 and date_str.isdigit():
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return date_str

    def get_stock_name(self, stock_code: str) -> str:
        """使用akshare获取股票名称"""
        try:
            stock_info = ak.stock_individual_info_em(symbol=stock_code)
            return stock_info.loc[stock_info['item'] == '股票简称', 'value'].iloc[0]
        except:
            return stock_code

    def get_all_available_stocks(self, max_stocks=200):
        """
        从qlib数据中获取所有在指定日期范围内有数据的股票

        Parameters:
        -----------
        max_stocks : int
            最大股票数量
        """
        assert self._qlib_initialized # 确保qlib已初始化

        try:
            print("正在从qlib数据中检测可用股票...")

            # 生成常见的股票代码进行测试
            potential_stocks = []

            # 先测试一些已知存在的股票代码
            common_stocks = [
                'SH600000', 'SH600036', 'SH600519', 'SH601318', 'SH601398',
                'SZ000001', 'SZ000002', 'SZ000858', 'SZ002415', 'SZ300059'
            ]
            potential_stocks.extend(common_stocks)

            # 然后生成更多候选代码
            # 上交所主要股票范围
            for i in range(600000, 602000, 50):
                potential_stocks.append(f"SH{i:06d}")

            # 深交所主板
            for i in range(1, 500, 10):
                potential_stocks.append(f"SZ{i:06d}")

            # 深交所中小板
            for i in range(2000, 2500, 25):
                potential_stocks.append(f"SZ{i:06d}")

            # 创业板
            for i in range(300000, 300500, 50):
                potential_stocks.append(f"SZ{i:06d}")

            # 去重
            potential_stocks = list(set(potential_stocks))

            print(f"测试{len(potential_stocks)}个候选股票...")
            available_stocks = []

            # 转换日期格式
            start_date_qlib = self._convert_date_format(self.start_date)
            end_date_qlib = self._convert_date_format(self.end_date)

            for i, stock_code in enumerate(potential_stocks):
                if len(available_stocks) >= max_stocks:
                    break

                if i % 50 == 0:
                    print(f"进度: {i}/{len(potential_stocks)}, 已找到: {len(available_stocks)}")

                try:
                    df = D.features(
                        instruments=[stock_code],
                        fields=['$close'],
                        start_time=start_date_qlib,
                        end_time=end_date_qlib,
                        freq='day',
                        disk_cache=0
                    )

                    if df is not None and not df.empty and len(df) >= 5:
                        # 转换为6位代码格式
                        six_digit_code = stock_code[2:] if stock_code.startswith(('SH', 'SZ')) else stock_code
                        available_stocks.append(six_digit_code)

                except Exception:
                    continue

            print(f"成功找到{len(available_stocks)}只有效股票")
            return available_stocks[:max_stocks]

        except Exception as e:
            print(f"获取可用股票失败: {e}")
            raise RuntimeError("获取可用股票失败")

    def get_stock_pool(self, index_code=None):
        """
        根据配置获取股票池（消除生存者偏差）

        Parameters:
        -----------
        index_code : str, optional
            指数代码，如果提供则覆盖默认配置
        """
        # 使用传入的index_code或默认配置
        actual_index_code = index_code or self.index_code

        if self.stock_pool_mode == 'custom':
            print(f"使用自定义股票池，共{len(self.custom_stocks)}只股票")
            self.stock_pool = self.custom_stocks

        elif self.stock_pool_mode == 'index':
            print(f"正在获取指数{actual_index_code}成分股...")
            # 警告生存者偏差风险
            print("⚠️  警告：使用当前时点成分股进行历史回测存在生存者偏差风险")
            print("⚠️  建议：使用历史时点成分股快照或固定全市场股票池")

            # 使用akshare获取指数成分股
            if actual_index_code == '000300':
                index_stocks = ak.index_stock_cons_csindex(symbol="000300")
            elif actual_index_code == '000905':
                index_stocks = ak.index_stock_cons_csindex(symbol="000905")
            else:
                index_stocks = ak.index_stock_cons_csindex(symbol=actual_index_code)

            self.stock_pool = index_stocks['成分券代码'].tolist()[:50]  # 限制前50只
            print(f"成功获取{len(self.stock_pool)}只股票")
        else:  # auto模式
            print("使用自动模式，基于qlib数据构建全市场股票池...")
            max_stocks = getattr(self, 'max_stocks', 200)
            self.stock_pool = self._get_universe_stocks_with_filters(max_stocks)

        return self.stock_pool

    def _get_universe_stocks_with_filters(self, max_stocks=200):
        """
        获取全市场股票池并应用质量过滤（减少生存者偏差）

        Parameters:
        -----------
        max_stocks : int
            最大股票数量
        """
        try:
            print("构建全市场股票池，应用流动性和基本面过滤...")

            # 使用更高效的批量方式获取股票池
            # 先用已知的活跃股票作为基础池
            base_pool = [
                # 主要指数成分股核心池
                '000001', '000002', '000858', '000651', '002415', '300059',
                '600000', '600036', '600519', '601318', '601398', '600887',
                '000063', '002027', '300142', '300498', '600050', '600900',
                '000333', '000568', '002304', '300760', '601668', '688981'
            ]

            # 添加更多活跃股票（按行业分布）
            additional_active = []
            for prefix in ['000', '002', '300', '600', '601', '603']:
                if prefix in ['000', '002', '300']:  # 深市
                    if prefix == '000':
                        for i in range(1, 100, 5):  # 主板
                            additional_active.append(f"{i:06d}")
                    elif prefix == '002':
                        for i in range(2000, 2100, 3):  # 中小板
                            additional_active.append(f"{i:06d}")
                    elif prefix == '300':
                        for i in range(300000, 300100, 3):  # 创业板
                            additional_active.append(f"{i:06d}")
                else:  # 沪市
                    if prefix == '600':
                        for i in range(600000, 600100, 3):
                            additional_active.append(f"{i:06d}")
                    elif prefix == '601':
                        for i in range(601000, 601050, 2):
                            additional_active.append(f"{i:06d}")
                    elif prefix == '603':
                        for i in range(603000, 603050, 2):
                            additional_active.append(f"{i:06d}")

            # 合并候选池
            candidate_pool = list(set(base_pool + additional_active))

            # 批量过滤：检查数据可用性和基本质量
            filtered_stocks = []
            start_date_qlib = self._convert_date_format(self.start_date)
            end_date_qlib = self._convert_date_format(self.end_date)

            # 分批处理以提高效率
            batch_size = 20
            for i in range(0, len(candidate_pool), batch_size):
                if len(filtered_stocks) >= max_stocks:
                    break

                batch = candidate_pool[i:i+batch_size]
                batch_codes = [self._normalize_instrument(code) for code in batch]

                try:
                    # 批量获取数据
                    batch_data = D.features(
                        instruments=batch_codes,
                        fields=['$close', '$volume'],
                        start_time=start_date_qlib,
                        end_time=end_date_qlib,
                        freq='day',
                        disk_cache=0
                    )

                    if batch_data is not None and not batch_data.empty:
                        # 检查每只股票的数据质量
                        for j, code in enumerate(batch):
                            qlib_code = batch_codes[j]
                            if qlib_code in batch_data.index.get_level_values(0):
                                stock_data = batch_data.xs(qlib_code, level=0)

                                # 应用基本过滤条件
                                if self._apply_stock_filters(stock_data, code):
                                    filtered_stocks.append(code)

                except Exception as e:
                    continue

                if i % 100 == 0:
                    print(f"处理进度: {i}/{len(candidate_pool)}, 已筛选: {len(filtered_stocks)}")

            print(f"从{len(candidate_pool)}个候选股票中筛选出{len(filtered_stocks)}只合格股票")
            return filtered_stocks[:max_stocks]

        except Exception as e:
            print(f"构建股票池失败: {e}")
            # 降级到原有方法
            return self.get_all_available_stocks(max_stocks)

    def _apply_stock_filters(self, stock_data, stock_code):
        """
        应用股票质量过滤条件

        Parameters:
        -----------
        stock_data : DataFrame
            股票历史数据
        stock_code : str
            股票代码
        """
        try:
            # 基本数据量要求
            if len(stock_data) < 10:  # 降低数据量要求
                return False

            # 去除停牌股票（连续多日无成交）
            if 'volume' in stock_data.columns:
                recent_volume = stock_data['volume'].iloc[-5:].sum()
                if recent_volume <= 0:  # 最近5天无成交
                    return False

            # 去除价格异常股票
            if 'close' in stock_data.columns:
                recent_prices = stock_data['close'].iloc[-10:]
                if recent_prices.std() / recent_prices.mean() > 2:  # 价格波动过大
                    return False
                if recent_prices.iloc[-1] < 1:  # 股价过低
                    return False

            # 去除ST股票（简单规则）
            if stock_code.startswith(('ST', '*ST', 'S*ST')):
                return False

            return True

        except Exception:
            return False

    def _get_price_limits(self, yesterday_close, is_st=False):
        """
        计算涨跌停价格限制

        Parameters:
        -----------
        yesterday_close : float
            昨日收盘价
        is_st : bool
            是否为ST股票
        """
        limit_pct = self.st_limit_pct if is_st else self.price_limit_pct
        upper_limit = yesterday_close * (1 + limit_pct)
        lower_limit = yesterday_close * (1 - limit_pct)
        return upper_limit, lower_limit

    def _simulate_order_execution(self, target_price, yesterday_close, volume_available, is_st=False, is_buy=True):
        """
        模拟A股订单执行（考虑涨跌停和滑点）

        Parameters:
        -----------
        target_price : float
            目标价格
        yesterday_close : float
            昨日收盘价
        volume_available : float
            可用成交量
        is_st : bool
            是否为ST股票
        is_buy : bool
            是否为买单
        """
        upper_limit, lower_limit = self._get_price_limits(yesterday_close, is_st)

        # 检查价格是否触及涨跌停
        if is_buy:
            if target_price >= upper_limit:
                # 买单触及涨停，可能无法成交
                execution_prob = min(0.3, volume_available / 1000000)  # 基于成交量估算成交概率
                if np.random.random() > execution_prob:
                    return None, "涨停无法买入"
                actual_price = upper_limit
            else:
                actual_price = target_price
        else:
            if target_price <= lower_limit:
                # 卖单触及跌停，可能无法成交
                execution_prob = min(0.3, volume_available / 1000000)
                if np.random.random() > execution_prob:
                    return None, "跌停无法卖出"
                actual_price = lower_limit
            else:
                actual_price = target_price

        # 应用滑点
        slippage = actual_price * self.slippage_bps / 10000
        if is_buy:
            final_price = actual_price + slippage
        else:
            final_price = actual_price - slippage

        # 应用交易成本
        cost = final_price * self.transaction_cost

        return {
            'executed_price': final_price,
            'transaction_cost': cost,
            'slippage': slippage,
            'price_limited': target_price != actual_price
        }, None

    def _calculate_realistic_stop_loss(self, current_price, atr, yesterday_close, is_st=False):
        """
        计算考虑A股制度约束的止损价格

        Parameters:
        -----------
        current_price : float
            当前价格
        atr : float
            ATR值
        yesterday_close : float
            昨日收盘价
        is_st : bool
            是否为ST股票
        """
        # 理论ATR止损
        theoretical_stop = current_price - (atr * self.atr_multiplier)

        # 考虑跌停限制
        upper_limit, lower_limit = self._get_price_limits(yesterday_close, is_st)

        # 如果理论止损低于跌停价，实际止损就是跌停价
        if theoretical_stop < lower_limit:
            actual_stop = lower_limit
            stop_risk_multiplier = (current_price - actual_stop) / (atr * self.atr_multiplier)
        else:
            actual_stop = theoretical_stop
            stop_risk_multiplier = 1.0

        return {
            'stop_price': actual_stop,
            'risk_multiplier': stop_risk_multiplier,  # 实际风险与理论风险的倍数
            'is_limited': theoretical_stop < lower_limit
        }

    def fetch_stock_data(self, stock_code):
        """
        使用qlib获取单只股票历史数据

        Parameters:
        -----------
        stock_code : str
            股票代码（6位格式，如'000001'）
        """
        if not self._qlib_initialized:
            print(f"Qlib未正确初始化，跳过股票{stock_code}")
            return None

        try:
            # 规范化股票代码
            qlib_code = self._normalize_instrument(stock_code)

            # 转换日期格式
            start_date_qlib = self._convert_date_format(self.start_date)
            end_date_qlib = self._convert_date_format(self.end_date)

            # 使用qlib获取历史数据
            fields = ['$open', '$high', '$low', '$close', '$volume']

            df = D.features(
                instruments=[qlib_code],
                fields=fields,
                start_time=start_date_qlib,
                end_time=end_date_qlib,
                freq='day',
                disk_cache=0
            )

            if df is not None and not df.empty:
                # 处理多级索引，提取股票数据
                df = df.xs(qlib_code, level=0)

                # 规范列名（去掉$前缀）
                df.columns = [col.replace('$', '') for col in df.columns]

                # 确保数据类型正确
                df = df.astype(float)

                stock_name = self.get_stock_name(stock_code)
                print(f"成功获取{stock_code} ({stock_name})数据，共{len(df)}条记录")
                return df
            else:
                stock_name = self.get_stock_name(stock_code)
                print(f"未获取到{stock_code} ({stock_name})的数据")
                return None

        except Exception as e:
            stock_name = self.get_stock_name(stock_code)
            print(f"获取{stock_code} ({stock_name})数据失败: {e}")
            return None

    def calculate_atr(self, df, period=14):
        """
        计算ATR（平均真实波幅）- 使用Wilder RMA平滑

        Parameters:
        -----------
        df : DataFrame
            股票价格数据
        period : int
            ATR周期
        """
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)

        # 使用Wilder RMA代替简单移动平均
        df['ATR'] = self._wilder_rma(true_range, period)

        # 计算ATR百分比（相对于价格）
        df['ATR_pct'] = df['ATR'] / df['close'] * 100

        return df

    def calculate_volatility(self, df, window=20):
        """
        计算历史波动率

        Parameters:
        -----------
        df : DataFrame
            股票价格数据
        window : int
            计算窗口
        """
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window).std() * np.sqrt(252)  # 年化

        return df

    def calculate_max_drawdown(self, df, window=60):
        """
        计算滚动最大回撤

        Parameters:
        -----------
        df : DataFrame
            股票价格数据
        window : int
            回看窗口
        """
        # 计算滚动最高点
        rolling_max = df['close'].rolling(window, min_periods=1).max()
        # 计算回撤
        df['drawdown'] = (df['close'] - rolling_max) / rolling_max
        # 计算滚动最大回撤
        df['max_drawdown'] = df['drawdown'].rolling(window, min_periods=1).min()

        return df

    def calculate_ma_signals(self, df, short_window=20, long_window=60):
        """
        计算移动平均线信号（增加趋势强度）

        Parameters:
        -----------
        df : DataFrame
            股票价格数据
        short_window : int
            短期均线周期
        long_window : int
            长期均线周期
        """
        df['MA_short'] = df['close'].rolling(window=short_window).mean()
        df['MA_long'] = df['close'].rolling(window=long_window).mean()
        df['MA_slope'] = (df['MA_short'] - df['MA_short'].shift(5)) / df['MA_short'].shift(5) * 100

        # 趋势信号：考虑均线斜率
        df['trend_signal'] = np.where(
            (df['MA_short'] > df['MA_long']) & (df['MA_slope'] > 0), 1,
            np.where((df['MA_short'] < df['MA_long']) & (df['MA_slope'] < 0), -1, 0)
        )

        # 趋势强度（0-100）
        df['trend_strength'] = np.abs(df['MA_short'] - df['MA_long']) / df['MA_long'] * 100

        return df

    def _wilder_rma(self, series, period):
        """
        计算Wilder RMA（与EMA的α=1/period等价）

        Parameters:
        -----------
        series : pd.Series
            输入序列
        period : int
            周期
        """
        alpha = 1.0 / period
        return series.ewm(alpha=alpha, adjust=False).mean()

    def calculate_rsi(self, df, period=14):
        """
        计算RSI指标 - 使用Wilder RMA平滑

        Parameters:
        -----------
        df : DataFrame
            股票价格数据
        period : int
            RSI周期
        """
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))

        # 使用Wilder RMA代替简单移动平均
        avg_gain = self._wilder_rma(gain, period)
        avg_loss = self._wilder_rma(loss, period)

        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        return df

    def calculate_bollinger_bands(self, df, window=20, num_std=2):
        """
        计算布林带

        Parameters:
        -----------
        df : DataFrame
            股票价格数据
        window : int
            计算窗口
        num_std : float
            标准差倍数
        """
        df['BB_middle'] = df['close'].rolling(window).mean()
        bb_std = df['close'].rolling(window).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * num_std)
        df['BB_lower'] = df['BB_middle'] - (bb_std * num_std)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle'] * 100
        df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

        return df

    def calculate_risk_metrics(self, df, stock_code, rolling_window=60):
        """
        计算综合风险指标（严格滚动窗口，消除前瞻偏差）

        Parameters:
        -----------
        df : DataFrame
            股票价格数据
        stock_code : str
            股票代码
        rolling_window : int
            滚动窗口长度
        """
        if len(df) < 5:  # 进一步降低最小数据要求
            return None

        # 使用滚动窗口计算风险指标，严格避免前瞻
        # 所有指标基于T-1及之前的数据

        # 获取可用的历史长度
        available_length = min(rolling_window, len(df) - 1)

        # 计算滚动风险指标（使用倒数第二天作为评估点）
        eval_point = -2 if len(df) > 1 else -1

        # 波动率：使用滚动窗口
        if 'volatility' in df.columns and not df['volatility'].iloc[:eval_point+1].empty:
            current_volatility = df['volatility'].iloc[:eval_point+1].iloc[-1]
        else:
            current_volatility = 0.25  # 默认值

        # 回撤：使用滚动窗口
        if 'drawdown' in df.columns and not df['drawdown'].iloc[:eval_point+1].empty:
            current_drawdown = abs(df['drawdown'].iloc[:eval_point+1].iloc[-1])
        else:
            current_drawdown = 0.05  # 默认值

        # 最大回撤：使用滚动窗口
        if 'max_drawdown' in df.columns and not df['max_drawdown'].iloc[:eval_point+1].empty:
            max_drawdown_60d = abs(df['max_drawdown'].iloc[:eval_point+1].iloc[-1])
        else:
            max_drawdown_60d = 0.10  # 默认值

        # ATR百分比
        if 'ATR_pct' in df.columns and not df['ATR_pct'].iloc[:eval_point+1].empty:
            atr_pct = df['ATR_pct'].iloc[:eval_point+1].iloc[-1]
        else:
            atr_pct = 2.0  # 默认值

        # 布林带宽度
        if 'BB_width' in df.columns and not df['BB_width'].iloc[:eval_point+1].empty:
            bb_width = df['BB_width'].iloc[:eval_point+1].iloc[-1]
        else:
            bb_width = 5.0  # 默认值

        # 计算滚动夏普比率（仅使用历史数据）
        if 'returns' in df.columns and len(df[:eval_point+1]) > 10:
            rolling_returns = df['returns'].iloc[:eval_point+1].dropna()
            if len(rolling_returns) > 0:
                # 使用滚动窗口计算夏普比率
                window_returns = rolling_returns.iloc[-min(available_length, len(rolling_returns)):]
                if len(window_returns) > 5 and window_returns.std() > 0:
                    sharpe_ratio = (window_returns.mean() * 252) / (window_returns.std() * np.sqrt(252))
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0

        # 计算下行偏差（滚动窗口）
        if 'returns' in df.columns and len(df[:eval_point+1]) > 10:
            rolling_returns = df['returns'].iloc[:eval_point+1].dropna()
            window_returns = rolling_returns.iloc[-min(available_length, len(rolling_returns)):]
            negative_returns = window_returns[window_returns < 0]
            downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        else:
            downside_deviation = 0.15  # 默认值

        # 调整风险评分公式（更合理的分母）
        risk_score = (
            (current_volatility / 0.8 * 25) +   # 波动率阈值调整为0.8
            (current_drawdown / 0.15 * 25) +    # 当前回撤阈值调整为0.15
            (max_drawdown_60d / 0.25 * 25) +    # 最大回撤阈值调整为0.25
            (atr_pct / 8 * 25)                  # ATR阈值调整为8%
        )
        risk_score = min(100, max(0, risk_score))

        self.risk_metrics[stock_code] = {
            'volatility': current_volatility,
            'current_drawdown': current_drawdown,
            'max_drawdown_60d': max_drawdown_60d,
            'atr_pct': atr_pct,
            'bb_width': bb_width,
            'sharpe_ratio': sharpe_ratio,
            'downside_deviation': downside_deviation,
            'risk_score': risk_score
        }

        return risk_score

    def calculate_position_size(self, stock_code, capital=100000):
        """
        基于风险计算仓位大小（凯利公式简化版）

        Parameters:
        -----------
        stock_code : str
            股票代码
        capital : float
            可用资金
        """
        if stock_code not in self.risk_metrics:
            return 0

        metrics = self.risk_metrics[stock_code]
        df = self.price_data[stock_code]

        # 基于ATR的仓位计算
        atr = df['ATR'].iloc[-1]
        price = df['close'].iloc[-1]

        # 每笔交易的风险金额
        risk_amount = capital * self.risk_per_trade

        # 止损距离（ATR的倍数）
        stop_distance = atr * self.atr_multiplier

        # 计算仓位
        shares = risk_amount / stop_distance
        position_value = shares * price

        # 根据风险评分调整仓位
        risk_adjustment = 1 - (metrics['risk_score'] / 200)  # 风险越高，仓位越小
        position_value *= risk_adjustment

        # 限制单一仓位不超过总资金的20%
        max_position = capital * 0.2
        position_value = min(position_value, max_position)

        return position_value

    def _calculate_transaction_costs(self, trade_value, is_buy=True):
        """
        计算A股交易成本

        Parameters:
        -----------
        trade_value : float
            交易金额
        is_buy : bool
            是否为买入交易
        """
        # A股交易成本构成：
        # 1. 印花税：卖出时收取0.1%，买入免收
        # 2. 券商佣金：双边收取，一般0.025%，最低5元
        # 3. 过户费：双边收取0.002%（仅上海）

        stamp_duty = 0
        if not is_buy:  # 只有卖出时收印花税
            stamp_duty = trade_value * 0.001

        # 券商佣金
        commission = max(trade_value * 0.00025, 5)  # 最低5元

        # 过户费（简化：统一按0.002%计算）
        transfer_fee = trade_value * 0.00002

        total_cost = stamp_duty + commission + transfer_fee

        return {
            'total_cost': total_cost,
            'stamp_duty': stamp_duty,
            'commission': commission,
            'transfer_fee': transfer_fee,
            'cost_rate': total_cost / trade_value if trade_value > 0 else 0
        }

    def calculate_relative_strength(self, momentum_windows=[63, 126, 252], skip_recent=21):
        """
        计算风险调整后的相对强度（多窗口动量，消除前瞻偏差）

        Parameters:
        -----------
        momentum_windows : list
            动量计算窗口列表（约3/6/12个月）
        skip_recent : int
            跳过的近期天数（避免短期反转）
        """
        rs_data = {}

        for stock in self.stock_pool:
            if stock in self.price_data and self.price_data[stock] is not None:
                df = self.price_data[stock]

                # 确保有足够的历史数据（降低要求）
                min_required = min(30, max(momentum_windows) + skip_recent + 5)  # 最多要求30天
                available_data = len(df)
                if available_data < 15 or stock not in self.risk_metrics:  # 最少15天
                    continue

                # 根据可用数据调整窗口
                available_windows = []
                available_skip = min(skip_recent, available_data // 3)  # 动态调整跳过天数

                for window in momentum_windows:
                    if available_data > window + available_skip + 5:
                        available_windows.append(window)

                if not available_windows:
                    # 如果没有窗口可用，使用最短窗口
                    available_windows = [min(available_data - available_skip - 2, 20)]

                try:
                    # 多窗口动量计算（跳过近期）
                    momentum_scores = []
                    eval_end = len(df) - available_skip  # 动态跳过天数

                    for window in available_windows:
                        if eval_end - window > 0:
                            # 严格使用历史数据
                            end_price = df['close'].iloc[eval_end - 1]  # T-skip_recent时点价格
                            start_price = df['close'].iloc[eval_end - window - 1]  # T-skip_recent-window时点价格

                            if start_price > 0:
                                momentum = (end_price / start_price - 1) * 100
                                momentum_scores.append(momentum)

                    if not momentum_scores:
                        continue

                    # 加权平均多个窗口的动量（长期权重更高）
                    weights = [0.2, 0.3, 0.5] if len(momentum_scores) == 3 else [1.0/len(momentum_scores)] * len(momentum_scores)
                    weighted_momentum = sum(score * weight for score, weight in zip(momentum_scores, weights[:len(momentum_scores)]))

                    # 风险调整：用夏普比率和风险评分调整
                    metrics = self.risk_metrics[stock]
                    risk_adjustment = max(0.3, (100 - metrics['risk_score']) / 100)  # 防止过度惩罚
                    sharpe_adjustment = max(0.5, min(1.5, metrics['sharpe_ratio'] + 1))

                    # 计算趋势确认（使用移动平均确认）
                    trend_confirmation = 1.0
                    if 'MA_short' in df.columns and 'MA_long' in df.columns:
                        # 使用历史时点的移动平均
                        eval_point = eval_end - 1
                        if (eval_point < len(df) and
                            not pd.isna(df['MA_short'].iloc[eval_point]) and
                            not pd.isna(df['MA_long'].iloc[eval_point])):
                            if df['MA_short'].iloc[eval_point] > df['MA_long'].iloc[eval_point]:
                                trend_confirmation = 1.2  # 趋势向上，加分
                            else:
                                trend_confirmation = 0.8  # 趋势向下，减分

                    # 最终相对强度评分
                    adjusted_rs = weighted_momentum * risk_adjustment * sharpe_adjustment * trend_confirmation

                    rs_data[stock] = {
                        'rs_score': adjusted_rs,
                        'raw_return': weighted_momentum,
                        'risk_score': metrics['risk_score'],
                        'volatility': metrics['volatility'],
                        'sharpe_ratio': metrics['sharpe_ratio'],
                        'trend_confirmation': trend_confirmation,
                        'momentum_3m': momentum_scores[0] if len(momentum_scores) > 0 else 0,
                        'momentum_6m': momentum_scores[1] if len(momentum_scores) > 1 else 0,
                        'momentum_12m': momentum_scores[2] if len(momentum_scores) > 2 else 0,
                    }

                except Exception as e:
                    # 静默跳过计算失败的股票
                    continue

        # 转换为DataFrame并排序
        self.rs_scores = pd.DataFrame.from_dict(rs_data, orient='index')
        self.rs_scores.index.name = 'stock_code'
        if not self.rs_scores.empty and 'rs_score' in self.rs_scores.columns:
            self.rs_scores = self.rs_scores.sort_values('rs_score', ascending=False)
        self.rs_scores.reset_index(inplace=True)

        return self.rs_scores

    def _filter_by_correlation(self, candidate_stocks, max_correlation=None):
        """
        基于相关性过滤股票，避免选中高度相关的股票

        Parameters:
        -----------
        candidate_stocks : list
            候选股票列表
        max_correlation : float
            最大相关性阈值，默认使用类属性
        """
        max_corr = max_correlation or self.max_correlation

        if len(candidate_stocks) <= 1:
            return candidate_stocks

        print(f"正在进行相关性过滤，阈值: {max_corr}")

        try:
            # 构建价格收益率矩阵
            returns_data = {}
            min_length = float('inf')

            for stock in candidate_stocks:
                if stock in self.price_data:
                    df = self.price_data[stock]
                    if 'returns' in df.columns:
                        returns = df['returns'].dropna()
                        if len(returns) > 20:  # 至少需要20个观测值
                            returns_data[stock] = returns
                            min_length = min(min_length, len(returns))

            if len(returns_data) <= 1:
                return candidate_stocks

            # 对齐时间序列长度
            aligned_returns = {}
            for stock, returns in returns_data.items():
                aligned_returns[stock] = returns.iloc[-min_length:]

            # 计算相关性矩阵
            returns_df = pd.DataFrame(aligned_returns)
            correlation_matrix = returns_df.corr()

            # 贪心算法进行相关性过滤
            selected = []
            remaining = candidate_stocks.copy()

            # 按照相对强度评分排序（优先选择评分高的）
            if hasattr(self, 'rs_scores') and not self.rs_scores.empty:
                rs_dict = dict(zip(self.rs_scores['stock_code'], self.rs_scores['rs_score']))
                remaining.sort(key=lambda x: rs_dict.get(x, 0), reverse=True)

            for candidate in remaining:
                if candidate not in returns_data:
                    continue

                # 检查与已选股票的相关性
                can_add = True
                for selected_stock in selected:
                    if selected_stock in correlation_matrix.index and candidate in correlation_matrix.index:
                        corr = abs(correlation_matrix.loc[candidate, selected_stock])
                        if corr > max_corr:
                            can_add = False
                            break

                if can_add:
                    selected.append(candidate)

            print(f"相关性过滤完成: {len(candidate_stocks)} -> {len(selected)}")

            # 如果过滤后股票太少，适当放宽标准
            if len(selected) < 3 and max_corr > 0.5:
                print(f"股票数量过少，放宽相关性阈值到 {max_corr + 0.1}")
                return self._filter_by_correlation(candidate_stocks, max_corr + 0.1)

            return selected

        except Exception as e:
            print(f"相关性过滤失败: {e}")
            return candidate_stocks  # 失败时返回原候选股票

    def check_market_regime(self):
        """
        检查市场整体状态（风险开关）- 多因子判断
        """
        if not self._qlib_initialized:
            print("Qlib未正确初始化，返回中性市场状态")
            return 'NEUTRAL'

        try:
            # 使用qlib获取上证指数数据（SH000001）
            end_date = self._convert_date_format(self.end_date)
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            start_dt = end_dt - timedelta(days=300)  # 获取更长历史用于计算趋势
            start_date = start_dt.strftime('%Y-%m-%d')

            market_df = D.features(
                instruments=['SH000001'],
                fields=['$close', '$volume'],
                start_time=start_date,
                end_time=end_date,
                freq='day',
                disk_cache=0
            )

            if market_df is None or market_df.empty:
                print("无法获取上证指数数据，返回中性市场状态")
                return 'NEUTRAL'

            # 提取数据并规范列名
            market_df = market_df.xs('SH000001', level=0)
            market_df.columns = [col.replace('$', '') for col in market_df.columns]

            if len(market_df) < 60:
                return 'NEUTRAL'

            # 多因子市场状态判断
            recent_60d = market_df.iloc[-60:]
            recent_20d = market_df.iloc[-20:]

            # 1. 波动率因子
            returns = recent_60d['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)

            # 2. 趋势因子（TSMOM）
            price_now = market_df['close'].iloc[-1]
            price_3m = market_df['close'].iloc[-63] if len(market_df) > 63 else price_now
            momentum_3m = (price_now / price_3m - 1) * 100 if price_3m > 0 else 0

            # 3. 回撤因子
            recent_high = recent_60d['close'].max()
            current_drawdown = (price_now - recent_high) / recent_high

            # 4. 成交量因子
            vol_recent = recent_20d['volume'].mean()
            vol_baseline = recent_60d['volume'].mean()
            volume_ratio = vol_recent / vol_baseline if vol_baseline > 0 else 1

            # 5. 移动平均趋势
            ma_20 = recent_60d['close'].rolling(20).mean().iloc[-1]
            ma_60 = recent_60d['close'].rolling(60).mean().iloc[-1]
            ma_trend = 1 if price_now > ma_20 > ma_60 else -1 if price_now < ma_20 < ma_60 else 0

            print(f"市场指标 - 波动率: {volatility:.3f}, 动量3m: {momentum_3m:.2f}%, 回撤: {current_drawdown:.3f}, 成交量比: {volume_ratio:.2f}, 趋势: {ma_trend}")

            # 综合评分系统
            risk_score = 0

            # 波动率评分
            if volatility > 0.35:
                risk_score += 3
            elif volatility > 0.25:
                risk_score += 1
            elif volatility < 0.15:
                risk_score -= 1

            # 趋势动量评分
            if momentum_3m > 10:
                risk_score -= 2
            elif momentum_3m > 0:
                risk_score -= 1
            elif momentum_3m < -15:
                risk_score += 3
            elif momentum_3m < -5:
                risk_score += 1

            # 回撤评分
            if current_drawdown < -0.15:
                risk_score += 3
            elif current_drawdown < -0.08:
                risk_score += 1
            elif current_drawdown > -0.02:
                risk_score -= 1

            # 成交量评分（放量下跌是危险信号）
            if volume_ratio > 1.3 and momentum_3m < -5:
                risk_score += 2
            elif volume_ratio < 0.7:
                risk_score += 1

            # 移动平均趋势评分
            risk_score -= ma_trend

            print(f"市场风险综合评分: {risk_score}")

            # 状态判断
            if risk_score >= 4:
                return 'RISK_OFF'   # 高风险
            elif risk_score <= -2:
                return 'RISK_ON'    # 低风险
            else:
                return 'NEUTRAL'    # 中性

        except Exception as e:
            print(f"获取市场数据失败: {e}，返回中性市场状态")
            return 'NEUTRAL'

    def run_strategy(self):
        """运行完整策略（风险优化版）"""
        print("开始运行风险敏感型策略...")

        # 1. 检查市场状态
        market_regime = self.check_market_regime()
        print(f"当前市场状态: {market_regime}")

        # 2. 获取股票池
        if not self.stock_pool:
            self.get_stock_pool()

        # 3. 获取所有股票数据并计算指标
        print("正在获取股票历史数据并计算风险指标...")
        for i, stock in enumerate(self.stock_pool):
            stock_name = self.get_stock_name(stock)
            print(f"进度: {i+1}/{len(self.stock_pool)} - {stock} ({stock_name})")
            df = self.fetch_stock_data(stock)
            if df is not None and len(df) > 5:  # 进一步降低数据量要求
                # 计算技术指标
                df = self.calculate_ma_signals(df)
                df = self.calculate_rsi(df)
                df = self.calculate_atr(df)
                df = self.calculate_volatility(df)
                df = self.calculate_max_drawdown(df)
                df = self.calculate_bollinger_bands(df)

                # 计算风险指标（使用调整后的阈值）
                risk_score = self.calculate_risk_metrics(df, stock)

                # 调整风险过滤阈值（更宽松）
                if risk_score is not None and risk_score < 85:  # 放宽风险评分阈值
                    self.price_data[stock] = df

        print(f"成功获取{len(self.price_data)}只股票数据（已过滤高风险）")

        # 4. 计算风险调整后的相对强度
        self.calculate_relative_strength()

        # 5. 选择股票（多重风险过滤）
        candidate_stocks = []

        # 首先通过技术指标过滤
        for _, row in self.rs_scores.head(20).iterrows():
            stock = row['stock_code']
            if stock in self.price_data:
                df = self.price_data[stock]
                metrics = self.risk_metrics[stock]

                # 多重过滤条件（放宽布林带限制）
                conditions = [
                    df['trend_signal'].iloc[-1] == 1,  # 趋势向上
                    df['RSI'].iloc[-1] < 75,           # RSI未严重超买（放宽到75）
                    df['RSI'].iloc[-1] > 25,           # RSI未严重超卖（放宽到25）
                    metrics['volatility'] < self.volatility_threshold * 1.2,  # 波动率限制放宽20%
                    metrics['max_drawdown_60d'] < self.max_drawdown_threshold * 1.3,  # 回撤限制放宽30%
                    # 移除布林带上轨限制 - 突破上轨是趋势加速信号
                    df['trend_strength'].iloc[-1] > 0.5,  # 趋势强度要求降低
                ]

                if all(conditions):
                    candidate_stocks.append(stock)

        print(f"技术指标过滤后候选股票数量: {len(candidate_stocks)}")

        # 6. 应用相关性过滤
        if len(candidate_stocks) > 1:
            filtered_stocks = self._filter_by_correlation(candidate_stocks)
        else:
            filtered_stocks = candidate_stocks

        # 7. 最终选择和仓位计算
        selected_stocks = filtered_stocks[:5]  # 最多选5只
        position_sizes = {}

        for stock in selected_stocks:
            position_sizes[stock] = self.calculate_position_size(stock)

        # 根据市场状态调整仓位
        if market_regime == 'RISK_OFF':
            print("市场风险较高，降低整体仓位50%")
            position_sizes = {k: v * 0.5 for k, v in position_sizes.items()}
        elif market_regime == 'RISK_ON':
            print("市场风险较低，维持正常仓位")

        return selected_stocks, position_sizes

    def generate_stop_loss_levels(self, selected_stocks):
        """
        生成止损位（考虑A股制度约束）

        Parameters:
        -----------
        selected_stocks : list
            选中的股票列表
        """
        stop_loss_levels = {}

        for stock in selected_stocks:
            if stock in self.price_data:
                df = self.price_data[stock]
                current_price = df['close'].iloc[-1]
                atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else current_price * 0.02
                yesterday_close = df['close'].iloc[-2] if len(df) > 1 else current_price

                # 判断是否为ST股票
                is_st = stock.startswith(('ST', '*ST', 'S*ST'))

                # ATR止损（考虑A股制度）
                atr_stop_info = self._calculate_realistic_stop_loss(
                    current_price, atr, yesterday_close, is_st
                )
                atr_stop = atr_stop_info['stop_price']

                # 支撑位止损（20日低点）
                support_stop = df['low'].iloc[-20:].min() if len(df) >= 20 else current_price * 0.95

                # 移动止损（从最高点回撤8%）
                trailing_stop = df['close'].iloc[-20:].max() * 0.92 if len(df) >= 20 else current_price * 0.92

                # 涨跌停限制
                upper_limit, lower_limit = self._get_price_limits(yesterday_close, is_st)

                # 取最合理的止损位（不一定是最高的）
                # 优先级：支撑位 > ATR止损 > 移动止损，但不能低于跌停价
                candidate_stops = [support_stop, atr_stop, trailing_stop]
                valid_stops = [stop for stop in candidate_stops if stop >= lower_limit]

                if valid_stops:
                    # 选择有效止损中最接近当前价格的（更积极的止损）
                    stop_loss = max(valid_stops)
                else:
                    # 如果所有止损都低于跌停价，使用跌停价
                    stop_loss = lower_limit

                stop_loss_levels[stock] = {
                    'current_price': current_price,
                    'stop_loss': stop_loss,
                    'stop_loss_pct': (stop_loss - current_price) / current_price * 100,
                    'atr_stop': atr_stop,
                    'support_stop': support_stop,
                    'trailing_stop': trailing_stop,
                    'upper_limit': upper_limit,
                    'lower_limit': lower_limit,
                    'is_st': is_st,
                    'risk_multiplier': atr_stop_info['risk_multiplier'],
                    'stop_limited': atr_stop_info['is_limited']
                }

        return stop_loss_levels

    def plot_risk_dashboard(self, selected_stocks, position_sizes):
        """
        绘制风险管理仪表板

        Parameters:
        -----------
        selected_stocks : list
            选中的股票列表
        position_sizes : dict
            仓位大小
        """
        # 创建子图
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                '风险调整后相对强度TOP10',
                '风险评分分布',
                '选中股票走势',
                '仓位分配',
                '波动率vs收益率',
                '止损位设置',
                '市场风险指标',
                '回撤分析'
            ],
            specs=[
                [{'type': 'bar'}, {'type': 'bar'}],
                [{'secondary_y': True}, {'type': 'pie'}],
                [{'type': 'scatter'}, {'type': 'bar'}],
                [{'type': 'indicator'}, {'type': 'box'}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.12
        )

        # 1. 风险调整后相对强度
        top_rs = self.rs_scores.head(10)
        colors = ['green' if stock in selected_stocks else 'lightgray'
                 for stock in top_rs['stock_code']]

        # 添加股票名称
        stock_names = [f"{stock}<br>{self.get_stock_name(stock)}"
                      for stock in top_rs['stock_code']]

        fig.add_trace(
            go.Bar(
                x=stock_names,
                y=top_rs['rs_score'],
                name='风险调整RS',
                marker_color=colors,
                text=top_rs['rs_score'].round(2),
                textposition='auto'
            ),
            row=1, col=1
        )

        # 2. 风险评分分布
        risk_stocks = list(self.risk_metrics.keys())[:10]
        risk_scores = [self.risk_metrics[s]['risk_score'] for s in risk_stocks]

        # 添加股票名称
        risk_stock_names = [f"{stock}<br>{self.get_stock_name(stock)}"
                           for stock in risk_stocks]

        fig.add_trace(
            go.Bar(
                x=risk_stock_names,
                y=risk_scores,
                name='风险评分',
                marker_color=['red' if s > 70 else 'yellow' if s > 50 else 'green'
                            for s in risk_scores],
                text=[f"{s:.1f}" for s in risk_scores],
                textposition='auto'
            ),
            row=1, col=2
        )

        # 3. 选中股票走势（只显示第一只）
        if selected_stocks:
            stock = selected_stocks[0]
            stock_name = self.get_stock_name(stock)
            df = self.price_data[stock]

            # K线图
            fig.add_trace(
                go.Candlestick(
                    x=df.index[-60:],
                    open=df['open'].iloc[-60:],
                    high=df['high'].iloc[-60:],
                    low=df['low'].iloc[-60:],
                    close=df['close'].iloc[-60:],
                    name=f'{stock} ({stock_name})',
                    showlegend=False
                ),
                row=2, col=1, secondary_y=False
            )

            # 布林带
            fig.add_trace(
                go.Scatter(
                    x=df.index[-60:],
                    y=df['BB_upper'].iloc[-60:],
                    name='布林上轨',
                    line=dict(color='rgba(250,128,114,0.3)'),
                    showlegend=False
                ),
                row=2, col=1, secondary_y=False
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index[-60:],
                    y=df['BB_lower'].iloc[-60:],
                    name='布林下轨',
                    line=dict(color='rgba(250,128,114,0.3)'),
                    fill='tonexty',
                    showlegend=False
                ),
                row=2, col=1, secondary_y=False
            )

            # ATR
            fig.add_trace(
                go.Scatter(
                    x=df.index[-60:],
                    y=df['ATR_pct'].iloc[-60:],
                    name='ATR%',
                    line=dict(color='purple', width=1),
                    showlegend=False
                ),
                row=2, col=1, secondary_y=True
            )

        # 4. 仓位分配饼图
        if position_sizes:
            # 添加股票名称
            position_labels = [f"{stock}<br>{self.get_stock_name(stock)}"
                             for stock in position_sizes.keys()]

            fig.add_trace(
                go.Pie(
                    labels=position_labels,
                    values=list(position_sizes.values()),
                    hole=0.3,
                    textinfo='label+percent',
                    showlegend=False
                ),
                row=2, col=2
            )

        # 5. 波动率vs收益率散点图
        scatter_data = self.rs_scores.head(15)
        # 添加股票名称
        scatter_text = [f"{stock}<br>{self.get_stock_name(stock)}"
                       for stock in scatter_data['stock_code']]

        fig.add_trace(
            go.Scatter(
                x=scatter_data['volatility'],
                y=scatter_data['raw_return'],
                mode='markers+text',
                text=scatter_text,
                textposition='top center',
                marker=dict(
                    size=scatter_data['sharpe_ratio'] * 10 + 5,
                    color=scatter_data['risk_score'],
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title="风险评分", x=0.45)
                ),
                showlegend=False
            ),
            row=3, col=1
        )

        # 6. 止损位设置
        stop_losses = self.generate_stop_loss_levels(selected_stocks[:5])
        if stop_losses:
            stocks = list(stop_losses.keys())
            stop_pcts = [stop_losses[s]['stop_loss_pct'] for s in stocks]

            # 添加股票名称
            stop_labels = [f"{stock}<br>{self.get_stock_name(stock)}"
                          for stock in stocks]

            fig.add_trace(
                go.Bar(
                    x=stop_labels,
                    y=stop_pcts,
                    name='止损距离%',
                    marker_color='orange',
                    text=[f"{p:.1f}%" for p in stop_pcts],
                    textposition='auto'
                ),
                row=3, col=2
            )

        # 7. 市场风险指标（仪表盘）
        market_risk_score = 50  # 示例值
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=market_risk_score,
                title={'text': "市场风险指数"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ),
            row=4, col=1
        )

        # 8. 回撤分析箱线图
        drawdowns = []
        labels = []
        for stock in selected_stocks[:5]:
            if stock in self.price_data:
                stock_name = self.get_stock_name(stock)
                stock_label = f"{stock}<br>{stock_name}"
                dd = self.price_data[stock]['drawdown'].iloc[-60:].values * 100
                drawdowns.extend(dd)
                labels.extend([stock_label] * len(dd))

        if drawdowns:
            fig.add_trace(
                go.Box(
                    y=drawdowns,
                    x=labels,
                    name='回撤分布',
                    showlegend=False
                ),
                row=4, col=2
            )

        # 更新布局
        fig.update_layout(
            title='风险管理仪表板',
            height=1200,
            showlegend=False
        )

        # 更新坐标轴
        fig.update_xaxes(title_text="股票代码", row=1, col=1)
        fig.update_xaxes(title_text="股票代码", row=1, col=2)
        fig.update_xaxes(title_text="波动率", row=3, col=1)
        fig.update_yaxes(title_text="风险调整RS", row=1, col=1)
        fig.update_yaxes(title_text="风险评分", row=1, col=2)
        fig.update_yaxes(title_text="收益率%", row=3, col=1)
        fig.update_yaxes(title_text="价格", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="ATR%", row=2, col=1, secondary_y=True)

        return fig

    def backtest_with_risk_management(self, selected_stocks, position_sizes, initial_capital=100000):
        """
        带风险管理的回测

        Parameters:
        -----------
        selected_stocks : list
            选中的股票列表
        position_sizes : dict
            仓位配置
        initial_capital : float
            初始资金
        """
        if not selected_stocks:
            print("没有选中的股票，无法进行回测")
            return None

        # 先找到所有股票的共同时间范围
        all_dates = []
        for stock in selected_stocks:
            if stock in self.price_data:
                all_dates.append(self.price_data[stock].index)

        if not all_dates:
            return None

        # 找到交集的时间范围
        common_dates = all_dates[0]
        for dates in all_dates[1:]:
            common_dates = common_dates.intersection(dates)

        if len(common_dates) == 0:
            print("股票间没有共同的交易日期")
            return None

        common_dates = common_dates.sort_values()

        results = []
        portfolio_value = np.ones(len(common_dates))

        for stock in selected_stocks:
            df = self.price_data[stock].reindex(common_dates).copy()

            # 填充可能的缺失值
            df = df.fillna(method='ffill')

            # 生成交易信号（考虑风险）
            df['position'] = 0

            for i in range(1, len(df)):
                # 入场条件
                if (df['trend_signal'].iloc[i] == 1 and
                    df['RSI'].iloc[i] < 70 and
                    df['RSI'].iloc[i] > 30 and
                    df['volatility'].iloc[i] < self.volatility_threshold and
                    abs(df['drawdown'].iloc[i]) < 0.1):
                    df.loc[df.index[i], 'position'] = 1

                # 出场条件（风险控制）
                elif (df['trend_signal'].iloc[i] == -1 or
                      df['RSI'].iloc[i] > 80 or
                      df['RSI'].iloc[i] < 20 or
                      abs(df['drawdown'].iloc[i]) > 0.15 or
                      df['volatility'].iloc[i] > self.volatility_threshold * 1.5):
                    df.loc[df.index[i], 'position'] = 0
                else:
                    df.loc[df.index[i], 'position'] = df['position'].iloc[i-1]

            # 计算收益
            df['returns'] = df['close'].pct_change()
            df['strategy_returns'] = df['position'].shift(1) * df['returns']

            # 应用止损
            df['cum_strategy_returns'] = 1.0
            max_price = df['close'].iloc[0]

            for i in range(1, len(df)):
                # 更新最高价
                if df['position'].iloc[i] == 1:
                    max_price = max(max_price, df['close'].iloc[i])

                    # 移动止损检查（从最高点回撤10%）
                    if df['close'].iloc[i] < max_price * 0.9:
                        df.loc[df.index[i], 'position'] = 0
                        df.loc[df.index[i], 'strategy_returns'] = -0.1
                        max_price = df['close'].iloc[i]

                # 累计收益
                df.loc[df.index[i], 'cum_strategy_returns'] = (
                    df['cum_strategy_returns'].iloc[i-1] *
                    (1 + df['strategy_returns'].iloc[i])
                )

            # 计算风险指标
            strategy_returns = df['strategy_returns'].dropna()

            # 夏普比率
            sharpe = (strategy_returns.mean() * 252) / (strategy_returns.std() * np.sqrt(252)) if strategy_returns.std() > 0 else 0

            # 最大回撤
            cum_returns = df['cum_strategy_returns']
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            max_dd = drawdown.min()

            # 胜率
            winning_trades = strategy_returns[strategy_returns > 0]
            losing_trades = strategy_returns[strategy_returns < 0]
            win_rate = len(winning_trades) / len(strategy_returns) if len(strategy_returns) > 0 else 0

            # 盈亏比
            avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0

            # 计算仓位调整后的收益
            position_weight = position_sizes.get(stock, initial_capital * 0.2) / initial_capital
            adjusted_return = (df['cum_strategy_returns'].iloc[-1] - 1) * position_weight * 100

            results.append({
                'stock': stock,
                'total_return': adjusted_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd * 100,
                'win_rate': win_rate * 100,
                'profit_factor': profit_factor,
                'volatility': strategy_returns.std() * np.sqrt(252) * 100,
                'position_size': position_sizes.get(stock, 0)
            })

            # 累加到组合价值
            portfolio_value *= (1 + df['strategy_returns'].fillna(0).values * position_weight)

        return pd.DataFrame(results), common_dates, portfolio_value

    def generate_risk_report(self, selected_stocks, position_sizes):
        """
        生成风险报告

        Parameters:
        -----------
        selected_stocks : list
            选中的股票列表
        position_sizes : dict
            仓位配置
        """
        report = []
        report.append("=" * 60)
        report.append("风险管理报告")
        report.append("=" * 60)

        # 市场风险评估
        market_regime = self.check_market_regime()
        report.append(f"\n【市场环境】")
        report.append(f"当前市场状态: {market_regime}")

        if market_regime == 'RISK_OFF':
            report.append("⚠️ 市场风险较高，建议降低仓位")
        elif market_regime == 'RISK_ON':
            report.append("✅ 市场环境良好，可正常配置")
        else:
            report.append("⚡ 市场中性，保持谨慎")

        # 选中股票风险分析
        report.append(f"\n【选中股票风险分析】")
        report.append(f"共选中 {len(selected_stocks)} 只股票")

        for stock in selected_stocks:
            if stock in self.risk_metrics:
                metrics = self.risk_metrics[stock]
                stock_name = self.get_stock_name(stock)
                report.append(f"\n{stock} ({stock_name}):")
                report.append(f"  - 风险评分: {metrics['risk_score']:.1f}/100")
                report.append(f"  - 年化波动率: {metrics['volatility']:.1%}")
                report.append(f"  - 最大回撤(60日): {metrics['max_drawdown_60d']:.1%}")
                report.append(f"  - 夏普比率: {metrics['sharpe_ratio']:.2f}")
                report.append(f"  - 建议仓位: ¥{position_sizes.get(stock, 0):,.0f}")

        # 止损设置
        report.append(f"\n【止损设置】")
        stop_losses = self.generate_stop_loss_levels(selected_stocks)

        for stock, levels in stop_losses.items():
            stock_name = self.get_stock_name(stock)
            report.append(f"\n{stock} ({stock_name}):")
            report.append(f"  - 当前价格: ¥{levels['current_price']:.2f}")
            report.append(f"  - 止损价格: ¥{levels['stop_loss']:.2f}")
            report.append(f"  - 止损距离: {levels['stop_loss_pct']:.1f}%")

        # 组合风险指标
        report.append(f"\n【组合风险指标】")

        total_position = sum(position_sizes.values())
        avg_risk_score = np.mean([self.risk_metrics[s]['risk_score']
                                  for s in selected_stocks if s in self.risk_metrics])
        avg_volatility = np.mean([self.risk_metrics[s]['volatility']
                                  for s in selected_stocks if s in self.risk_metrics])

        report.append(f"  - 总仓位: ¥{total_position:,.0f}")
        report.append(f"  - 平均风险评分: {avg_risk_score:.1f}/100")
        report.append(f"  - 平均波动率: {avg_volatility:.1%}")

        # 风险提示
        report.append(f"\n【风险提示】")
        if avg_risk_score > 60:
            report.append("⚠️ 组合整体风险偏高，建议减少仓位或增加防御性资产")
        if avg_volatility > 0.25:
            report.append("⚠️ 组合波动较大，注意控制回撤")
        if total_position > 80000:
            report.append("⚠️ 仓位较重，建议保留部分现金应对突发情况")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


# 命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='A股趋势跟踪 + 相对强度策略 (风险优化版)')

    # 基本参数
    parser.add_argument('--start-date', '-s', default='20250101',
                       help='开始日期，格式YYYYMMDD (默认: 20250101)')
    parser.add_argument('--end-date', '-e', default=None,
                       help='结束日期，格式YYYYMMDD (默认: 今天)')
    parser.add_argument('--qlib-dir', default='~/.qlib/qlib_data/cn_data',
                       help='qlib数据目录路径 (默认: ~/.qlib/qlib_data/cn_data)')

    # 股票池配置
    parser.add_argument('--pool-mode', choices=['auto', 'index', 'custom'], default='auto',
                       help='股票池模式: auto(自动从qlib获取所有可用股票), index(指数成分股), custom(自定义)')
    parser.add_argument('--index-code', default='000300',
                       help='指数代码，当pool-mode=index时使用 (默认: 000300沪深300)')
    parser.add_argument('--stocks', nargs='*',
                       help='自定义股票代码列表(6位格式)，当pool-mode=custom时使用，如: 000001 600000 300750')
    parser.add_argument('--max-stocks', type=int, default=200,
                       help='auto模式下的最大股票数量 (默认: 200)')

    # 输出选项
    parser.add_argument('--no-dashboard', action='store_true',
                       help='不生成风险仪表板HTML文件')
    parser.add_argument('--no-backtest', action='store_true',
                       help='不运行回测')

    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()

    # 处理自定义股票列表
    custom_stocks = args.stocks if args.pool_mode == 'custom' else None

    # 初始化风险敏感策略
    strategy = RiskSensitiveTrendStrategy(
        start_date=args.start_date,
        end_date=args.end_date,
        qlib_dir=args.qlib_dir,
        stock_pool_mode=args.pool_mode,
        custom_stocks=custom_stocks,
        index_code=args.index_code
    )

    # 如果是auto模式，设置最大股票数
    if args.pool_mode == 'auto':
        strategy.max_stocks = args.max_stocks

    # 运行策略
    selected_stocks, position_sizes = strategy.run_strategy()

    if selected_stocks:
        # 显示选中股票（含股票名称）
        print(f"\n策略选中的股票:")
        for stock in selected_stocks:
            stock_name = strategy.get_stock_name(stock)
            print(f"  {stock} - {stock_name}")

        # 显示风险调整后的相对强度（添加股票名称）
        print("\n风险调整后相对强度TOP10:")
        top10_rs = strategy.rs_scores[['stock_code', 'rs_score', 'risk_score',
                                      'volatility', 'sharpe_ratio']].head(10).copy()
        top10_rs['stock_name'] = top10_rs['stock_code'].apply(strategy.get_stock_name)
        print(top10_rs[['stock_code', 'stock_name', 'rs_score', 'risk_score',
                       'volatility', 'sharpe_ratio']])

        # 显示仓位配置（含股票名称）
        print("\n仓位配置:")
        for stock, size in position_sizes.items():
            stock_name = strategy.get_stock_name(stock)
            print(f"  {stock} - {stock_name}: ¥{size:,.0f}")

        # 生成风险报告
        risk_report = strategy.generate_risk_report(selected_stocks, position_sizes)
        print("\n" + risk_report)

        # 绘制风险仪表板
        fig = strategy.plot_risk_dashboard(selected_stocks, position_sizes)
        # 保存为HTML文件而不是直接显示
        fig.write_html("risk_dashboard.html")
        print("风险仪表板已保存为 risk_dashboard.html")

        # 运行带风险管理的回测
        backtest_results, dates, portfolio_value = strategy.backtest_with_risk_management(
            selected_stocks, position_sizes
        )

        if backtest_results is not None:
            print("\n回测结果（风险调整后）:")
            # 在回测结果中添加股票名称
            backtest_display = backtest_results.copy()
            backtest_display['stock_name'] = backtest_display['stock'].apply(strategy.get_stock_name)
            # 重新排列列顺序
            cols = ['stock', 'stock_name'] + [col for col in backtest_display.columns if col not in ['stock', 'stock_name']]
            backtest_display = backtest_display[cols]
            print(backtest_display)
            print(f"\n组合整体表现:")
            print(f"  - 平均收益率: {backtest_results['total_return'].mean():.2f}%")
            print(f"  - 平均夏普比率: {backtest_results['sharpe_ratio'].mean():.2f}")
            print(f"  - 平均最大回撤: {backtest_results['max_drawdown'].mean():.2f}%")
            print(f"  - 平均胜率: {backtest_results['win_rate'].mean():.1f}%")

            # 绘制组合净值曲线
            fig_portfolio = go.Figure()
            fig_portfolio.add_trace(go.Scatter(
                x=dates,
                y=portfolio_value,
                mode='lines',
                name='组合净值',
                line=dict(color='blue', width=2)
            ))

            fig_portfolio.update_layout(
                title='组合净值曲线（风险调整后）',
                xaxis_title='日期',
                yaxis_title='净值',
                hovermode='x',
                height=400
            )
            # 保存为HTML文件而不是直接显示
            fig_portfolio.write_html("portfolio_curve.html")
            print("组合净值曲线已保存为 portfolio_curve.html")
    else:
        print("没有符合风险条件的股票")


if __name__ == "__main__":
    main()