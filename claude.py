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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial, lru_cache
import threading
import multiprocessing as mp
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️  Numba未安装，将使用标准pandas计算（建议安装numba以获得更好性能）")
import random
import logging
import json
warnings.filterwarnings('ignore')


class RiskSensitiveTrendStrategy:
    """风险敏感型趋势跟踪 + 相对强度策略"""

    def __init__(self, start_date='20230101', end_date=None, qlib_dir="~/.qlib/qlib_data/cn_data",
                 stock_pool_mode='auto', custom_stocks=None, index_code='000300', filter_st=False):
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
        filter_st : bool
            是否过滤ST股票，True=过滤ST股票，False=保留ST股票
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y%m%d')
        self.qlib_dir = os.path.expanduser(qlib_dir)
        self.stock_pool_mode = stock_pool_mode
        self.custom_stocks = custom_stocks or []
        self.index_code = index_code
        self.filter_st = filter_st
        self.stock_pool = []
        self.price_data = {}
        self.rs_scores = pd.DataFrame()
        self.risk_metrics = {}
        # 原始6位代码 → 规范化(带交易所前缀)代码的映射
        self.code_alias: dict[str, str] = {}
        self._qlib_initialized = False

        # 风险参数
        self.max_drawdown_threshold = 0.15  # 最大回撤阈值15%
        self.volatility_threshold = 0.35    # 年化波动率阈值35%
        self.atr_multiplier = 2.0          # ATR止损倍数
        self.risk_per_trade = 0.02         # 每笔交易风险2%
        self.max_correlation = 0.7         # 最大相关性阈值

        # 回撤门控参数（基于指数）
        self.drawdown_lookback = 252            # 回撤观测窗口（默认1年，单位：交易日）
        self.drawdown_risk_off_scale = 0.0      # 风险关闭时的仓位缩放（0=清仓，可设为0.3等）
        self._risk_regime_df = None             # 预计算的风险门控表：drawdown / risk_on

        # A股交易制度参数
        self.t_plus_1 = True               # T+1交易制度
        self.price_limit_pct = 0.10        # 沪深涨跌停幅度（10%）
        self.st_limit_pct = 0.05           # ST股涨跌停幅度（5%）
        self.bj_limit_pct = 0.30           # 北交所涨跌停幅度（30%）

        # 交易费用分拆（符合A股实际费率）
        self.commission_rate = 0.0003      # 券商佣金率（双边各0.03%）
        self.commission_min = 5.0          # 最低佣金5元
        self.stamp_tax_rate = 0.0005       # 印花税率（卖出单边0.05%，2023-08-28下调）
        self.transfer_fee_rate = 0.00002   # 过户费率（双边各0.002%）

        # 向后兼容：总体交易成本（用于简化计算的地方）
        self.transaction_cost = self.commission_rate + self.stamp_tax_rate/2 + self.transfer_fee_rate

        self.slippage_bps = 5              # 滑点（5个基点）

        # ST股票本地缓存
        self._local_st_stocks = self._load_local_st_stocks()

        # 股票名称映射缓存，避免频繁网络请求
        self._code_name_map = {}
        self._name_cache_built = False

        # T+1持仓账本：记录每笔买入的可卖日期
        self.position_ledger = {}  # {stock_code: [{'shares': int, 'buy_date': str, 'sellable_date': str, 'buy_price': float}]}

        # 流动性过滤参数
        self.min_adv_20d = 20_000_000      # 20日平均成交额阈值：2000万元
        self.min_adv_20d_bj = 50_000_000   # 北交所单独阈值：5000万元（更严格）
        self.max_suspend_days_60d = 10     # 60日内最大停牌天数
        self.exclude_bj_stocks = True      # 默认排除北交所股票（风险控制）

        # ADV单位校准参数
        self.amount_scale = None           # amount字段的单位缩放：None=自动检测, 1=元, 10000=万元

        # 性能优化配置
        self.enable_numba = NUMBA_AVAILABLE        # 是否启用Numba加速
        self.enable_vectorized_indicators = True  # 是否使用面板化技术指标计算
        self.enable_vectorized_tradable = True    # 是否使用向量化可交易性掩码
        self.io_workers_ratio = 0.75               # I/O线程数相对于CPU核心数的比例
        self.cpu_workers_ratio = 0.5               # CPU进程数相对于CPU核心数的比例

        # 交易统计和审计
        self.trading_stats = {
            'total_orders': 0,
            'successful_fills': 0,
            'partial_fills': 0,
            'rejected_orders': 0,
            'price_limited_orders': 0,
            'volume_limited_orders': 0,
            'total_slippage': 0.0,
            'total_transaction_costs': 0.0,
            'fill_ratio_sum': 0.0
        }
        self.audit_log = []  # 详细的交易审计日志

        # 初始化日志
        self._setup_logging()

        # 初始化qlib
        self._init_qlib()

        # 初始化名称映射缓存（在后台进行，不阻塞主流程）
        self._build_name_cache_async()

    def _load_local_st_stocks(self):
        """从本地JSON文件加载ST股票列表"""
        st_file_path = "st_stocks_akshare.json"
        try:
            with open(st_file_path, 'r', encoding='utf-8') as f:
                st_data = json.load(f)

            # 提取股票代码
            st_codes = {item['code'] for item in st_data}
            print(f"📋 从本地文件加载了 {len(st_codes)} 只ST股票")
            # 显示部分ST股票信息用于验证
            if st_codes:
                sample_names = [item['name'] for item in st_data[:5]]  # 显示前5个
                print(f"   示例ST股票: {', '.join(sample_names)}")

            return st_codes

        except FileNotFoundError:
            print(f"⚠️  本地ST股票文件 {st_file_path} 未找到，将不进行ST股票过滤")
            return set()
        except Exception as e:
            print(f"❌ 加载本地ST股票文件失败: {e}")
            return set()

    def _build_name_cache_async(self):
        """异步构建股票名称映射缓存"""
        def _build_cache():
            try:
                # 一次性获取全市场A股名称映射
                print("🔄 正在构建股票名称缓存...")
                name_map = {}

                # 获取A股信息
                try:
                    df_a = ak.stock_info_a_code_name()
                    if df_a is not None and not df_a.empty:
                        # 兼容不同的列名
                        code_col = None
                        name_col = None
                        for c in df_a.columns:
                            if '代码' in c or 'code' in c.lower():
                                code_col = c
                            if '简称' in c or '名称' in c or 'name' in c.lower():
                                name_col = c

                        if code_col and name_col:
                            for _, row in df_a.iterrows():
                                code = str(row[code_col]).strip()
                                name = str(row[name_col]).strip()
                                if code and name and len(code) == 6:
                                    name_map[code] = name
                except Exception:
                    pass

                # 获取北交所信息
                try:
                    df_bj = ak.stock_info_bj_name_code()
                    if df_bj is not None and not df_bj.empty:
                        code_col = None
                        name_col = None
                        for c in df_bj.columns:
                            if '代码' in c or 'code' in c.lower():
                                code_col = c
                            if '简称' in c or '名称' in c or 'name' in c.lower():
                                name_col = c

                        if code_col and name_col:
                            for _, row in df_bj.iterrows():
                                code = str(row[code_col]).strip()
                                name = str(row[name_col]).strip()
                                # 统一存储为6位代码
                                if code and name:
                                    if len(code) == 8 and code.startswith('BJ'):
                                        code = code[2:]
                                    if len(code) == 6:
                                        name_map[code] = name
                except Exception:
                    pass

                self._code_name_map = name_map
                self._name_cache_built = True
                print(f"✅ 股票名称缓存构建完成，共缓存 {len(name_map)} 只股票")

            except Exception as e:
                print(f"⚠️  股票名称缓存构建失败: {e}")
                self._name_cache_built = True  # 标记为已尝试，避免重复尝试

        # 在后台线程中构建缓存
        threading.Thread(target=_build_cache, daemon=True).start()

    @lru_cache(maxsize=8192)
    def get_stock_name(self, stock_code: str) -> str:
        """获取股票名称（优化版，使用缓存）"""
        code = str(stock_code).strip().upper()
        # 提取6位数字代码
        numeric = code[2:] if len(code) > 6 and code[:2] in ("SH", "SZ", "BJ") else code

        # 如果缓存已构建，直接从缓存获取
        if self._name_cache_built and numeric in self._code_name_map:
            return self._code_name_map[numeric]

        # 缓存未构建或未命中时，回退到原始方法（但只对特定股票调用）
        if not self._name_cache_built:
            # 如果缓存正在构建，先返回股票代码，避免阻塞
            return stock_code

        # 缓存已构建但未命中，可能是新股或特殊情况，进行单次查询
        try:
            info = ak.stock_individual_info_em(symbol=numeric)
            if info is not None and not info.empty and {"item", "value"}.issubset(set(info.columns)):
                row = info.loc[info["item"].isin(["股票简称", "证券简称"])]
                if not row.empty:
                    name_val = str(row["value"].iloc[0]).strip()
                    if name_val:
                        # 更新缓存
                        self._code_name_map[numeric] = name_val
                        return name_val
        except Exception:
            pass

        # 最后回退
        return stock_code

    def _setup_logging(self):
        """设置交易审计日志"""
        # 创建交易日志器
        self.trade_logger = logging.getLogger('trading_audit')
        self.trade_logger.setLevel(logging.INFO)

        # 避免重复添加handler
        if not self.trade_logger.handlers:
            # 文件handler
            log_filename = f"trading_audit_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(log_filename)
            file_handler.setLevel(logging.INFO)

            # 格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)

            self.trade_logger.addHandler(file_handler)

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
            elif c[0] in ('4', '8'):
                return 'BJ' + c
        return c

    def _convert_date_format(self, date_str: str) -> str:
        """转换日期格式从YYYYMMDD到YYYY-MM-DD"""
        if len(date_str) == 8 and date_str.isdigit():
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return date_str

    def _to_yyyymmdd(self, date_str: str) -> str:
        s = str(date_str).strip()
        if len(s) == 8 and s.isdigit():
            return s
        return s.replace("-", "")

    def _list_all_qlib_instruments_in_range(self) -> list[str]:
        """按时间窗获取全市场可交易股票（用 Qlib 官方接口过滤，不再手工枚举）"""
        assert self._qlib_initialized
        start_date_qlib = self._convert_date_format(self.start_date)
        end_date_qlib = self._convert_date_format(self.end_date)
        instruments_cfg = D.instruments(market="all")
        codes = D.list_instruments(
            instruments=instruments_cfg,
            start_time=start_date_qlib,
            end_time=end_date_qlib,
            as_list=True,
        )
        return [c[2:] if c.startswith(("SH", "SZ", "BJ")) else c for c in codes]

    def _fetch_sh_index_df(self):
        """
        获取上证指数（sh000001）日线数据：Qlib 优先，缺失则回退 AkShare。
        返回包含至少 ['close'] 列的 DataFrame（索引为日期）。
        """
        # --- Qlib 尝试 ---
        start_q = self._convert_date_format(self.start_date)
        end_q = self._convert_date_format(self.end_date)
        qlib_code = "SH000001"
        try:
            df = D.features(
                instruments=[qlib_code],
                fields=["$open", "$high", "$low", "$close", "$volume"],
                start_time=start_q,
                end_time=end_q,
                freq="day",
                disk_cache=1,  # 开启数据集缓存，显著提升I/O性能
            )
        except Exception:
            df = None

        if df is not None and not df.empty:
            df = df.xs(qlib_code, level=0)
            df.columns = [c.replace("$", "") for c in df.columns]
            df = df.astype(float)
            df.index.name = "date"
            return df

        # --- AkShare 回退 ---
        start_em = self._to_yyyymmdd(self.start_date)
        end_em = self._to_yyyymmdd(self.end_date)
        idx = ak.stock_zh_index_daily_em(symbol="sh000001", start_date=start_em, end_date=end_em)
        if idx is None or idx.empty:
            return None
        if "date" in idx.columns:
            idx = idx.set_index("date")
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in idx.columns]
        return idx[keep]

    def _build_risk_regime(self):
        """
        基于上证指数构建回撤门控：
        - 以收盘价的历史峰值计算回撤序列
        - 当回撤不超过阈值（例如15%）→ risk_on=True，否则 False
        """
        try:
            idx = self._fetch_sh_index_df()
            if idx is None or idx.empty or 'close' not in idx.columns:
                # 若无法获取指数数据，则默认全程 risk_on
                self._risk_regime_df = pd.DataFrame({'risk_on': []})
                return

            close = idx['close'].astype(float).dropna()
            # 可选：仅在观测窗口内做局部峰值；默认用全局峰值
            rolling_peak = close.cummax()
            dd = (close / rolling_peak) - 1.0
            df = pd.DataFrame({'drawdown': dd})
            df['risk_on'] = df['drawdown'].ge(-float(self.max_drawdown_threshold))
            self._risk_regime_df = df
        except Exception:
            # 兜底：任何异常均视为不启用门控
            self._risk_regime_df = pd.DataFrame({'risk_on': []})

    def is_risk_on(self, date_str: str) -> bool:
        """
        查询某交易日是否处于 risk-on 状态；若无该日（节假日），向前寻找最近一个交易日。
        输入日期可为 'YYYYMMDD' 或 'YYYY-MM-DD'。
        """
        if self._risk_regime_df is None:
            self._build_risk_regime()
        if self._risk_regime_df is None or self._risk_regime_df.empty:
            return True

        s = str(date_str).replace('-', '')
        if len(s) >= 8:
            ts = pd.to_datetime(f"{s[:4]}-{s[4:6]}-{s[6:8]}")
        else:
            ts = pd.to_datetime(date_str)

        idx = self._risk_regime_df.index
        if ts in idx:
            return bool(self._risk_regime_df.loc[ts, 'risk_on'])

        # 找到不超过 ts 的最近日期
        pos = idx.searchsorted(ts, side='right') - 1
        if pos >= 0:
            return bool(self._risk_regime_df.iloc[pos]['risk_on'])
        # 若在最左侧之前，视为 risk_on（保守）
        return True

    def scale_weights_by_drawdown(self, weights):
        """
        对按“日期索引”的权重序列/权重矩阵执行回撤门控缩放：
        - 当 risk_on=True → 权重不变
        - 当 risk_on=False → 权重乘以 drawdown_risk_off_scale

        参数
        ----
        weights : pandas.Series 或 pandas.DataFrame
            index 为日期（DatetimeIndex 或能被 to_datetime 解析）。
        返回
        ----
        与输入同类型的对象，按日缩放后的权重。
        """
        if weights is None:
            return None
        if self._risk_regime_df is None:
            self._build_risk_regime()
        if self._risk_regime_df is None or self._risk_regime_df.empty:
            return weights

        # 统一日期索引
        w = weights.copy()
        if not isinstance(w.index, pd.DatetimeIndex):
            w.index = pd.to_datetime(w.index)

        gate = self._risk_regime_df['risk_on'].astype(int)
        gate = gate.reindex(w.index).fillna(method='ffill').fillna(1).astype(int)
        # 1 → 保持；0 → 乘以 off_scale
        scale = gate + (1 - gate) * float(self.drawdown_risk_off_scale)
        if isinstance(w, pd.Series):
            return w.mul(scale)
        else:
            return w.mul(scale, axis=0)

    def analyze_portfolio_drawdown(self, daily_returns: pd.Series) -> dict:
        """
        对组合日收益率进行回撤分析，返回与 Qlib 报告口径一致的核心指标。
        返回字段：
        - max_drawdown: float，最大回撤（负数）
        - nav_end: float，期末净值
        """
        if daily_returns is None or len(daily_returns) == 0:
            return {'max_drawdown': 0.0, 'nav_end': 1.0}
        ret = pd.Series(daily_returns).astype(float).fillna(0.0)
        # 允许索引不是日期；不强制转换
        nav = (1.0 + ret).cumprod()
        peak = nav.cummax()
        drawdown = nav / peak - 1.0
        return {
            'max_drawdown': float(drawdown.min()),
            'nav_end': float(nav.iloc[-1])
        }


    def _is_st_stock(self, stock_code: str) -> bool:
        """
        简化的ST股票判断（基于本地缓存文件）

        Parameters:
        -----------
        stock_code : str
            股票代码（6位数字格式）
        """
        # 规范化代码为6位数字
        numeric_code = stock_code
        if len(stock_code) > 6:
            numeric_code = stock_code[2:] if stock_code[:2] in ('SH', 'SZ', 'BJ') else stock_code
        numeric_code = str(numeric_code).zfill(6)

        return numeric_code in self._local_st_stocks

    def get_all_available_stocks(self):
        """
        从qlib数据中获取所有在指定日期范围内有数据的股票
        """
        assert self._qlib_initialized
        print("正在从 Qlib instruments 中读取全市场股票列表（按时间窗口过滤）...")
        codes = self._list_all_qlib_instruments_in_range()
        print(f"全市场在 {self._convert_date_format(self.start_date)} ~ {self._convert_date_format(self.end_date)} 范围内可交易的股票数: {len(codes)}")
        return codes

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
            max_stocks = getattr(self, 'max_stocks', None)
            self.stock_pool = self._get_universe_stocks_with_filters(max_stocks)

        return self.stock_pool

    def _get_universe_stocks_with_filters(self, max_stocks=None):
        """
        获取全市场股票池并应用质量过滤（减少生存者偏差）

        Parameters:
        -----------
        max_stocks : int, optional
            最大股票数量限制，None表示不限制
        """
        try:
            print("构建全市场股票池，应用流动性和基本面过滤...")

            # 候选池：直接使用 Qlib 在时间窗口内的全市场股票
            candidate_pool = self._list_all_qlib_instruments_in_range()
            print(f"候选股票数量（来自 Qlib instruments）：{len(candidate_pool)}")

            # 首先剔除ST股票（如果启用ST过滤）
            if self.filter_st and self._local_st_stocks:
                original_count = len(candidate_pool)
                candidate_pool = [code for code in candidate_pool if code not in self._local_st_stocks]
                removed_count = original_count - len(candidate_pool)
                print(f"🚫 已剔除 {removed_count} 只ST股票，剩余 {len(candidate_pool)} 只股票")

            # 批量过滤：检查数据可用性和基本质量
            print("📊 开始股票池质量过滤...")
            filtered_stocks = []
            start_date_qlib = self._convert_date_format(self.start_date)
            end_date_qlib = self._convert_date_format(self.end_date)

            # 使用并发处理批量筛选
            batch_size = 200  # 提高批量大小到200，减少调用次数和I/O开销
            batches = [candidate_pool[i:i+batch_size] for i in range(0, len(candidate_pool), batch_size)]

            # 优化并发策略：I/O密集型使用线程池
            io_workers = max(1, int(mp.cpu_count() * 0.75))  # I/O密集型可以使用更多线程
            print(f"股票池筛选使用{io_workers}个I/O线程处理{len(batches)}个批次")

            with ThreadPoolExecutor(max_workers=io_workers) as executor:
                # 提交所有批次任务
                future_to_batch = {
                    executor.submit(self._process_stock_batch, batch, start_date_qlib, end_date_qlib): batch
                    for batch in batches
                }

                # 处理完成的批次，支持提前中断
                batch_count = 0
                for future in as_completed(future_to_batch):
                    batch_count += 1
                    batch = future_to_batch[future]

                    try:
                        batch_filtered = future.result()
                        if batch_filtered:
                            # 检查是否需要提前中断
                            if max_stocks is not None and len(filtered_stocks) + len(batch_filtered) > max_stocks:
                                # 只添加需要的数量
                                remaining = max_stocks - len(filtered_stocks)
                                if remaining > 0:
                                    filtered_stocks.extend(batch_filtered[:remaining])
                                print(f"批次进度: {batch_count}/{len(batches)}, 已筛选: {len(filtered_stocks)} (已达到max_stocks={max_stocks}，提前停止)")
                                # 取消剩余任务
                                for f in future_to_batch:
                                    if not f.done():
                                        f.cancel()
                                break
                            else:
                                filtered_stocks.extend(batch_filtered)

                        # 每处理5个批次或最后一个批次才显示进度
                        if batch_count % 5 == 0 or batch_count == len(batches):
                            progress_pct = (batch_count / len(batches)) * 100
                            print(f"批次进度: {batch_count}/{len(batches)} ({progress_pct:.1f}%), 已筛选: {len(filtered_stocks)} 只股票")
                    except Exception as e:
                        print(f"处理批次时出错: {e}")

            print(f"✅ 股票池筛选完成：从{len(candidate_pool)}个候选股票中筛选出{len(filtered_stocks)}只合格股票")

            # 随机化筛选结果，避免偏差
            if filtered_stocks:
                random.shuffle(filtered_stocks)
                print("已随机打乱股票顺序，避免筛选偏差")

            # 注意：数量限制现在在并发处理中已经控制，这里无需额外处理

            return filtered_stocks

        except Exception as e:
            print(f"构建股票池失败: {e}")
            # 降级到原有方法
            all_stocks = self.get_all_available_stocks()
            if max_stocks is not None and len(all_stocks) > max_stocks:
                random.shuffle(all_stocks)
                all_stocks = all_stocks[:max_stocks]
                print(f"降级方法：随机选择{max_stocks}只股票")
            return all_stocks

    def _process_stock_batch(self, batch, start_date_qlib, end_date_qlib):
        """
        并发处理单个股票批次的筛选（用于股票池构建）

        Parameters:
        -----------
        batch : list
            股票代码批次
        start_date_qlib : str
            开始日期（qlib格式）
        end_date_qlib : str
            结束日期（qlib格式）
        """
        batch_filtered = []
        batch_codes = [self._normalize_instrument(code) for code in batch]

        # 批量获取数据（增加成交额字段用于流动性过滤）
        batch_data = D.features(
            instruments=batch_codes,
            fields=['$close', '$volume', '$amount'],  # 添加成交额
            start_time=start_date_qlib,
            end_time=end_date_qlib,
            freq='day',
            disk_cache=1  # 开启数据集缓存，显著提升I/O性能
        )

        if batch_data is not None and not batch_data.empty:
            # 检查每只股票的数据质量
            for j, code in enumerate(batch):
                qlib_code = batch_codes[j]
                if qlib_code in batch_data.index.get_level_values(0):
                    stock_data = batch_data.xs(qlib_code, level=0)

                    # 应用基本过滤条件
                    if self._apply_stock_filters(stock_data, code):
                        batch_filtered.append(code)

        return batch_filtered

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
            # 北交所股票特殊处理
            normalized_code = self._normalize_instrument(stock_code)
            is_bj_stock = normalized_code.startswith('BJ')

            # 如果启用北交所排除，直接过滤
            if is_bj_stock and self.exclude_bj_stocks:
                return False

            # 基本数据量要求
            if len(stock_data) < 10:  # 降低数据量要求
                return False

            # 增强流动性过滤
            if 'volume' in stock_data.columns:
                # 1. 基础流动性：最近5天有成交
                recent_volume = stock_data['volume'].iloc[-5:].sum()
                if recent_volume <= 0:  # 最近5天无成交
                    return False

                # 2. 停牌天数过滤：60日内停牌天数不超过阈值
                volume_60d = stock_data['volume'].iloc[-60:] if len(stock_data) >= 60 else stock_data['volume']
                suspend_days = (volume_60d <= 0).sum()
                if suspend_days > self.max_suspend_days_60d:
                    return False

            # 3. 日均成交额过滤：ADV20要求（北交所使用更严格标准）
            if 'amount' in stock_data.columns and len(stock_data) >= 20:
                amount_20d = stock_data['amount'].iloc[-20:]
                # 使用动态单位缩放
                scale_factor = self._get_amount_scale()
                avg_amount = amount_20d.mean() * scale_factor  # 转换为元

                # 北交所使用更严格的ADV阈值
                min_adv = self.min_adv_20d_bj if is_bj_stock else self.min_adv_20d
                if avg_amount < min_adv:
                    return False

            # 去除价格异常股票
            if 'close' in stock_data.columns:
                recent_prices = stock_data['close'].iloc[-10:]
                if recent_prices.std() / recent_prices.mean() > 2:  # 价格波动过大
                    return False
                if recent_prices.iloc[-1] < 1:  # 股价过低
                    return False

            # ST股票已在候选池阶段预先剔除，此处无需重复过滤

            return True

        except Exception:
            return False

    def _get_price_limits(self, yesterday_close, stock_code=None, is_st=None):
        """
        计算涨跌停价格限制（优化版：独立板块识别和ST识别）

        Parameters:
        -----------
        yesterday_close : float
            昨日收盘价
        stock_code : str, optional
            股票代码，用于自动判断类型
        is_st : bool, optional
            是否为ST股票，如果提供则直接使用
        """
        # 统一带前缀代码
        code = (stock_code or '').strip().upper() if stock_code else ''

        # 板块识别（优先级最高，独立于ST识别）
        if code.startswith('BJ'):
            # 北交所30%
            limit_pct = self.bj_limit_pct
        elif code.startswith('SH688') or code.startswith('SZ30'):
            # 科创板(688)或创业板(30)20%
            limit_pct = 0.20
        else:
            # 沪深主板，需要判断ST状态
            if is_st is None and stock_code is not None:
                # 提取数值代码用于ST判定
                code_up = str(stock_code).strip().upper()
                numeric = code_up[2:] if len(code_up) > 6 and code_up[:2] in ('SH','SZ','BJ') else code_up
                is_st = self._is_st_stock(numeric)

            if is_st:
                # ST股票5%
                limit_pct = self.st_limit_pct
            else:
                # 主板普通股票10%
                limit_pct = self.price_limit_pct

        upper_limit = yesterday_close * (1 + limit_pct)
        lower_limit = yesterday_close * (1 - limit_pct)
        return upper_limit, lower_limit

    def _calculate_transaction_cost(self, price, shares, is_buy=True):
        """
        计算按边计费的交易成本（简化版，返回总成本）

        Parameters:
        -----------
        price : float
            成交价格
        shares : int
            成交股数
        is_buy : bool
            是否为买入订单

        Returns:
        --------
        float
            总交易成本（元）
        """
        trade_amount = price * shares
        cost_details = self._calculate_transaction_costs(trade_amount, is_buy)
        return cost_details['total_cost']

    def _get_next_trading_date(self, date_str):
        """
        获取下一个交易日（T+1）

        Parameters:
        -----------
        date_str : str
            当前日期，格式YYYYMMDD

        Returns:
        --------
        str
            下一个交易日，格式YYYYMMDD
        """
        from datetime import datetime, timedelta

        current_date = datetime.strptime(date_str, '%Y%m%d')
        next_date = current_date + timedelta(days=1)

        # 简化处理：假设下一天就是下一个交易日
        # 在实际应用中，应该查询交易日历
        return next_date.strftime('%Y%m%d')

    def _add_position_to_ledger(self, stock_code, shares, buy_date, buy_price):
        """
        向持仓账本添加新的买入记录

        Parameters:
        -----------
        stock_code : str
            股票代码
        shares : int
            买入股数
        buy_date : str
            买入日期，格式YYYYMMDD
        buy_price : float
            买入价格
        """
        if stock_code not in self.position_ledger:
            self.position_ledger[stock_code] = []

        sellable_date = self._get_next_trading_date(buy_date)

        position_record = {
            'shares': shares,
            'buy_date': buy_date,
            'sellable_date': sellable_date,
            'buy_price': buy_price
        }

        self.position_ledger[stock_code].append(position_record)

    def _get_sellable_shares(self, stock_code, current_date):
        """
        获取当前日期可卖出的股数

        Parameters:
        -----------
        stock_code : str
            股票代码
        current_date : str
            当前日期，格式YYYYMMDD

        Returns:
        --------
        int
            可卖出的股数
        """
        if stock_code not in self.position_ledger:
            return 0

        sellable_shares = 0
        for record in self.position_ledger[stock_code]:
            if record['sellable_date'] <= current_date:
                sellable_shares += record['shares']

        return sellable_shares

    def _remove_from_ledger(self, stock_code, shares_to_sell, current_date):
        """
        从持仓账本中移除卖出的股票（FIFO原则）

        Parameters:
        -----------
        stock_code : str
            股票代码
        shares_to_sell : int
            要卖出的股数
        current_date : str
            当前日期，格式YYYYMMDD

        Returns:
        --------
        bool
            是否成功移除（True表示有足够的可卖股数）
        """
        if stock_code not in self.position_ledger:
            return False

        remaining_to_sell = shares_to_sell
        records_to_remove = []

        # FIFO：从最早买入的开始卖出
        for i, record in enumerate(self.position_ledger[stock_code]):
            if record['sellable_date'] <= current_date and remaining_to_sell > 0:
                if record['shares'] <= remaining_to_sell:
                    # 这笔买入的股票全部卖出
                    remaining_to_sell -= record['shares']
                    records_to_remove.append(i)
                else:
                    # 这笔买入的股票部分卖出
                    record['shares'] -= remaining_to_sell
                    remaining_to_sell = 0
                    break

        # 移除已清仓的记录
        for i in reversed(records_to_remove):
            del self.position_ledger[stock_code][i]

        # 如果该股票已无持仓，删除整个条目
        if not self.position_ledger[stock_code]:
            del self.position_ledger[stock_code]

        return remaining_to_sell == 0

    def _detect_amount_scale(self, sample_stocks=None, sample_size=5):
        """
        自动检测amount字段的单位缩放

        Parameters:
        -----------
        sample_stocks : list, optional
            用于检测的样本股票代码，默认使用股票池中的前几只
        sample_size : int
            样本大小，默认5只股票

        Returns:
        --------
        float
            检测到的缩放因子：1表示"元"，10000表示"万元"
        """
        if not self._qlib_initialized:
            return 10000  # 默认假设万元

        # 选择样本股票
        if sample_stocks is None:
            sample_stocks = self.stock_pool[:sample_size] if len(self.stock_pool) >= sample_size else self.stock_pool

        if not sample_stocks:
            return 10000  # 默认假设万元

        total_amount_samples = []

        for stock_code in sample_stocks:
            try:
                # 获取最近几天的数据来判断数量级
                df = self.fetch_stock_data(stock_code)
                if df is not None and 'amount' in df.columns and len(df) > 0:
                    recent_amounts = df['amount'].iloc[-5:].dropna()
                    if len(recent_amounts) > 0:
                        avg_amount = recent_amounts.mean()
                        total_amount_samples.append(avg_amount)
            except Exception:
                continue

        if not total_amount_samples:
            print("警告：无法获取样本数据，使用默认ADV单位（万元）")
            return 10000

        # 分析数量级
        import numpy as np
        median_amount = np.median(total_amount_samples)

        # 启发式判断：如果中位数在千万以上，可能是"元"单位；如果在万以下，可能是"万元"单位
        if median_amount > 10_000_000:
            detected_scale = 1  # 元
            print(f"自动检测ADV单位：元（样本中位数：{median_amount:,.0f}）")
        else:
            detected_scale = 10000  # 万元
            print(f"自动检测ADV单位：万元（样本中位数：{median_amount:,.0f}）")

        return detected_scale

    def _get_amount_scale(self):
        """
        获取amount字段的缩放因子

        Returns:
        --------
        float
            缩放因子
        """
        if self.amount_scale is None:
            # 第一次调用时自动检测
            self.amount_scale = self._detect_amount_scale()

        return self.amount_scale

    def _simulate_order_execution(self, target_price, yesterday_close, target_shares, volume_available, stock_code=None, is_st=None, is_buy=True, max_participation_rate=0.1):
        """
        模拟A股订单执行（考虑涨跌停、滑点和成交量约束）

        Parameters:
        -----------
        target_price : float
            目标价格
        yesterday_close : float
            昨日收盘价
        target_shares : int
            目标成交股数
        volume_available : float
            当日可用成交量（股数）
        stock_code : str, optional
            股票代码，用于ST判断
        is_st : bool, optional
            是否为ST股票
        is_buy : bool
            是否为买单
        max_participation_rate : float
            最大成交量参与率，默认10%

        Returns:
        --------
        tuple
            (execution_result, error_message)
            execution_result包含: executed_shares, executed_price, transaction_cost, slippage, fill_ratio等
        """
        upper_limit, lower_limit = self._get_price_limits(yesterday_close, stock_code=stock_code, is_st=is_st)

        # 检查价格是否触及涨跌停（硬约束，直接拒绝成交）
        if is_buy:
            if target_price >= upper_limit:
                return None, "涨停无法买入"
            else:
                actual_price = target_price
        else:
            if target_price <= lower_limit:
                return None, "跌停无法卖出"
            else:
                actual_price = target_price

        # 成交量约束：限制最大可成交数量
        max_tradable_shares = int(volume_available * max_participation_rate) if volume_available > 0 else target_shares
        executed_shares = min(target_shares, max_tradable_shares)

        # 如果无法成交任何股数，返回失败
        if executed_shares <= 0:
            return None, "成交量不足，无法执行订单"

        # 应用滑点
        slippage = actual_price * self.slippage_bps / 10000
        if is_buy:
            final_price = actual_price + slippage
        else:
            final_price = actual_price - slippage

        # 计算交易成本
        cost = self._calculate_transaction_cost(final_price, executed_shares, is_buy=is_buy)

        # 计算成交率
        fill_ratio = executed_shares / target_shares if target_shares > 0 else 0.0

        # 更新交易统计
        self._update_trading_stats(target_shares, executed_shares, cost, slippage, fill_ratio,
                                   target_price != actual_price, executed_shares < target_shares)

        # 记录审计日志
        self._log_trade_audit(stock_code, target_shares, executed_shares, target_price, final_price,
                              cost, slippage, fill_ratio, is_buy, volume_available)

        return {
            'executed_shares': executed_shares,
            'executed_price': final_price,
            'transaction_cost': cost,
            'slippage': slippage,
            'fill_ratio': fill_ratio,
            'price_limited': target_price != actual_price,
            'volume_limited': executed_shares < target_shares,
            'unfilled_shares': target_shares - executed_shares
        }, None

    def _update_trading_stats(self, target_shares, executed_shares, cost, slippage, fill_ratio,
                             price_limited, volume_limited):
        """更新交易统计"""
        self.trading_stats['total_orders'] += 1

        if executed_shares > 0:
            self.trading_stats['successful_fills'] += 1
            self.trading_stats['total_transaction_costs'] += cost
            self.trading_stats['total_slippage'] += abs(slippage)
            self.trading_stats['fill_ratio_sum'] += fill_ratio

            if executed_shares < target_shares:
                self.trading_stats['partial_fills'] += 1
        else:
            self.trading_stats['rejected_orders'] += 1

        if price_limited:
            self.trading_stats['price_limited_orders'] += 1

        if volume_limited:
            self.trading_stats['volume_limited_orders'] += 1

    def _log_trade_audit(self, stock_code, target_shares, executed_shares, target_price,
                        final_price, cost, slippage, fill_ratio, is_buy, volume_available):
        """记录详细的交易审计日志"""
        audit_record = {
            'timestamp': datetime.now().isoformat(),
            'stock_code': stock_code,
            'direction': 'BUY' if is_buy else 'SELL',
            'target_shares': target_shares,
            'executed_shares': executed_shares,
            'target_price': target_price,
            'executed_price': final_price,
            'slippage': slippage,
            'transaction_cost': cost,
            'fill_ratio': fill_ratio,
            'volume_available': volume_available,
            'unfilled_shares': target_shares - executed_shares,
            'price_limited': target_price != final_price,
            'volume_limited': executed_shares < target_shares
        }

        # 添加到内存日志
        self.audit_log.append(audit_record)

        # 写入文件日志
        if hasattr(self, 'trade_logger'):
            log_message = (
                f"TRADE: {stock_code} {audit_record['direction']} "
                f"Target:{target_shares} Executed:{executed_shares} "
                f"Price:{final_price:.3f} Cost:{cost:.2f} "
                f"FillRatio:{fill_ratio:.2%} "
                f"Slippage:{slippage:.4f}"
            )
            self.trade_logger.info(log_message)

    def get_trading_statistics(self):
        """获取交易统计报告"""
        stats = self.trading_stats.copy()

        # 计算衍生指标
        if stats['total_orders'] > 0:
            stats['success_rate'] = stats['successful_fills'] / stats['total_orders']
            stats['rejection_rate'] = stats['rejected_orders'] / stats['total_orders']
            stats['partial_fill_rate'] = stats['partial_fills'] / stats['total_orders']
            stats['price_limit_rate'] = stats['price_limited_orders'] / stats['total_orders']
            stats['volume_limit_rate'] = stats['volume_limited_orders'] / stats['total_orders']

        if stats['successful_fills'] > 0:
            stats['avg_fill_ratio'] = stats['fill_ratio_sum'] / stats['successful_fills']
            stats['avg_transaction_cost'] = stats['total_transaction_costs'] / stats['successful_fills']
            stats['avg_slippage'] = stats['total_slippage'] / stats['successful_fills']

        return stats

    def export_audit_log(self, filename=None):
        """导出审计日志到文件"""
        if filename is None:
            filename = f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.audit_log, f, ensure_ascii=False, indent=2)

        print(f"审计日志已导出到: {filename}")
        return filename

    def create_enhanced_portfolio_dashboard(self, equity_curve, performance_stats, selected_stocks, position_sizes):
        """创建增强版组合分析仪表板"""

        # 创建子图布局 - 更多的分析图表
        fig = make_subplots(
            rows=5, cols=2,
            subplot_titles=[
                '净值曲线 & 回撤', '月度收益热力图',
                '日收益分布', '滚动夏普比率',
                '累计收益分解', '风险指标雷达图',
                '持仓权重分布', '个股贡献分析',
                '交易统计概览', '风险-收益散点图'
            ],
            specs=[
                [{'secondary_y': True}, {'type': 'heatmap'}],
                [{'type': 'histogram'}, {'type': 'scatter'}],
                [{'secondary_y': True}, {'type': 'scatterpolar'}],
                [{'type': 'pie'}, {'type': 'bar'}],
                [{'type': 'table'}, {'type': 'scatter'}]
            ],
            vertical_spacing=0.06,
            horizontal_spacing=0.1,
            row_heights=[0.25, 0.2, 0.2, 0.2, 0.15]
        )

        # 1. 净值曲线 & 回撤
        daily_returns = self.daily_return if hasattr(self, 'daily_return') and self.daily_return is not None else equity_curve.pct_change().dropna()

        # 计算回撤
        nav = equity_curve
        peak = nav.cummax()
        drawdown = (nav / peak - 1) * 100

        # 净值曲线
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode='lines',
                name='净值曲线',
                line=dict(color='blue', width=2),
                hovertemplate='日期: %{x}<br>净值: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )

        # 回撤曲线
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode='lines',
                name='回撤(%)',
                line=dict(color='red', width=1),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.3)',
                yaxis='y2',
                hovertemplate='日期: %{x}<br>回撤: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1, secondary_y=True
        )

        # 2. 月度收益热力图
        if len(daily_returns) > 30:
            monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
            monthly_df = monthly_returns.to_frame('return')
            monthly_df['year'] = monthly_df.index.year
            monthly_df['month'] = monthly_df.index.month

            # 创建透视表
            pivot_table = monthly_df.pivot(index='year', columns='month', values='return')

            fig.add_trace(
                go.Heatmap(
                    z=pivot_table.values,
                    x=[f"{i}月" for i in range(1, 13)],
                    y=pivot_table.index,
                    colorscale='RdYlGn',
                    name='月度收益(%)',
                    hovertemplate='%{y}年%{x}: %{z:.2f}%<extra></extra>'
                ),
                row=1, col=2
            )

        # 3. 日收益分布直方图
        fig.add_trace(
            go.Histogram(
                x=daily_returns * 100,
                nbinsx=50,
                name='日收益分布',
                marker_color='lightblue',
                opacity=0.7,
                hovertemplate='收益率: %{x:.2f}%<br>频次: %{y}<extra></extra>'
            ),
            row=2, col=1
        )

        # 4. 滚动夏普比率
        if len(daily_returns) > 63:
            rolling_sharpe = daily_returns.rolling(63).mean() / daily_returns.rolling(63).std() * np.sqrt(252)
            rolling_sharpe = rolling_sharpe.dropna()

            fig.add_trace(
                go.Scatter(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe.values,
                    mode='lines',
                    name='滚动夏普比率(63日)',
                    line=dict(color='green', width=2),
                    hovertemplate='日期: %{x}<br>夏普比率: %{y:.3f}<extra></extra>'
                ),
                row=2, col=2
            )

            # 添加参考线
            fig.add_hline(y=1.0, line_dash="dash", line_color="red", row=2, col=2)
            fig.add_hline(y=2.0, line_dash="dash", line_color="green", row=2, col=2)

        # 5. 累计收益分解 - 按年份
        yearly_returns = daily_returns.resample('Y').apply(lambda x: (1 + x).prod() - 1) * 100
        cumulative_yearly = (1 + yearly_returns/100).cumprod()

        fig.add_trace(
            go.Scatter(
                x=yearly_returns.index.year,
                y=cumulative_yearly.values,
                mode='lines+markers',
                name='年度累计收益',
                line=dict(color='purple', width=3),
                marker=dict(size=8),
                hovertemplate='年份: %{x}<br>累计收益: %{y:.2f}<extra></extra>'
            ),
            row=3, col=1
        )

        # 年度收益柱状图
        fig.add_trace(
            go.Bar(
                x=yearly_returns.index.year,
                y=yearly_returns.values,
                name='年度收益率(%)',
                marker_color=['green' if x > 0 else 'red' for x in yearly_returns.values],
                yaxis='y2',
                opacity=0.6,
                hovertemplate='年份: %{x}<br>年收益率: %{y:.2f}%<extra></extra>'
            ),
            row=3, col=1, secondary_y=True
        )

        # 6. 风险指标雷达图
        radar_metrics = {
            '收益率': min(performance_stats.get('annual_return', 0) * 5, 1),  # 标准化到0-1
            '夏普比率': min(max(performance_stats.get('sharpe', 0) / 3, 0), 1),
            '胜率': performance_stats.get('win_rate', 0),
            '稳定性': 1 - min(abs(performance_stats.get('max_drawdown', 0)) * 5, 1),
            'Sortino': min(max(performance_stats.get('sortino', 0) / 3, 0), 1),
            '信息比率': min(max(performance_stats.get('info_ratio', 0) / 2 + 0.5, 0), 1)
        }

        fig.add_trace(
            go.Scatterpolar(
                r=list(radar_metrics.values()),
                theta=list(radar_metrics.keys()),
                fill='toself',
                name='策略表现',
                line_color='blue'
            ),
            row=3, col=2
        )

        # 7. 持仓权重分布饼图
        if position_sizes:
            total_position = sum(position_sizes.values())
            weights = [(v/total_position)*100 for v in position_sizes.values()]
            stock_names = [f"{k}<br>{self.get_stock_name(k)}" for k in position_sizes.keys()]

            fig.add_trace(
                go.Pie(
                    labels=stock_names,
                    values=weights,
                    name="持仓权重",
                    hovertemplate='%{label}<br>权重: %{value:.1f}%<extra></extra>'
                ),
                row=4, col=1
            )

        # 8. 个股贡献分析（风险评分 vs 仓位）
        if selected_stocks and hasattr(self, 'risk_metrics'):
            risk_scores = []
            positions = []
            stock_labels = []

            for stock in selected_stocks:
                if stock in self.risk_metrics and stock in position_sizes:
                    risk_scores.append(self.risk_metrics[stock].get('risk_score', 0))
                    positions.append(position_sizes[stock])
                    stock_labels.append(f"{stock}<br>{self.get_stock_name(stock)}")

            fig.add_trace(
                go.Bar(
                    x=stock_labels,
                    y=positions,
                    name='仓位大小',
                    marker_color='lightgreen',
                    hovertemplate='%{x}<br>仓位: ¥%{y:,.0f}<extra></extra>'
                ),
                row=4, col=2
            )

        # 9. 交易统计表格
        trading_stats = self.get_trading_statistics()
        if trading_stats['total_orders'] > 0:
            table_data = [
                ['总订单数', f"{trading_stats['total_orders']}"],
                ['成功成交', f"{trading_stats['successful_fills']}"],
                ['成交率', f"{trading_stats.get('success_rate', 0):.2%}"],
                ['平均成交比例', f"{trading_stats.get('avg_fill_ratio', 0):.2%}"],
                ['平均交易成本', f"¥{trading_stats.get('avg_transaction_cost', 0):.2f}"],
                ['价格限制订单', f"{trading_stats['price_limited_orders']}"],
                ['成交量限制订单', f"{trading_stats['volume_limited_orders']}"]
            ]
        else:
            table_data = [['暂无交易统计', '请运行实际交易']]

        fig.add_trace(
            go.Table(
                header=dict(values=['指标', '数值'], fill_color='lightblue'),
                cells=dict(values=list(zip(*table_data)), fill_color='white')
            ),
            row=5, col=1
        )

        # 10. 风险-收益散点图（选中股票）
        if selected_stocks and hasattr(self, 'risk_metrics'):
            volatilities = []
            returns = []
            sizes = []
            colors = []
            labels = []

            for stock in selected_stocks:
                if stock in self.risk_metrics:
                    metrics = self.risk_metrics[stock]
                    volatilities.append(metrics.get('volatility', 0) * 100)
                    # 估算收益率（简化）
                    returns.append(metrics.get('sharpe_ratio', 0) * metrics.get('volatility', 0) * 100)
                    sizes.append(position_sizes.get(stock, 0) / 10000)  # 规模调整
                    colors.append(100 - metrics.get('risk_score', 50))  # 颜色表示质量
                    labels.append(f"{stock}<br>{self.get_stock_name(stock)}")

            fig.add_trace(
                go.Scatter(
                    x=volatilities,
                    y=returns,
                    mode='markers',
                    marker=dict(
                        size=[max(s, 10) for s in sizes],
                        color=colors,
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="质量分数")
                    ),
                    text=labels,
                    name='个股分析',
                    hovertemplate='%{text}<br>波动率: %{x:.1f}%<br>预期收益: %{y:.1f}%<extra></extra>'
                ),
                row=5, col=2
            )

        # 更新布局
        fig.update_layout(
            height=2000,
            title={
                'text': f'增强版组合分析报告 - {equity_curve.index[0].date()} 至 {equity_curve.index[-1].date()}',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            showlegend=False,
            template='plotly_white'
        )

        # 设置轴标签
        fig.update_xaxes(title_text="日期", row=1, col=1)
        fig.update_yaxes(title_text="净值", row=1, col=1)
        fig.update_yaxes(title_text="回撤(%)", secondary_y=True, row=1, col=1)

        fig.update_xaxes(title_text="日收益率(%)", row=2, col=1)
        fig.update_yaxes(title_text="频次", row=2, col=1)

        fig.update_xaxes(title_text="日期", row=2, col=2)
        fig.update_yaxes(title_text="夏普比率", row=2, col=2)

        fig.update_xaxes(title_text="年份", row=3, col=1)
        fig.update_yaxes(title_text="累计收益", row=3, col=1)
        fig.update_yaxes(title_text="年收益率(%)", secondary_y=True, row=3, col=1)

        fig.update_xaxes(title_text="波动率(%)", row=5, col=2)
        fig.update_yaxes(title_text="预期收益(%)", row=5, col=2)

        return fig

    def print_enhanced_metrics_summary(self, equity_curve, performance_stats, selected_stocks, position_sizes):
        """打印增强版分析报告的关键指标摘要到终端"""

        print("\n" + "="*100)
        print(" " * 35 + "📊 增强版策略分析报告摘要")
        print("="*100)

        # 基本信息
        print(f"\n🗓️  回测周期: {equity_curve.index[0].date()} 至 {equity_curve.index[-1].date()}")
        print(f"📈 选中股票: {len(selected_stocks)} 只")
        if position_sizes:
            total_position = sum(position_sizes.values())
            print(f"💰 总仓位: ¥{total_position:,.0f}")

        # 核心收益指标
        print(f"\n🎯 核心收益指标:")
        print(f"   总收益率          : {performance_stats.get('total_return', 0):>8.2%}")
        print(f"   年化收益率        : {performance_stats.get('annual_return', 0):>8.2%}")
        print(f"   年化波动率        : {performance_stats.get('annual_vol', 0):>8.2%}")

        # 风险调整指标 (最重要)
        print(f"\n⚖️  风险调整指标:")
        sharpe = performance_stats.get('sharpe', 0)
        sortino = performance_stats.get('sortino', 0)
        calmar = performance_stats.get('calmar', 0)

        sharpe_emoji = "🟢" if sharpe > 1 else "🟡" if sharpe > 0.5 else "🔴"
        sortino_emoji = "🟢" if sortino > 1.5 else "🟡" if sortino > 0.8 else "🔴"
        calmar_emoji = "🟢" if calmar > 2 else "🟡" if calmar > 1 else "🔴"

        print(f"   夏普比率          : {sharpe:>8.3f} {sharpe_emoji}")
        print(f"   Sortino比率       : {sortino:>8.3f} {sortino_emoji}")
        print(f"   Calmar比率        : {calmar:>8.3f} {calmar_emoji}")

        # 基准比较
        print(f"\n📊 基准比较 (vs 沪深300年化8%):")
        alpha = performance_stats.get('alpha', 0)
        info_ratio = performance_stats.get('info_ratio', 0)
        alpha_emoji = "🟢" if alpha > 0 else "🔴"
        info_emoji = "🟢" if info_ratio > 0.5 else "🟡" if info_ratio > 0 else "🔴"

        print(f"   超额收益(Alpha)   : {alpha:>8.2%} {alpha_emoji}")
        print(f"   信息比率          : {info_ratio:>8.3f} {info_emoji}")
        print(f"   跟踪误差          : {performance_stats.get('tracking_error', 0):>8.2%}")

        # 回撤分析
        print(f"\n📉 回撤风险:")
        max_dd = performance_stats.get('max_drawdown', 0)
        dd_duration = performance_stats.get('max_dd_duration', 0)
        dd_emoji = "🟢" if max_dd > -0.1 else "🟡" if max_dd > -0.2 else "🔴"

        print(f"   最大回撤          : {max_dd:>8.2%} {dd_emoji}")
        print(f"   回撤持续天数      : {dd_duration:>8.0f} 天")

        # 胜负统计
        print(f"\n🎯 胜负统计:")
        win_rate = performance_stats.get('win_rate', 0)
        monthly_win_rate = performance_stats.get('monthly_win_rate', 0)
        profit_factor = performance_stats.get('profit_factor', 0)

        win_emoji = "🟢" if win_rate > 0.55 else "🟡" if win_rate > 0.45 else "🔴"
        pf_emoji = "🟢" if profit_factor > 1.5 else "🟡" if profit_factor > 1.0 else "🔴"

        print(f"   日胜率            : {win_rate:>8.2%} {win_emoji}")
        print(f"   月胜率            : {monthly_win_rate:>8.2%}")
        print(f"   盈亏比            : {profit_factor:>8.2f} {pf_emoji}")

        # 尾部风险
        print(f"\n⚠️  尾部风险:")
        var_95 = performance_stats.get('var_95', 0)
        cvar_95 = performance_stats.get('cvar_95', 0)
        var_emoji = "🟢" if var_95 > -0.03 else "🟡" if var_95 > -0.05 else "🔴"

        print(f"   VaR(95%)         : {var_95:>8.2%} {var_emoji}")
        print(f"   CVaR(95%)        : {cvar_95:>8.2%}")

        # 持仓分析
        if position_sizes:
            print(f"\n💼 持仓配置:")
            sorted_positions = sorted(position_sizes.items(), key=lambda x: x[1], reverse=True)

            for i, (stock_code, position) in enumerate(sorted_positions[:5]):  # 显示前5大持仓
                stock_name = self.get_stock_name(stock_code)
                weight = (position / total_position) * 100
                risk_score = self.risk_metrics.get(stock_code, {}).get('risk_score', 0) if hasattr(self, 'risk_metrics') else 0
                risk_emoji = "🟢" if risk_score < 30 else "🟡" if risk_score < 60 else "🔴"

                print(f"   #{i+1} {stock_code} {stock_name[:6]:>6s}: {weight:>5.1f}% (¥{position:>7,.0f}) {risk_emoji}")

            if len(sorted_positions) > 5:
                print(f"   ... 还有 {len(sorted_positions)-5} 只股票")

        # 交易执行统计
        trading_stats = self.get_trading_statistics()
        if trading_stats['total_orders'] > 0:
            print(f"\n🔄 交易执行统计:")
            success_rate = trading_stats.get('success_rate', 0)
            fill_rate = trading_stats.get('avg_fill_ratio', 0)
            exec_emoji = "🟢" if success_rate > 0.9 else "🟡" if success_rate > 0.7 else "🔴"

            print(f"   总订单数          : {trading_stats['total_orders']:>8.0f}")
            print(f"   成交成功率        : {success_rate:>8.2%} {exec_emoji}")
            print(f"   平均成交比例      : {fill_rate:>8.2%}")
            print(f"   平均交易成本      : ¥{trading_stats.get('avg_transaction_cost', 0):>6.2f}")

        # 策略评级总结
        print(f"\n🏆 策略综合评级:")

        # 计算综合评分
        score_components = []
        if sharpe > 1.5: score_components.append(("收益质量", "优秀", "🟢"))
        elif sharpe > 1.0: score_components.append(("收益质量", "良好", "🟡"))
        else: score_components.append(("收益质量", "一般", "🔴"))

        if max_dd > -0.1: score_components.append(("风险控制", "优秀", "🟢"))
        elif max_dd > -0.2: score_components.append(("风险控制", "良好", "🟡"))
        else: score_components.append(("风险控制", "需改进", "🔴"))

        if win_rate > 0.55: score_components.append(("稳定性", "优秀", "🟢"))
        elif win_rate > 0.45: score_components.append(("稳定性", "良好", "🟡"))
        else: score_components.append(("稳定性", "一般", "🔴"))

        for component, rating, emoji in score_components:
            print(f"   {component:12s}: {rating:>6s} {emoji}")

        # 建议
        print(f"\n💡 策略建议:")
        suggestions = []

        if sharpe < 1.0:
            suggestions.append("• 考虑优化选股标准或调整仓位管理")
        if max_dd < -0.15:
            suggestions.append("• 加强回撤控制，可考虑降低单笔仓位或增加止损")
        if win_rate < 0.45:
            suggestions.append("• 检查入场时机，提高交易成功率")
        if alpha < 0:
            suggestions.append("• 策略未能跑赢基准，需要优化选股或择时逻辑")
        if not suggestions:
            suggestions.append("• 策略表现良好，可考虑适当增加仓位或扩大股票池")

        for suggestion in suggestions[:3]:  # 最多显示3条建议
            print(f"   {suggestion}")

        print("\n" + "="*100)
        print(f"📄 详细图表分析请查看: portfolio_analysis_enhanced.html")
        print("="*100 + "\n")

    def _calculate_realistic_stop_loss(self, current_price, atr, yesterday_close, stock_code=None, is_st=None):
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
        upper_limit, lower_limit = self._get_price_limits(yesterday_close, stock_code=stock_code, is_st=is_st)

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

            # 为了只打印原始（未复权）价格，需要同时取出 $factor 用于还原
            fields = ['$open', '$high', '$low', '$close', '$volume', '$factor']

            df = D.features(
                instruments=[qlib_code],
                fields=fields,
                start_time=start_date_qlib,
                end_time=end_date_qlib,
                freq='day',
                disk_cache=1  # 开启数据集缓存，显著提升I/O性能
            )

            if df is not None and not df.empty:
                # 处理多级索引，提取股票数据
                df = df.xs(qlib_code, level=0)

                # 规范列名（去掉$前缀）
                df.columns = [col.replace('$', '') for col in df.columns]

                # 确保数据类型正确
                df = df.astype(float)

                # === 使用 Qlib 的调整后价格进行回测；同时保留未复权价用于可视化 ===
                # Qlib 文档：$open/$close 等为“调整后价格”，可用 $factor 还原原始价（raw=adjusted/factor）。
                # 我们将：
                #  - 保留调整后列：open/high/low/close （用于计算收益与指标）
                #  - 额外添加 raw_close 列：用于可视化或对比
                if all(col in df.columns for col in ['open', 'high', 'low', 'close', 'factor']):
                    # 不再对 open/high/low/close 进行除以 factor 的还原，保持为“调整后价格”
                    df['raw_close'] = df['close'] / df['factor']
                    # 仍然保留 volume 与 factor，供上游过滤或诊断使用
                    # 下游指标函数均以调整后价格为基准（df['close'] 等）
                else:
                    print(f"警告：{stock_code} 缺少 factor 列，无法生成 raw_close（原始未复权价）")

                stock_name = self.get_stock_name(stock_code)
                return df
            else:
                stock_name = self.get_stock_name(stock_code)
                print(f"未获取到{stock_code} ({stock_name})的数据")
                return None

        except Exception as e:
            stock_name = self.get_stock_name(stock_code)
            print(f"获取{stock_code} ({stock_name})数据失败: {e}")
            return None

    def _process_single_stock(self, stock_code):
        """
        处理单只股票的数据获取和指标计算（用于并发处理）
        """
        try:
            stock_name = self.get_stock_name(stock_code)
            df = self.fetch_stock_data(stock_code)

            if df is not None and len(df) > 5:
                # 计算技术指标
                df = self.calculate_ma_signals(df)
                df = self.calculate_rsi(df)
                df = self.calculate_atr(df)
                df = self.calculate_volatility(df)
                df = self.calculate_max_drawdown(df)
                df = self.calculate_bollinger_bands(df)

                # 计算风险指标
                risk_score = self.calculate_risk_metrics(df, stock_code)

                # 返回结果
                if risk_score is not None and risk_score < 85:
                    return stock_code, df, risk_score, True
                else:
                    return stock_code, None, risk_score, False
            else:
                return stock_code, None, None, False

        except Exception as e:
            stock_name = self.get_stock_name(stock_code)
            print(f"处理{stock_code} ({stock_name})时出错: {e}")
            return stock_code, None, None, False

    def fetch_stocks_data_concurrent(self, max_workers=None):
        """
        并发获取所有股票数据并计算指标
        Parameters:
        -----------
        max_workers : int, optional
            最大并发数，默认为CPU核心数的75%
        """
        if max_workers is None:
            max_workers = max(1, int(mp.cpu_count() * 0.75))

        cpu_count = mp.cpu_count()
        print(f"📈 正在并发获取股票历史数据并计算风险指标...")
        print(f"🔧 系统信息: CPU核心数={cpu_count}, 使用并发线程数={max_workers}")
        print(f"📊 股票池规模: {len(self.stock_pool)} 只股票")

        # 估算处理时间
        estimated_time = len(self.stock_pool) * 0.5 / max_workers  # 假设每只股票0.5秒
        if estimated_time > 60:
            print(f"⏱️  预计处理时间: {estimated_time/60:.1f} 分钟")
        else:
            print(f"⏱️  预计处理时间: {estimated_time:.0f} 秒")

        successful_count = 0
        total_count = len(self.stock_pool)
        completed_count = 0

        # 优化并发策略：使用ThreadPoolExecutor处理I/O密集型任务（数据获取）
        # 对于I/O密集型的Qlib数据获取，线程池比进程池更合适
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_stock = {
                executor.submit(self._process_single_stock, stock): stock
                for stock in self.stock_pool
            }

            # 处理完成的任务
            for future in as_completed(future_to_stock):
                completed_count += 1
                original_stock = future_to_stock[future]

                try:
                    stock_code, df, risk_score, is_valid = future.result()
                    stock_name = self.get_stock_name(stock_code)

                    # 显示进度，包含风险评分信息
                    risk_info = f"风险评分={risk_score:.1f}" if risk_score is not None else "数据不足"
                    status = "✓通过" if is_valid else "✗过滤"
                    print(f"进度: {completed_count}/{total_count} - {stock_code} ({stock_name}) - {risk_info} - {status}")

                    if is_valid and df is not None:
                        norm_code = self._normalize_instrument(stock_code)
                        self.price_data[norm_code] = df
                        # 建立原始→规范化代码映射
                        self.code_alias[stock_code] = norm_code
                        successful_count += 1

                except Exception as e:
                    stock_name = self.get_stock_name(original_stock)
                    print(f"进度: {completed_count}/{total_count} - {original_stock} ({stock_name}) - 处理失败: {e}")

        efficiency = (successful_count / total_count * 100) if total_count > 0 else 0
        print(f"并发处理完成：成功获取{successful_count}/{total_count}只股票数据 (筛选通过率={efficiency:.1f}%)")

        # 在 fetch_stocks_data_concurrent 末尾这行之后：
        # print(f"并发处理完成：成功获取{successful_count}/{total_count}只股票数据 (筛选通过率={efficiency:.1f}%)")

        # 添加👇
        try:
            eq = self.backtest_equity_curve()
            if eq is not None and not eq.empty:
                print(f"回测完成：净值首末 = {float(eq.iloc[0]):.6f} → {float(eq.iloc[-1]):.6f}")
        except Exception as e:
            print(f"自动回测失败: {e}")

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


    def _get_calendar(self):
        """获取交易日历（优先使用 Qlib 提供的市场日历）。"""
        try:
            cal = D.calendar(
                start_time=self._convert_date_format(self.start_date),
                end_time=self._convert_date_format(self.end_date),
                freq="day",
            )
            return pd.DatetimeIndex(cal)
        except Exception:
            return None

    def build_price_panel(self, use_adjusted: bool = True) -> pd.DataFrame | None:
        """
        构建价格面板（列=股票，索引=交易日），使用日期并集并重建为交易日历索引。
        use_adjusted=True 使用调整后价格（close）；False 使用原始未复权价（raw_close）。
        """
        if not self.price_data:
            print("price_data 为空，尚未加载任何股票数据")
            return None
        col = 'close' if use_adjusted else 'raw_close'
        series = []
        for code, df in self.price_data.items():
            if col not in df.columns:
                # 如果选了 raw_close 但缺失，则跳过该标的
                if not use_adjusted:
                    continue
            s = df[col].rename(code)
            s.index = pd.to_datetime(s.index)
            series.append(s)
        if not series:
            print("无可用价格序列")
            return None
        # 替换 build_price_panel 里合并与 reindex 的那段
        prices = pd.concat(series, axis=1).sort_index()
        prices.index = pd.to_datetime(prices.index).normalize()  # 关键：索引只保留日期

        cal = self._get_calendar()
        if cal is not None and len(cal) > 0:
            cal = pd.DatetimeIndex(pd.to_datetime(cal)).normalize()  # 同样归一
            # 若同一日多条记录（数据补齐），以最后一条为准，再按日历并集重建索引
            prices = prices.groupby(prices.index).last().reindex(cal)

        return prices

    def build_multi_price_panels(self, use_adjusted: bool = True) -> dict:
        """
        构建多个价格面板（高开低收量）用于面板化技术指标计算

        Returns:
        --------
        dict: 包含 'high', 'low', 'close', 'open', 'volume' 的面板字典
        """
        if not self.price_data:
            print("price_data 为空，尚未加载任何股票数据")
            return {}

        price_cols = ['high', 'low', 'close', 'open', 'volume']
        if not use_adjusted:
            price_cols = ['high', 'low', 'raw_close', 'open', 'volume']  # raw_close替代close

        panels = {}

        for col in price_cols:
            series = []
            for code, df in self.price_data.items():
                if col in df.columns:
                    s = df[col].rename(code)
                    s.index = pd.to_datetime(s.index)
                    series.append(s)

            if series:
                panel = pd.concat(series, axis=1).sort_index()
                panel.index = pd.to_datetime(panel.index).normalize()

                # 使用交易日历对齐
                cal = self._get_calendar()
                if cal is not None and len(cal) > 0:
                    cal = pd.DatetimeIndex(pd.to_datetime(cal)).normalize()
                    panel = panel.groupby(panel.index).last().reindex(cal)

                # 统一key名称
                key = 'close' if col == 'raw_close' else col
                panels[key] = panel

        return panels

    def compute_indicators_panel(self, panels: dict, atr_period=14, vol_window=20, drawdown_window=60):
        """
        面板化计算技术指标，使用向量化和Numba加速

        Parameters:
        -----------
        panels : dict
            包含价格面板的字典
        atr_period : int
            ATR计算周期
        vol_window : int
            波动率计算窗口
        drawdown_window : int
            回撤计算窗口

        Returns:
        --------
        dict: 包含计算结果的面板字典
        """
        results = {}

        if 'high' not in panels or 'low' not in panels or 'close' not in panels:
            print("⚠️  缺少必要的价格面板，跳过指标计算")
            return results

        hi = panels['high']
        lo = panels['low']
        cl = panels['close']

        print(f"🔬 开始面板化技术指标计算...")

        # 1. ATR计算（面板化 + Numba加速）
        try:
            atr_panel, atr_pct_panel = self._compute_atr_panel_optimized(hi, lo, cl, atr_period)
            results['atr'] = atr_panel
            results['atr_pct'] = atr_pct_panel
            print(f"✅ ATR计算完成")
        except Exception as e:
            print(f"❌ ATR计算失败: {e}")

        # 2. 波动率计算（向量化）
        try:
            rets = cl.pct_change()
            vol_panel = rets.rolling(vol_window).std() * np.sqrt(252)  # 年化波动率
            results['volatility'] = vol_panel
            print(f"✅ 波动率计算完成")
        except Exception as e:
            print(f"❌ 波动率计算失败: {e}")

        # 3. 回撤计算（向量化）
        try:
            rolling_max = cl.rolling(drawdown_window, min_periods=1).max()
            drawdown_panel = (cl - rolling_max) / rolling_max
            max_drawdown_panel = drawdown_panel.rolling(drawdown_window, min_periods=1).min()
            results['drawdown'] = drawdown_panel
            results['max_drawdown'] = max_drawdown_panel
            print(f"✅ 回撤计算完成")
        except Exception as e:
            print(f"❌ 回撤计算失败: {e}")

        # 4. RSI计算（面板化）
        try:
            rsi_panel = self._compute_rsi_panel_optimized(cl, 14)
            results['rsi'] = rsi_panel
            print(f"✅ RSI计算完成")
        except Exception as e:
            print(f"❌ RSI计算失败: {e}")

        print(f"🎯 面板化指标计算完成，共计算 {len(results)} 个指标")
        return results

    def _compute_atr_panel_optimized(self, hi: pd.DataFrame, lo: pd.DataFrame, cl: pd.DataFrame, period=14):
        """面板化ATR计算，使用Numba加速（如果可用）"""
        # True Range 计算（完全向量化）
        tr1 = (hi - lo)
        tr2 = (hi - cl.shift(1)).abs()
        tr3 = (lo - cl.shift(1)).abs()

        # 使用numpy的maximum.reduce更高效
        tr_array = np.maximum.reduce([tr1.values, tr2.values, tr3.values])
        tr = pd.DataFrame(tr_array, index=cl.index, columns=cl.columns)

        # Wilder RMA计算
        if NUMBA_AVAILABLE:
            try:
                atr = tr.rolling(window=period, min_periods=period).apply(
                    self._wilder_rma_numba_wrapper,
                    args=(period,),
                    raw=True,
                    engine="numba",
                    engine_kwargs={"parallel": False, "nogil": True}
                )
            except Exception as e:
                print(f"⚠️  Numba ATR计算失败，回退到标准方法: {e}")
                atr = tr.ewm(alpha=1.0/period, adjust=False).mean()
        else:
            # 回退到pandas标准方法
            atr = tr.ewm(alpha=1.0/period, adjust=False).mean()

        # ATR百分比
        atr_pct = atr / cl * 100.0

        return atr, atr_pct

    def _wilder_rma_numba_wrapper(self, window_data, period):
        """Numba兼容的Wilder RMA计算wrapper"""
        if len(window_data) == 0:
            return np.nan

        alpha = 1.0 / period
        rma = window_data[0]

        for i in range(1, len(window_data)):
            rma = rma * (1 - alpha) + window_data[i] * alpha

        return rma

    def _compute_rsi_panel_optimized(self, cl: pd.DataFrame, period=14):
        """面板化RSI计算"""
        # 计算价格变化
        delta = cl.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))

        # 使用EWM计算平均收益和损失
        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

        # 计算RSI
        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def backtest_equity_curve(self, weights: pd.DataFrame | None = None, use_adjusted: bool = True, min_live_stocks: int = 3) -> pd.Series | None:
        """
        修复版回测组合净值，解决fix.md中指出的结构性问题：
          - 正确处理缺失值（保持NaN而非填充0）
          - 实现可交易性掩码（涨跌停/停牌过滤）
          - 动态起点选择（避免长期空仓=1）
          - A股T+1交易约束
          - 北交所30%涨跌幅处理
        """
        prices = self.build_price_panel(use_adjusted=use_adjusted)
        if prices is None or prices.empty:
            print("无法构建价格面板，回测中止")
            return None

        # 1. 构建有效性掩码（关键：保持NaN而非填充0）
        valid = prices.notna() & prices.shift(1).notna()

        # 2. 计算日收益（保持NaN）
        rets = (prices / prices.shift(1) - 1).where(valid)

        # 3. 构建可交易性掩码（涨跌停/停牌过滤）
        tradable_mask = self._build_tradable_mask(prices, valid)

        # 4. 对齐并准备权重
        if weights is None:
            # 当日可交易标的等权归一
            w = tradable_mask.astype(float)
            row_sum = w.sum(axis=1)
            # 只对有交易标的的日期归一化
            w = w.div(row_sum, axis=0).fillna(0.0)
        else:
            w = weights.reindex(rets.index).fillna(0.0)
            # 在可交易标的内重归一化
            w = w * tradable_mask.astype(float)
            rs = w.sum(axis=1)
            w = w.div(rs.where(rs > 0, 1.0), axis=0).fillna(0.0)

        # 5. A股T+1：权重次日生效
        if self.t_plus_1:
            w = w.shift(1).fillna(0.0)

        # 6. 找到首个活跃日（当日可交易标的数≥阈值）
        live_stocks_count = w.sum(axis=1)
        first_active_idx = (live_stocks_count >= min_live_stocks).idxmax()
        if not (live_stocks_count >= min_live_stocks).any():
            print(f"警告：没有找到可交易标的数≥{min_live_stocks}的交易日，使用默认起点")
            first_active_idx = w.index[0]
        else:
            print(f"回测起点自动对齐到首个活跃日: {first_active_idx}（可交易标的数≥{min_live_stocks}）")

        # 7. 从活跃日开始计算组合收益
        active_slice = slice(first_active_idx, None)
        w_active = w.loc[active_slice]
        rets_active = rets.loc[active_slice]

        # 8. 组合日收益（只在有效收益上聚合）
        port_ret = (w_active * rets_active).sum(axis=1, skipna=True)

        # 9. 交易成本（使用加权平均费率：买入+卖出各占50%）
        turnover = w_active.diff().abs().sum(axis=1).fillna(0.0)
        # 计算平均交易成本率（买卖各占一半）
        avg_buy_cost_rate = (self.commission_rate + self.transfer_fee_rate)
        avg_sell_cost_rate = (self.commission_rate + self.transfer_fee_rate + self.stamp_tax_rate)
        avg_transaction_cost_rate = (avg_buy_cost_rate + avg_sell_cost_rate) / 2
        port_ret_net = port_ret - turnover * avg_transaction_cost_rate

        # 10. 处理NaN：若当日无任何有效标的→延续前值而非强制0
        valid_ret_mask = port_ret_net.notna()
        if not valid_ret_mask.all():
            print(f"发现{(~valid_ret_mask).sum()}个无效收益日，将延续前值")
            port_ret_net = port_ret_net.ffill()

        # 11. 累计净值
        equity = (1.0 + port_ret_net.fillna(0.0)).cumprod()

        # 12. 诊断信息 - 增强异常数据检测
        nonzero_w_days = int((w_active.abs().sum(axis=1) > 1e-12).sum())
        nonzero_ret_days = int((rets_active.abs().sum(axis=1, skipna=True) > 1e-12).sum())

        # 检测异常收益率数据
        extreme_returns = port_ret_net.abs() > 0.2  # 日收益超过20%
        if extreme_returns.any():
            extreme_count = extreme_returns.sum()
            extreme_dates = port_ret_net[extreme_returns].index.tolist()[:5]  # 显示前5个
            print(f"🚨 警告：发现{extreme_count}个极端日收益(>20%)，前5个日期: {extreme_dates}")
            print(f"🚨 极端收益值: {port_ret_net[extreme_returns].head().tolist()}")

        # 检测收益率统计
        port_ret_stats = port_ret_net.describe()
        print(f"[数据质量] 组合日收益统计:")
        print(f"  均值: {port_ret_stats['mean']:.4f} (年化{port_ret_stats['mean']*252:.1%})")
        print(f"  标准差: {port_ret_stats['std']:.4f}")
        print(f"  最大: {port_ret_stats['max']:.4f}, 最小: {port_ret_stats['min']:.4f}")

        # 检测个股收益异常
        individual_extreme = (rets_active.abs() > 0.15).any(axis=1)  # 某天有个股收益>15%
        if individual_extreme.any():
            extreme_stock_days = individual_extreme.sum()
            print(f"⚠️  发现{extreme_stock_days}天存在个股极端收益(>15%)")

        print(f"[诊断] 活跃权重日={nonzero_w_days}, 有效收益日={nonzero_ret_days}, 回测周期={len(equity)}")
        print(f"[诊断] 净值区间: {equity.iloc[0]:.6f} → {equity.iloc[-1]:.6f} (总收益{((equity.iloc[-1]/equity.iloc[0])-1)*100:.1f}%)")

        # 暴露给外部
        self.daily_return = port_ret_net
        self.equity_curve = equity
        return equity

    def _compute_performance_stats(self, equity: pd.Series | None = None) -> dict:
        """基于回测结果计算全面绩效指标。若 equity 为空则使用 self.equity_curve/self.daily_return。"""
        if equity is None:
            equity = getattr(self, 'equity_curve', None)
        daily_ret = getattr(self, 'daily_return', None)
        if equity is None or daily_ret is None or equity.empty or daily_ret.empty:
            return {}

        # 基础收益指标 - 修正计算方式
        total_return = float((equity.iloc[-1] / equity.iloc[0]) - 1.0)
        ann_return = float((1.0 + daily_ret.mean()) ** 252 - 1.0)
        ann_vol = float(daily_ret.std() * np.sqrt(252)) if daily_ret.std() == daily_ret.std() else 0.0

        # 基准比较（使用沪深300作为基准）
        try:
            # 简化基准收益率估算（年化8%）
            benchmark_daily = 0.08 / 252
            excess_ret = daily_ret - benchmark_daily
            alpha = float(excess_ret.mean() * 252)
            tracking_error = float(excess_ret.std() * np.sqrt(252))
            info_ratio = alpha / tracking_error if tracking_error > 0 else 0.0
        except:
            alpha, tracking_error, info_ratio = 0.0, 0.0, 0.0

        # 风险调整指标
        rf_daily = 0.025 / 252
        excess = daily_ret - rf_daily
        sharpe = float((excess.mean() * 252) / (daily_ret.std() * np.sqrt(252))) if daily_ret.std() > 0 else 0.0

        # Sortino比率（下行标准差）
        downside_ret = daily_ret[daily_ret < 0]
        downside_std = float(downside_ret.std() * np.sqrt(252)) if len(downside_ret) > 0 else 0.0
        sortino = float((daily_ret.mean() - rf_daily) * 252 / downside_std) if downside_std > 0 else 0.0

        # 回撤分析 - 修正计算方式
        nav = equity.copy()
        peak = nav.cummax()
        dd = (nav / peak - 1.0)
        max_dd = float(dd.min()) if len(dd) > 0 else 0.0

        # 回撤持续时间
        dd_periods = (dd < -0.01)  # 回撤超过1%的时期
        if dd_periods.any():
            dd_duration = 0
            current_dd = 0
            max_dd_duration = 0
            for is_dd in dd_periods:
                if is_dd:
                    current_dd += 1
                    max_dd_duration = max(max_dd_duration, current_dd)
                else:
                    current_dd = 0
        else:
            max_dd_duration = 0

        # 胜负分析 - 修正计算方式
        wins = int((daily_ret > 0).sum())
        losses = int((daily_ret < 0).sum())
        total_trades = wins + losses
        win_rate = float(wins) / float(total_trades) if total_trades > 0 else 0.0
        avg_win = float(daily_ret[daily_ret > 0].mean()) if wins > 0 else 0.0
        avg_loss = float(abs(daily_ret[daily_ret < 0].mean())) if losses > 0 else 0.0
        profit_factor = (avg_win / avg_loss) if avg_loss > 0 else 0.0

        # 尾部风险
        var_95 = float(np.percentile(daily_ret, 5)) if len(daily_ret) > 0 else 0.0
        cvar_95 = float(daily_ret[daily_ret <= var_95].mean()) if len(daily_ret[daily_ret <= var_95]) > 0 else 0.0

        # 一致性指标 - 修正计算方式
        try:
            monthly_rets = daily_ret.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_wins = int((monthly_rets > 0).sum())
            monthly_total = len(monthly_rets)
            monthly_win_rate = float(monthly_wins) / float(monthly_total) if monthly_total > 0 else 0.0
        except Exception:
            monthly_win_rate = 0.0

        # Calmar比率 (年化收益/最大回撤)
        calmar = abs(ann_return / max_dd) if max_dd != 0 else 0.0

        return {
            # 基础收益指标
            'total_return': total_return,
            'annual_return': ann_return,
            'annual_vol': ann_vol,

            # 风险调整指标
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar,

            # 基准比较
            'alpha': alpha,
            'tracking_error': tracking_error,
            'info_ratio': info_ratio,

            # 回撤分析
            'max_drawdown': max_dd,
            'max_dd_duration': max_dd_duration,

            # 胜负分析
            'win_rate': win_rate,
            'monthly_win_rate': monthly_win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,

            # 尾部风险
            'var_95': var_95,
            'cvar_95': cvar_95,

            # 其他统计
            'total_days': len(daily_ret),
            'trading_days': len(daily_ret[daily_ret != 0]),
        }

    def run_rolling_backtest(self, top_k: int = 5, rebalance: str = 'M', skip_recent: int = 21, mom_window: int = 126, min_live_stocks: int = 3):
        """
        使用滚动动量+再平衡权重进行整段回测，自动应用回撤门控与 T+1。返回 (equity, stats)。
        """
        weights = self.build_rolling_weights(top_k=top_k, rebalance=rebalance, skip_recent=skip_recent, mom_window=mom_window)
        if weights is None or weights.empty:
            print("滚动权重生成失败：无可用价格或窗口不足")
            return None, {}

        # 回撤门控缩放
        weights = self.scale_weights_by_drawdown(weights)

        # 回测净值（内部已实现 T+1 与可交易掩码）
        equity = self.backtest_equity_curve(weights=weights, use_adjusted=True, min_live_stocks=min_live_stocks)
        if equity is None or equity.empty:
            print("回测失败：净值为空")
            return None, {}

        stats = self._compute_performance_stats(equity)
        print("="*80)
        print("                     策略全面绩效分析报告")
        print("="*80)

        # 基础收益指标
        print("\n📊 基础收益指标:")
        print(f"  总收益率           : {stats.get('total_return', 0):8.2%}")
        print(f"  年化收益率         : {stats.get('annual_return', 0):8.2%}")
        print(f"  年化波动率         : {stats.get('annual_vol', 0):8.2%}")
        print(f"  回测天数           : {stats.get('total_days', 0):8.0f} 天")
        print(f"  有效交易日         : {stats.get('trading_days', 0):8.0f} 天")

        # 风险调整指标
        print("\n⚖️  风险调整指标:")
        print(f"  夏普比率           : {stats.get('sharpe', 0):8.3f}")
        print(f"  Sortino比率        : {stats.get('sortino', 0):8.3f}")
        print(f"  Calmar比率         : {stats.get('calmar', 0):8.3f}")

        # 基准比较
        print("\n📈 基准比较(vs 沪深300):")
        print(f"  超额收益(Alpha)    : {stats.get('alpha', 0):8.2%}")
        print(f"  跟踪误差           : {stats.get('tracking_error', 0):8.2%}")
        print(f"  信息比率           : {stats.get('info_ratio', 0):8.3f}")

        # 回撤分析
        print("\n📉 回撤分析:")
        print(f"  最大回撤           : {stats.get('max_drawdown', 0):8.2%}")
        print(f"  最大回撤持续       : {stats.get('max_dd_duration', 0):8.0f} 天")

        # 胜负分析
        print("\n🎯 胜负分析:")
        print(f"  日胜率             : {stats.get('win_rate', 0):8.2%}")
        print(f"  月胜率             : {stats.get('monthly_win_rate', 0):8.2%}")
        print(f"  盈亏比             : {stats.get('profit_factor', 0):8.2f}")
        print(f"  平均盈利           : {stats.get('avg_win', 0):8.2%}")
        print(f"  平均亏损           : {stats.get('avg_loss', 0):8.2%}")

        # 尾部风险
        print("\n⚠️  尾部风险:")
        print(f"  VaR(95%)          : {stats.get('var_95', 0):8.2%}")
        print(f"  CVaR(95%)         : {stats.get('cvar_95', 0):8.2%}")

        print("="*80)
        return equity, stats

    def _build_tradable_mask(self, prices: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
        """
        构建可交易性掩码，处理涨跌停、停牌等不可交易情况（向量化优化版）

        Parameters:
        -----------
        prices : pd.DataFrame
            价格面板
        valid : pd.DataFrame
            基础有效性掩码

        Returns:
        --------
        pd.DataFrame
            可交易性掩码（True=可交易，False=不可交易）
        """
        # 基础掩码：必须有有效价格
        tradable = valid.copy()

        # 向量化的涨跌停掩码计算
        tradable_vectorized = self._build_tradable_mask_vectorized(prices, tradable)

        return tradable_vectorized.fillna(False)

    def _build_tradable_mask_vectorized(self, prices: pd.DataFrame, base_mask: pd.DataFrame) -> pd.DataFrame:
        """
        向量化构建可交易性掩码，避免逐股票循环（重大性能优化）

        Parameters:
        -----------
        prices : pd.DataFrame
            价格面板 [日期 x 股票代码]
        base_mask : pd.DataFrame
            基础有效性掩码

        Returns:
        --------
        pd.DataFrame
            可交易性掩码
        """
        cl = prices
        pc = cl.shift(1)  # 前一交易日收盘价

        # 1. 构建股票分类的布尔矩阵（向量化）
        columns = pd.Index(cl.columns)
        is_bj = columns.str.startswith("BJ")
        is_sh688 = columns.str.startswith("SH688")  # 科创板
        is_sz30 = columns.str.startswith("SZ30")    # 创业板
        is_ke = is_sh688 | is_sz30  # 科创+创业

        # ST股票向量化判断
        numeric_codes = columns.map(lambda c: c[2:] if len(c) > 6 and c[:2] in ('SH','SZ','BJ') else c)
        is_st = numeric_codes.map(lambda code: self._is_st_stock(code))

        # 2. 构建涨跌停限制百分比矩阵（向量化）
        # 优先级：北交所30% > 科创/创业20% > ST 5% > 主板10%
        limit_pct = np.where(is_bj, 0.30,
                      np.where(is_ke, 0.20,
                        np.where(is_st, 0.05, 0.10))).astype(float)

        # 3. 广播为完整的价格限制矩阵
        limit_pct_matrix = pd.DataFrame(
            np.broadcast_to(limit_pct, cl.shape),
            index=cl.index,
            columns=cl.columns
        )

        # 4. 计算涨跌停价格限制（完全向量化）
        upper_limit = pc * (1 + limit_pct_matrix)
        lower_limit = pc * (1 - limit_pct_matrix)

        # 5. 检测涨跌停触发（向量化比较）
        # 留出0.1%的容差避免浮点误差
        limit_tolerance = 0.001
        upper_hit = cl >= (upper_limit * (1 - limit_tolerance))
        lower_hit = cl <= (lower_limit * (1 + limit_tolerance))
        limit_hit = upper_hit | lower_hit

        # 6. 应用可交易性掩码
        tradable_mask = base_mask & (~limit_hit)

        # 添加调试信息
        if limit_hit.any().any():
            hit_count = limit_hit.sum().sum()
            total_observations = cl.notna().sum().sum()
            hit_rate = hit_count / total_observations * 100 if total_observations > 0 else 0
            print(f"🔍 发现涨跌停触发: {hit_count} 次 ({hit_rate:.2f}%)")

        return tradable_mask


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
                # 使用滚动窗口计算夏普比率（统一口径：超额收益）
                window_returns = rolling_returns.iloc[-min(available_length, len(rolling_returns)):]
                if len(window_returns) > 5 and window_returns.std() > 0:
                    # 统一使用2.5%无风险利率
                    daily_rf_rate = 0.025 / 252
                    excess_returns = window_returns - daily_rf_rate
                    sharpe_ratio = (excess_returns.mean() * 252) / (window_returns.std() * np.sqrt(252))
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

        metrics_obj = {
            'volatility': current_volatility,
            'current_drawdown': current_drawdown,
            'max_drawdown_60d': max_drawdown_60d,
            'atr_pct': atr_pct,
            'bb_width': bb_width,
            'sharpe_ratio': sharpe_ratio,
            'downside_deviation': downside_deviation,
            'risk_score': risk_score
        }
        norm_code = self._normalize_instrument(stock_code)
        self.risk_metrics[norm_code] = metrics_obj
        self.risk_metrics[stock_code] = metrics_obj

        return risk_score

    def calculate_position_size(self, stock_code, capital=100000):
        """
        基于风险的精确仓位计算（与ATR止损闭环）

        Parameters:
        -----------
        stock_code : str
            股票代码
        capital : float
            可用资金

        Returns:
        --------
        dict : 包含股数、仓位价值、风险指标等详细信息
        """
        if stock_code not in self.price_data:
            return None

        df = self.price_data[stock_code]
        if len(df) < 20:  # 确保有足够数据计算ATR
            return None

        # 获取当前价格和ATR
        current_price = df['close'].iloc[-1]
        atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else df['close'].rolling(14).std().iloc[-1]
        yesterday_close = df['close'].iloc[-2] if len(df) >= 2 else current_price

        # 计算理论止损位，考虑A股制度约束
        stop_loss_info = self._calculate_realistic_stop_loss(
            current_price, atr, yesterday_close, stock_code=stock_code
        )
        stop_distance = current_price - stop_loss_info['stop_price']

        # 基于risk_per_trade计算风险预算
        risk_amount = capital * self.risk_per_trade

        # 计算理论股数: shares = risk_amount / stop_distance
        if stop_distance <= 0:
            return None

        theoretical_shares = risk_amount / stop_distance

        # 调整为100股整数倍（A股交易单位）
        shares = int(theoretical_shares // 100) * 100
        if shares <= 0:
            shares = 100  # 最小单位

        # 计算实际仓位价值
        position_value = shares * current_price

        # 应用各种约束
        # 1. 单票最大比例约束（15%）
        max_single_position = capital * 0.15
        if position_value > max_single_position:
            shares = int(max_single_position / current_price // 100) * 100
            position_value = shares * current_price

        # 2. ADV流动性约束（单日成交不超过20日平均成交额的5%）
        if self._check_adv_constraint_for_sizing(stock_code, shares, current_price):
            shares = self._adjust_for_adv_constraint_sizing(stock_code, current_price)
            position_value = shares * current_price

        # 3. 行业/相关性约束（简化版，可后续扩展）
        # 这里可以加入与已持仓股票的相关性检查

        # 计算实际风险指标
        actual_risk = shares * stop_distance
        risk_utilization = actual_risk / risk_amount if risk_amount > 0 else 0

        return {
            'shares': shares,
            'position_value': position_value,
            'current_price': current_price,
            'stop_loss_price': stop_loss_info['stop_price'],
            'stop_distance': stop_distance,
            'risk_amount_budget': risk_amount,
            'actual_risk_amount': actual_risk,
            'risk_utilization': risk_utilization,
            'atr': atr,
            'position_pct': position_value / capital,
            'is_stop_limited': stop_loss_info.get('is_limited', False)
        }

    def _check_adv_constraint_for_sizing(self, stock_code, shares, price):
        """检查仓位计算时的ADV流动性约束"""
        if stock_code not in self.price_data:
            return False

        df = self.price_data[stock_code]
        if 'amount' in df.columns and len(df) >= 20:
            amount_20d = df['amount'].iloc[-20:].mean() * 10000  # 万元转元
            trade_value = shares * price
            # 检查是否超过ADV20的5%
            if trade_value > amount_20d * 0.05:
                return True
        return False

    def _adjust_for_adv_constraint_sizing(self, stock_code, price):
        """根据ADV约束调整仓位（用于仓位计算）"""
        if stock_code not in self.price_data:
            return 100

        df = self.price_data[stock_code]
        if 'amount' in df.columns and len(df) >= 20:
            amount_20d = df['amount'].iloc[-20:].mean() * 10000  # 万元转元
            max_trade_value = amount_20d * 0.05
            max_shares = int(max_trade_value / price // 100) * 100  # 调整为100股整数倍
            return max(100, max_shares)  # 至少100股
        return 100

    def _validate_amount_unit(self, stock_code=None, sample_size=5):
        """
        验证Qlib数据中amount字段的单位定义
        通过采样对比成交额和价格*成交量来推断单位

        Parameters:
        -----------
        stock_code : str, optional
            指定股票代码进行验证，None则随机选择
        sample_size : int
            验证样本数量

        Returns:
        --------
        dict : 包含单位推断结果和建议
        """
        print("正在验证amount字段单位定义...")

        # 选择验证样本
        if stock_code and stock_code in self.price_data:
            test_stocks = [stock_code]
        else:
            available_stocks = list(self.price_data.keys())
            test_stocks = random.sample(available_stocks, min(sample_size, len(available_stocks)))

        unit_results = []

        for stock in test_stocks:
            df = self.price_data[stock]
            if 'amount' in df.columns and 'volume' in df.columns and len(df) >= 10:
                # 取最近10天数据进行验证
                recent_data = df.iloc[-10:]

                for i, row in recent_data.iterrows():
                    price = row['close']
                    volume = row['volume']
                    amount = row['amount']

                    if price > 0 and volume > 0 and amount > 0:
                        # 理论成交额 = 价格 * 成交量
                        theoretical_amount = price * volume

                        # 计算比值来推断单位
                        ratio = amount / theoretical_amount

                        if 0.0001 <= ratio <= 0.001:  # amount单位为万元
                            unit_type = "万元"
                            multiplier = 10000
                        elif 0.9 <= ratio <= 1.1:  # amount单位为元
                            unit_type = "元"
                            multiplier = 1
                        elif 900 <= ratio <= 1100:  # amount单位为千元
                            unit_type = "千元"
                            multiplier = 1000
                        else:
                            unit_type = "未知"
                            multiplier = None

                        unit_results.append({
                            'stock': stock,
                            'date': i,
                            'ratio': ratio,
                            'unit_type': unit_type,
                            'multiplier': multiplier,
                            'price': price,
                            'volume': volume,
                            'amount': amount
                        })

        if not unit_results:
            return {'status': 'error', 'message': '无法获取足够的验证数据'}

        # 统计结果
        unit_counts = {}
        for result in unit_results:
            unit_type = result['unit_type']
            unit_counts[unit_type] = unit_counts.get(unit_type, 0) + 1

        # 确定最可能的单位
        most_likely_unit = max(unit_counts, key=unit_counts.get)
        confidence = unit_counts[most_likely_unit] / len(unit_results)

        # 获取对应的乘数
        if most_likely_unit == "万元":
            recommended_multiplier = 10000
        elif most_likely_unit == "千元":
            recommended_multiplier = 1000
        elif most_likely_unit == "元":
            recommended_multiplier = 1
        else:
            recommended_multiplier = 10000  # 默认按万元处理（保守）

        result = {
            'status': 'success',
            'most_likely_unit': most_likely_unit,
            'confidence': confidence,
            'recommended_multiplier': recommended_multiplier,
            'current_code_multiplier': 10000,  # 当前代码使用的乘数
            'unit_distribution': unit_counts,
            'sample_count': len(unit_results),
            'needs_adjustment': recommended_multiplier != 10000
        }

        print(f"验证结果：amount字段最可能的单位是 {most_likely_unit}（置信度：{confidence:.2%}）")
        if result['needs_adjustment']:
            print(f"⚠️ 建议调整乘数从 {result['current_code_multiplier']} 到 {recommended_multiplier}")
        else:
            print("✅ 当前代码中的单位处理是正确的")

        return result

    def run_consistency_test(self, test_runs=3, random_seed_base=42):
        """
        回测一致性测试：多次运行相同参数，验证结果一致性

        Parameters:
        -----------
        test_runs : int
            测试运行次数
        random_seed_base : int
            随机种子基数

        Returns:
        --------
        dict : 一致性测试结果
        """
        print(f"开始进行{test_runs}次回测一致性测试...")

        results = []

        for i in range(test_runs):
            print(f"执行第{i+1}次测试...")

            # 设置固定随机种子确保可重现性
            random.seed(random_seed_base + i)
            np.random.seed(random_seed_base + i)

            try:
                # 重新运行策略选股和回测
                selected_stocks = self.select_stocks()
                if not selected_stocks:
                    print(f"第{i+1}次测试：选股失败")
                    continue

                # 计算仓位（使用新的精确方法）
                position_info = {}
                for stock in selected_stocks:
                    pos_info = self.calculate_position_size(stock, capital=1000000)
                    if pos_info:
                        position_info[stock] = pos_info['position_value']

                if not position_info:
                    print(f"第{i+1}次测试：仓位计算失败")
                    continue

                # 执行回测
                equity_curve, performance_stats = self.backtest_with_risk_management(
                    selected_stocks, position_info, initial_capital=1000000
                )

                results.append({
                    'run': i + 1,
                    'selected_stocks': selected_stocks.copy(),
                    'position_info': position_info.copy(),
                    'final_return': performance_stats.get('total_return', 0),
                    'sharpe_ratio': performance_stats.get('sharpe_ratio', 0),
                    'max_drawdown': performance_stats.get('max_drawdown', 0),
                    'success': True
                })

            except Exception as e:
                print(f"第{i+1}次测试出现异常: {e}")
                results.append({
                    'run': i + 1,
                    'error': str(e),
                    'success': False
                })

        # 分析一致性
        successful_runs = [r for r in results if r.get('success', False)]

        if len(successful_runs) < 2:
            return {
                'status': 'failed',
                'message': f'成功运行次数不足: {len(successful_runs)}/{test_runs}',
                'results': results
            }

        # 检查选股一致性
        stock_consistency = True
        base_stocks = set(successful_runs[0]['selected_stocks'])
        for run in successful_runs[1:]:
            if set(run['selected_stocks']) != base_stocks:
                stock_consistency = False
                break

        # 检查收益率一致性（允许小幅差异）
        returns = [r['final_return'] for r in successful_runs]
        return_std = np.std(returns)
        return_consistency = return_std < 0.001  # 允许0.1%的差异

        consistency_result = {
            'status': 'success',
            'total_runs': test_runs,
            'successful_runs': len(successful_runs),
            'stock_consistency': stock_consistency,
            'return_consistency': return_consistency,
            'return_std': return_std,
            'avg_return': np.mean(returns),
            'results': results
        }

        print(f"一致性测试完成：")
        print(f"  成功运行: {len(successful_runs)}/{test_runs}")
        print(f"  选股一致性: {'✅' if stock_consistency else '❌'}")
        print(f"  收益一致性: {'✅' if return_consistency else '❌'} (标准差: {return_std:.4f})")

        return consistency_result

    def create_detailed_trading_log(self):
        """
        创建详细的交易日志记录器
        记录信号生成、约束检查、订单执行、成交回报等全流程
        """
        self.trading_log = {
            'signals': [],          # 信号记录
            'constraints': [],      # 约束检查记录
            'orders': [],          # 订单记录
            'executions': [],      # 执行记录
            'failures': [],        # 失败记录
            'daily_summary': {}    # 日度汇总
        }
        print("已初始化详细交易日志系统")

    def log_signal(self, stock_code, signal_type, signal_value, metadata=None):
        """记录交易信号"""
        if not hasattr(self, 'trading_log'):
            self.create_detailed_trading_log()

        self.trading_log['signals'].append({
            'timestamp': datetime.now().isoformat(),
            'stock_code': stock_code,
            'signal_type': signal_type,
            'signal_value': signal_value,
            'metadata': metadata or {}
        })

    def log_constraint_check(self, stock_code, constraint_type, passed, details=None):
        """记录约束检查结果"""
        if not hasattr(self, 'trading_log'):
            self.create_detailed_trading_log()

        self.trading_log['constraints'].append({
            'timestamp': datetime.now().isoformat(),
            'stock_code': stock_code,
            'constraint_type': constraint_type,
            'passed': passed,
            'details': details or {}
        })

    def log_order(self, stock_code, order_type, quantity, target_price, metadata=None):
        """记录订单信息"""
        if not hasattr(self, 'trading_log'):
            self.create_detailed_trading_log()

        order_id = f"{stock_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.trading_log['orders'].append({
            'order_id': order_id,
            'timestamp': datetime.now().isoformat(),
            'stock_code': stock_code,
            'order_type': order_type,
            'quantity': quantity,
            'target_price': target_price,
            'metadata': metadata or {}
        })
        return order_id

    def log_execution(self, order_id, executed_quantity, executed_price, slippage, success, reason=None):
        """记录执行结果"""
        if not hasattr(self, 'trading_log'):
            self.create_detailed_trading_log()

        self.trading_log['executions'].append({
            'timestamp': datetime.now().isoformat(),
            'order_id': order_id,
            'executed_quantity': executed_quantity,
            'executed_price': executed_price,
            'slippage': slippage,
            'success': success,
            'reason': reason
        })

    def export_trading_log(self, filepath=None):
        """导出交易日志到文件"""
        if not hasattr(self, 'trading_log'):
            return None

        if filepath is None:
            filepath = f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.trading_log, f, ensure_ascii=False, indent=2)

        print(f"交易日志已导出到: {filepath}")
        return filepath

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
        # A股交易成本构成（更新至2023-08-28印花税下调）：
        # 1. 印花税：卖出时收取0.05%，买入免收
        # 2. 券商佣金：双边收取，一般0.03%，最低5元
        # 3. 过户费：双边收取0.002%

        # 印花税（仅卖出，2023-08-28下调至0.05%）
        stamp_duty = 0
        if not is_buy:
            stamp_duty = trade_value * self.stamp_tax_rate

        # 券商佣金（双边）
        commission = max(trade_value * self.commission_rate, self.commission_min)

        # 过户费（双边）
        transfer_fee = trade_value * self.transfer_fee_rate

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
            norm_code = self._normalize_instrument(stock)
            if norm_code in self.price_data and self.price_data[norm_code] is not None:
                df = self.price_data[norm_code]

                # 确保有足够的历史数据（降低要求）
                min_required = min(30, max(momentum_windows) + skip_recent + 5)  # 最多要求30天
                available_data = len(df)
                metrics = self.risk_metrics.get(norm_code, self.risk_metrics.get(stock, {}))
                if available_data < 15 or not metrics:  # 最少15天
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
                    metrics = self.risk_metrics.get(norm_code, self.risk_metrics.get(stock, {}))
                    risk_adjustment = max(0.3, (100 - metrics.get('risk_score', 50)) / 100)  # 防止过度惩罚
                    sharpe_adjustment = max(0.5, min(1.5, metrics.get('sharpe_ratio', 0) + 1))

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

                    # 增加 norm_code 字段，便于后续对齐
                    rs_entry = {
                        'rs_score': adjusted_rs,
                        'raw_return': weighted_momentum,
                        'risk_score': metrics.get('risk_score', 50),
                        'volatility': metrics.get('volatility', 0.25),
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                        'trend_confirmation': trend_confirmation,
                        'momentum_3m': momentum_scores[0] if len(momentum_scores) > 0 else 0,
                        'momentum_6m': momentum_scores[1] if len(momentum_scores) > 1 else 0,
                        'momentum_12m': momentum_scores[2] if len(momentum_scores) > 2 else 0,
                        'norm_code': norm_code
                    }
                    rs_data[stock] = rs_entry

                except Exception as e:
                    # 静默跳过计算失败的股票
                    continue

        # 转换为DataFrame并排序，确保 norm_code 字段成为列
        self.rs_scores = pd.DataFrame.from_dict(rs_data, orient='index')
        self.rs_scores.index.name = 'stock_code'
        # 若 norm_code 字段缺失（如空df），补齐
        if not self.rs_scores.empty and 'norm_code' not in self.rs_scores.columns:
            self.rs_scores['norm_code'] = self.rs_scores.index.map(lambda x: self._normalize_instrument(x))
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
                disk_cache=1  # 开启数据集缓存，显著提升I/O性能
            )

            if market_df is None or market_df.empty:
                # 回退到本地 Qlib 失败时，使用 AkShare 获取指数数据
                market_df = self._fetch_sh_index_df()
                assert market_df is not None and not market_df.empty, "上证指数数据获取失败（Qlib 与 AkShare 均未返回数据）"
            else:
                # Qlib 返回的是 MultiIndex(index=[instrument, date])，只取 SH000001 这一条
                if isinstance(market_df.index, pd.MultiIndex):
                    market_df = market_df.xs('SH000001', level=0)
                    market_df.columns = [col.replace('$', '') for col in market_df.columns]
                else:
                    # 某些环境下可能直接返回单指数的普通索引，这里也统一去掉列名前缀
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

    def run_strategy(self, use_concurrent=True, max_workers=None, rolling_backtest: bool = False, rolling_top_k: int = 5, rolling_rebalance: str = 'M'):
        """
        运行完整策略（风险优化版）

        Parameters:
        -----------
        use_concurrent : bool, default True
            是否使用并发处理加速数据获取
        max_workers : int, optional
            最大并发数，默认为CPU核心数的75%
        """
        print("开始运行风险敏感型策略...")

        # 1. 检查市场状态
        market_regime = self.check_market_regime()
        print(f"当前市场状态: {market_regime}")

        # 2. 获取股票池
        if not self.stock_pool:
            self.get_stock_pool()

        # 3. 获取所有股票数据并计算指标
        if use_concurrent:
            self.fetch_stocks_data_concurrent(max_workers)
        else:
            # 原始顺序处理方式
            print("正在获取股票历史数据并计算风险指标...")
            for i, stock in enumerate(self.stock_pool):
                stock_name = self.get_stock_name(stock)
                print(f"进度: {i+1}/{len(self.stock_pool)} - {stock} ({stock_name})")
                df = self.fetch_stock_data(stock)
                if df is not None and len(df) > 5:
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
                    if risk_score is not None and risk_score < 85:
                        norm_code = self._normalize_instrument(stock)
                        self.price_data[norm_code] = df
                        self.code_alias[stock] = norm_code

            print(f"成功获取{len(self.price_data)}只股票数据（已过滤高风险）")
            if hasattr(self, 'filter_st') and self.filter_st:
                print("✓ ST股票已在股票池构建阶段预先剔除")
            else:
                print("✓ ST股票已保留（如需过滤请使用 --filter-st 选项）")

        # 4. 计算风险调整后的相对强度
        self.calculate_relative_strength()

        # 5. 选择股票（多重风险过滤）
        candidate_stocks = []

        # 首先通过技术指标过滤
        for _, row in self.rs_scores.head(20).iterrows():
            raw_code = row['stock_code']
            # 规范化代码优先使用norm_code列，否则自动规范化
            norm_code = row['norm_code'] if 'norm_code' in row and isinstance(row['norm_code'], str) and len(row['norm_code']) > 0 else self._normalize_instrument(raw_code)

            # 统一使用规范化代码访问内部数据结构
            df = self.price_data.get(norm_code)
            if df is None:
                continue

            # 风险指标既可能以规范化也可能以原始键入库，这里做双重回退
            metrics = self.risk_metrics.get(norm_code, self.risk_metrics.get(raw_code))
            if not isinstance(metrics, dict) or not metrics:
                continue

            # 多重过滤条件（与原逻辑一致）
            try:
                conditions = [
                    df['trend_signal'].iloc[-1] == 1,  # 趋势向上
                    df['RSI'].iloc[-1] < 75,           # RSI未严重超买（放宽到75）
                    df['RSI'].iloc[-1] > 25,           # RSI未严重超卖（放宽到25）
                    metrics.get('volatility', 1.0) < self.volatility_threshold * 1.2,  # 波动率限制放宽20%
                    metrics.get('max_drawdown_60d', 1.0) < self.max_drawdown_threshold * 1.3,  # 回撤限制放宽30%
                    df['trend_strength'].iloc[-1] > 0.5,  # 趋势强度要求降低
                ]
            except Exception:
                # 任一字段缺失则跳过该标的
                continue

            if all(conditions):
                # 将候选统一保存为规范化代码，便于后续与 self.price_data 等对齐
                candidate_stocks.append(norm_code)

        if len(candidate_stocks) == 0:
            print("无候选股票：可能原因→ 代码未规范化或过滤条件过严。已自动使用规范化代码对齐自检，建议检查 RSI/趋势/波动率阈值。")

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
            pos_info = self.calculate_position_size(stock)
            if pos_info:
                position_sizes[stock] = pos_info['position_value']
            else:
                position_sizes[stock] = 0

        # 根据市场状态调整仓位
        if market_regime == 'RISK_OFF':
            print("市场风险较高，降低整体仓位50%")
            position_sizes = {k: v * 0.5 for k, v in position_sizes.items()}
        elif market_regime == 'RISK_ON':
            print("市场风险较低，维持正常仓位")

        # 可选：使用滚动再平衡方案进行整段回测（避免前视），不依赖末日选股
        if rolling_backtest:
            print("启用滚动动量+再平衡回测……")
            equity, stats = self.run_rolling_backtest(top_k=min(rolling_top_k, max(1, len(self.price_data))), rebalance=rolling_rebalance)
            # 这里保留 selected_stocks/position_sizes 做展示；绩效以滚动方案为准

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

                code_pref = stock if stock[:2] in ('SH','SZ','BJ') else self._normalize_instrument(stock)
                numeric = code_pref[2:]
                is_st = self._is_st_stock(numeric)

                atr_stop_info = self._calculate_realistic_stop_loss(
                    current_price, atr, yesterday_close, is_st=is_st
                )
                atr_stop = atr_stop_info['stop_price']

                # 支撑位止损（20日低点）
                support_stop = df['low'].iloc[-20:].min() if len(df) >= 20 else current_price * 0.95

                # 移动止损（从最高点回撤8%）
                trailing_stop = df['close'].iloc[-20:].max() * 0.92 if len(df) >= 20 else current_price * 0.92

                # 涨跌停限制
                upper_limit, lower_limit = self._get_price_limits(yesterday_close, stock_code=None, is_st=is_st)

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
                    size=np.maximum(scatter_data['sharpe_ratio'] * 10 + 15, 5),  # 确保最小值为5
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

    def build_rolling_weights(self, top_k: int = 5, rebalance: str = 'M', skip_recent: int = 21, mom_window: int = 126) -> pd.DataFrame | None:
        """
        基于月度再平衡与历史动量（跳过近端）构建滚动权重矩阵，严格使用 t-1 及更早数据，避免前视。
        - top_k: 每次调仓选股数量
        - rebalance: 调仓频率（'W'、'M' 等 Pandas offset）
        - skip_recent: 跳过最近天数（防止短期反转）
        - mom_window: 动量评估窗口
        返回：index=交易日, columns=规范化代码 的权重矩阵（未应用 T+1）
        """
        prices = self.build_price_panel(use_adjusted=True)
        if prices is None or prices.empty:
            return None

        cal = prices.index
        # 每期的"最后一个交易日"为调仓日
        rebal_dates = pd.DatetimeIndex(pd.Series(cal).resample(rebalance).last().dropna())
        rebal_dates = rebal_dates[rebal_dates.isin(cal)]
        if len(rebal_dates) == 0:
            return None

        w = pd.DataFrame(0.0, index=cal, columns=prices.columns)

        for rd in rebal_dates:
            if rd not in cal:
                continue
            rd_pos = cal.get_loc(rd)
            eval_end_pos = rd_pos - skip_recent   # 评估截止点：跳过近端
            if isinstance(eval_end_pos, slice) or eval_end_pos <= 0:
                continue
            start_pos = eval_end_pos - mom_window
            if start_pos <= 0:
                continue

            # 仅用完整无缺失的列
            window = prices.iloc[start_pos:eval_end_pos]
            if window.empty:
                continue
            valid_cols = window.columns[window.notna().all().values]
            if len(valid_cols) == 0:
                continue

            p_begin = prices.iloc[start_pos]
            p_end = prices.iloc[eval_end_pos - 1]
            ret = (p_end[valid_cols] / p_begin[valid_cols] - 1.0).dropna()
            if ret.empty:
                continue

            picks = ret.sort_values(ascending=False).head(min(top_k, len(ret))).index.tolist()
            if not picks:
                continue

            weight = 1.0 / len(picks)
            w.loc[rd, picks] = weight

        # 调仓日之间前向填充
        w = w.replace(0.0, np.nan).ffill().fillna(0.0)
        return w

    def backtest_with_risk_management(self, selected_stocks, position_sizes, initial_capital=100000):
        """
        修复版带风险管理的回测，使用新的回测框架

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

        print(f"开始风险管理回测：{len(selected_stocks)}只股票，初始资金{initial_capital:,.0f}元")

        # 1. 构建权重矩阵（基于position_sizes）
        weights = self._build_weights_matrix(selected_stocks, position_sizes, initial_capital)
        if weights is None:
            return None

        # 基于指数回撤进行仓位门控缩放（risk_on 保持，risk_off 乘以 drawdown_risk_off_scale）
        weights = self.scale_weights_by_drawdown(weights)

        # 2. 使用修复版回测引擎
        equity_curve = self.backtest_equity_curve(weights=weights, use_adjusted=True, min_live_stocks=2)
        if equity_curve is None or equity_curve.empty:
            print("回测失败：无法生成净值曲线")
            return None

        # 3. 计算组合级绩效指标（统一口径）
        performance_stats = self._calculate_portfolio_performance(equity_curve)

        # 4. 生成回测报告
        self._generate_backtest_report(selected_stocks, position_sizes, equity_curve, performance_stats)

        return {
            'equity_curve': equity_curve,
            'performance_stats': performance_stats,
            'selected_stocks': selected_stocks,
            'position_sizes': position_sizes
        }

    def _build_weights_matrix(self, selected_stocks, position_sizes, initial_capital):
        """构建权重矩阵"""
        try:
            # 获取价格面板
            prices = self.build_price_panel(use_adjusted=True)
            if prices is None:
                return None

            # 过滤选中的股票
            available_stocks = [s for s in selected_stocks if s in prices.columns and s in self.price_data]
            if not available_stocks:
                print("错误：没有选中股票的价格数据")
                return None

            # 构建权重矩阵
            weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

            # 计算总仓位价值
            total_position_value = sum(position_sizes.get(s, 0) for s in available_stocks)
            if total_position_value <= 0:
                print("错误：总仓位价值为0")
                return None

            # 设置权重（仓位价值/总资金）
            for stock in available_stocks:
                weight = position_sizes.get(stock, 0) / initial_capital
                weights[stock] = weight

            print(f"权重矩阵构建完成：{len(available_stocks)}只股票，总权重{weights.sum(axis=1).max():.2%}")
            return weights

        except Exception as e:
            print(f"构建权重矩阵失败: {e}")
            return None

    def _calculate_portfolio_performance(self, equity_curve):
        """
        计算组合级绩效指标（统一口径）

        统一计算标准：
        - 夏普比率：日频超额均值 × √252 / 日频波动率（fix.md推荐）
        - 年化收益：几何年化（按净值序列复合）
        - 无风险利率：2.5%（当前中国1年期国债收益率）
        - 统一使用同一价格口径（复权价格）
        """
        if self.daily_return is None or len(self.daily_return) == 0:
            return {}

        returns = self.daily_return.dropna()
        if len(returns) == 0:
            return {}

        # 基础指标 - 修正：返回比例而非百分比
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)  # 不乘100，保持比例
        # 使用几何年化（复合收益）
        periods = len(returns)
        annual_return = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / periods) - 1)  # 不乘100
        volatility = returns.std() * np.sqrt(252)  # 不乘100，保持比例

        # 夏普比率（统一口径：日频超额均值 × √252 / 日频波动率）
        # 假设无风险利率为2.5%（当前中国1年期国债收益率）
        risk_free_rate = 0.025
        daily_rf_rate = risk_free_rate / 252
        excess_returns = returns - daily_rf_rate

        if returns.std() > 0:
            sharpe_ratio = (excess_returns.mean() * 252) / (returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0

        # 最大回撤 - 修正：保持比例格式
        cumulative = equity_curve
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()  # 不乘100，保持比例

        # 胜率和盈亏比（基于日度收益）- 修正：保持比例格式
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0  # 不乘100，保持比例
        profit_factor = positive_returns.sum() / abs(negative_returns.sum()) if len(negative_returns) > 0 and negative_returns.sum() < 0 else 0

        # 基准比较
        benchmark_daily = 0.08 / 252
        excess_ret = returns - benchmark_daily
        alpha = excess_ret.mean() * 252
        tracking_error = excess_ret.std() * np.sqrt(252)
        info_ratio = alpha / tracking_error if tracking_error > 0 else 0

        # Sortino比率和Calmar比率
        downside_ret = returns[returns < 0]
        downside_std = downside_ret.std() * np.sqrt(252) if len(downside_ret) > 0 else 0
        sortino = (returns.mean() - daily_rf_rate) * 252 / downside_std if downside_std > 0 else 0
        calmar = abs(annual_return / max_drawdown) if max_drawdown != 0 else 0

        # 尾部风险
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0

        # 月度胜率
        try:
            monthly_rets = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_win_rate = (monthly_rets > 0).mean() if len(monthly_rets) > 0 else 0
        except:
            monthly_win_rate = 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_vol': volatility,  # 改名匹配
            'sharpe': sharpe_ratio,    # 改名匹配
            'sortino': sortino,        # 新增
            'calmar': calmar,          # 新增
            'alpha': alpha,            # 新增
            'tracking_error': tracking_error,  # 新增
            'info_ratio': info_ratio,  # 新增
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'monthly_win_rate': monthly_win_rate,  # 新增
            'profit_factor': profit_factor,
            'var_95': var_95,          # 新增
            'cvar_95': cvar_95,        # 新增
            'max_dd_duration': 0,      # 简化处理
            'total_trades': len(returns),
            'periods': len(equity_curve)
        }

    def _generate_backtest_report(self, selected_stocks, position_sizes, equity_curve, performance_stats):
        """生成回测报告"""
        print("\n" + "="*50)
        print("风险管理回测报告")
        print("="*50)

        print(f"回测周期: {equity_curve.index[0].date()} 至 {equity_curve.index[-1].date()}")
        print(f"交易日数: {performance_stats.get('periods', 0)}")
        print(f"选中股票: {len(selected_stocks)}只")

        print("\n组合绩效指标 (统一口径):")
        print(f"  总收益率: {performance_stats.get('total_return', 0):.2f}%")
        print(f"  年化收益率: {performance_stats.get('annual_return', 0):.2f}%")
        print(f"  年化波动率: {performance_stats.get('volatility', 0):.2f}%")
        print(f"  夏普比率: {performance_stats.get('sharpe_ratio', 0):.3f}")
        print(f"  最大回撤: {performance_stats.get('max_drawdown', 0):.2f}%")
        print(f"  胜率: {performance_stats.get('win_rate', 0):.1f}%")
        print(f"  盈亏比: {performance_stats.get('profit_factor', 0):.2f}")

        print("\n仓位配置:")
        for stock, size in position_sizes.items():
            stock_name = self.get_stock_name(stock)
            print(f"  {stock} ({stock_name}): {size:,.0f}元")

        print("="*50)

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
    parser = argparse.ArgumentParser(description='A股趋势跟踪 + 相对强度策略 (风险优化版) - 含实盘交易引擎')

    # 运行模式
    parser.add_argument('--mode', choices=['analysis', 'trading'], default='analysis',
                       help='运行模式: analysis(策略分析), trading(每日交易引擎)')

    # 基本参数
    parser.add_argument('--start-date', '-s', default='20250101',
                       help='开始日期，格式YYYYMMDD (默认: 20250101)')
    parser.add_argument('--end-date', '-e', default=None,
                       help='结束日期，格式YYYYMMDD (默认: 今天)')
    parser.add_argument('--qlib-dir', default='~/.qlib/qlib_data/cn_data',
                       help='qlib数据目录路径 (默认: ~/.qlib/qlib_data/cn_data)')

    # 交易引擎专用参数
    parser.add_argument('--capital', type=float, default=1000000,
                       help='总资本金额（交易模式）(默认: 100万)')
    parser.add_argument('--max-positions', type=int, default=5,
                       help='最大持仓数量（交易模式）(默认: 5只)')
    parser.add_argument('--trade-date', default=None,
                       help='交易日期 (YYYYMMDD)，默认为今天')
    parser.add_argument('--current-holdings', default=None,
                       help='当前持仓JSON文件路径（交易模式）')

    # 股票池配置
    parser.add_argument('--pool-mode', choices=['auto', 'index', 'custom'], default='auto',
                       help='股票池模式: auto(自动从qlib获取所有可用股票), index(指数成分股), custom(自定义)')
    parser.add_argument('--index-code', default='000300',
                       help='指数代码，当pool-mode=index时使用 (默认: 000300沪深300)')
    parser.add_argument('--stocks', nargs='*',
                       help='自定义股票代码列表(6位格式)，当pool-mode=custom时使用，如: 000001 600000 300750')
    parser.add_argument('--max-stocks', type=int, default=200,
                       help='auto模式下的最大股票数量，设置为0表示不限制 (默认: 200)')

    # 性能选项
    parser.add_argument('--no-concurrent', action='store_true',
                       help='禁用并发处理，使用顺序处理(默认使用并发)')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='最大并发线程数(默认为CPU核心数的75%%)')

    # 过滤选项
    parser.add_argument('--filter-st', action='store_true',
                       help='过滤ST股票（基于本地st_stocks_akshare.json文件，在股票池构建初期就剔除）')

    # 输出选项
    parser.add_argument('--no-dashboard', action='store_true',
                       help='不生成风险仪表板HTML文件')
    parser.add_argument('--no-backtest', action='store_true',
                       help='不运行回测')

    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()

    if args.mode == 'trading':
        # 交易引擎模式
        print(f"\n=== 启动每日交易引擎 ===")
        print(f"运行模式: 交易引擎")
        print(f"总资本: ¥{args.capital:,.0f}")
        print(f"最大持仓: {args.max_positions}只")
        print(f"交易日期: {args.trade_date if args.trade_date else '今天'}")
        print(f"ST股票过滤: {'开启' if args.filter_st else '关闭'}")

        # 读取当前持仓
        current_holdings = {}
        if args.current_holdings and os.path.exists(args.current_holdings):
            import json
            with open(args.current_holdings, 'r', encoding='utf-8') as f:
                current_holdings = json.load(f)
                print(f"已读取持仓文件: {args.current_holdings}")

        # 运行交易引擎
        daily_plan, strategy = run_daily_trading_engine(
            start_date=args.start_date,
            end_date=args.end_date,
            max_stocks=args.max_stocks if args.max_stocks > 0 else 200,
            capital=args.capital,
            max_positions=args.max_positions,
            current_holdings=current_holdings,
            filter_st=args.filter_st
        )

        print(f"\n=== 交易引擎完成 ===")
        print(f"交易计划文件: {daily_plan['csv_path']}")
        print(f"风险利用率: {daily_plan['summary']['risk_utilization']:.1f}%")
        print(f"总投入资金: ¥{daily_plan['summary']['total_value']:,.0f}")

        # 生成执行提示
        print(f"\n=== 执行提示 ===")
        print("1. 收盘后: 已生成明日交易计划CSV文件")
        print("2. 盘前9:20-9:30: 核对前收与涨跌停价")
        print("3. 盘中: 按计划执行，注意风控触发")
        print("4. 收盘后: 记录实际成交，更新持仓文件")

        return daily_plan
    else:
        # 策略分析模式
        print(f"\n=== 策略分析模式 ===")

        # 处理自定义股票列表
        custom_stocks = args.stocks if args.pool_mode == 'custom' else None

        # 初始化风险敏感策略
        strategy = RiskSensitiveTrendStrategy(
            start_date=args.start_date,
            end_date=args.end_date,
            qlib_dir=args.qlib_dir,
            stock_pool_mode=args.pool_mode,
            custom_stocks=custom_stocks,
            index_code=args.index_code,
            filter_st=args.filter_st
        )

        # 设置股票数量限制（如果是auto模式且指定了max_stocks）
        if args.pool_mode == 'auto':
            if args.max_stocks > 0:
                strategy.max_stocks = args.max_stocks
                print(f"设置股票池最大数量限制: {args.max_stocks}")
            else:
                strategy.max_stocks = None
                print("不限制股票池数量")

        # 运行策略
        use_concurrent = not args.no_concurrent
        selected_stocks, position_sizes = strategy.run_strategy(
            use_concurrent=use_concurrent,
            max_workers=args.max_workers
        )

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
        backtest_result = strategy.backtest_with_risk_management(
            selected_stocks, position_sizes
        )

        if backtest_result is not None:
            print("\n回测结果（修复版风险管理回测）:")
            equity_curve = backtest_result['equity_curve']
            performance_stats = backtest_result['performance_stats']

            # 显示绩效统计 - 修正格式化，使用统一的百分比显示
            print(f"组合绩效指标（统一口径）:")
            print(f"  - 总收益率: {performance_stats.get('total_return', 0):.2%}")
            print(f"  - 年化收益率: {performance_stats.get('annual_return', 0):.2%}")
            print(f"  - 年化波动率: {performance_stats.get('annual_vol', performance_stats.get('volatility', 0)):.2%}")
            print(f"  - 夏普比率: {performance_stats.get('sharpe', performance_stats.get('sharpe_ratio', 0)):.3f}")
            print(f"  - 最大回撤: {performance_stats.get('max_drawdown', 0):.2%}")
            print(f"  - 胜率: {performance_stats.get('win_rate', 0):.2%}")
            print(f"  - 盈亏比: {performance_stats.get('profit_factor', 0):.2f}")

            # 绘制组合净值曲线
            fig_portfolio = go.Figure()
            fig_portfolio.add_trace(go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
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

            # 生成增强版的组合分析报告
            enhanced_fig = strategy.create_enhanced_portfolio_dashboard(equity_curve, performance_stats, selected_stocks, position_sizes)
            enhanced_fig.write_html("portfolio_analysis_enhanced.html")
            print("增强版组合分析报告已保存为 portfolio_analysis_enhanced.html")

            # 打印增强版关键指标摘要
            strategy.print_enhanced_metrics_summary(equity_curve, performance_stats, selected_stocks, position_sizes)
    else:
        print("没有符合风险条件的股票")


class DailyTradingPlan:
    """每日交易计划生成器 - 实盘信号&风控引擎"""

    def __init__(self, strategy_instance):
        self.strategy = strategy_instance
        self.trade_date = datetime.now().strftime('%Y%m%d')
        self.max_position_pct = 0.05  # 单笔交易不超过ADV20的5%

    def set_random_seed(self, trade_date=None):
        """基于交易日期设置固定随机种子，确保结果可复现"""
        if trade_date:
            self.trade_date = trade_date

        # 将交易日期转换为数字种子
        seed = int(self.trade_date) % 2147483647  # 限制在int32范围内
        random.seed(seed)
        np.random.seed(seed)
        print(f"已设置随机种子: {seed} (基于交易日期: {self.trade_date})")

    def calculate_precise_position_size(self, stock_code, capital, current_holdings=None):
        """
        精确的风险法仓位计算 - 基于ATR止损和risk_per_trade

        Parameters:
        -----------
        stock_code : str
            股票代码
        capital : float
            总资本
        current_holdings : dict, optional
            当前持仓，格式: {stock_code: shares}
        """
        if stock_code not in self.strategy.price_data:
            return None

        df = self.strategy.price_data[stock_code]
        current_price = df['close'].iloc[-1]
        atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else current_price * 0.02

        # 计算ATR止损价
        stop_loss_price = current_price - (atr * self.strategy.atr_multiplier)

        # 风险金额 = 总资本 × 每笔风险比例
        risk_amount = capital * self.strategy.risk_per_trade

        # 止损距离
        stop_distance = current_price - stop_loss_price

        if stop_distance <= 0:
            return None

        # 理论股数 = 风险金额 / 止损距离
        theoretical_shares = risk_amount / stop_distance

        # 调整为100股的整数倍（A股最小交易单位）
        shares = int(theoretical_shares // 100) * 100

        if shares <= 0:
            return None

        # 计算实际投入金额
        position_value = shares * current_price

        # ADV流动性约束检查
        if self._check_adv_constraint(stock_code, shares, current_price):
            shares = self._adjust_for_adv_constraint(stock_code, current_price)
            position_value = shares * current_price

        # 考虑交易成本
        total_cost = self.strategy._calculate_transaction_costs(position_value, is_buy=True)

        # 实际风险占用
        actual_risk = shares * stop_distance
        risk_utilization = actual_risk / risk_amount if risk_amount > 0 else 0

        return {
            'shares': shares,
            'position_value': position_value,
            'entry_price': current_price,
            'stop_loss': stop_loss_price,
            'atr': atr,
            'risk_amount': actual_risk,
            'risk_utilization': risk_utilization,
            'transaction_cost': total_cost['total_cost'],
            'cost_rate': total_cost['cost_rate']
        }

    def _check_adv_constraint(self, stock_code, shares, price):
        """检查是否违反ADV流动性约束"""
        if stock_code not in self.strategy.price_data:
            return False

        df = self.strategy.price_data[stock_code]

        # 计算过去20日平均成交额（单位：元）
        if 'amount' in df.columns and len(df) >= 20:
            amount_20d = df['amount'].iloc[-20:].mean() * 10000  # 万元转元
            trade_value = shares * price

            # 检查是否超过ADV20的5%
            if trade_value > amount_20d * self.max_position_pct:
                return True

        return False

    def _adjust_for_adv_constraint(self, stock_code, price):
        """根据ADV约束调整仓位"""
        df = self.strategy.price_data[stock_code]

        if 'amount' in df.columns and len(df) >= 20:
            amount_20d = df['amount'].iloc[-20:].mean() * 10000  # 万元转元
            max_trade_value = amount_20d * self.max_position_pct
            max_shares = int(max_trade_value / price // 100) * 100  # 调整为100股整数倍
            return max(100, max_shares)  # 至少100股

        return 100  # 默认最小单位

    def check_price_limit_risk(self, stock_code, target_price, is_buy=True):
        """检查涨跌停风险"""
        if stock_code not in self.strategy.price_data:
            return "数据不足"

        df = self.strategy.price_data[stock_code]
        yesterday_close = df['close'].iloc[-1]  # 最新收盘价作为昨收

        # 判断股票类型
        is_st = self.strategy._is_st_stock(stock_code)

        # 获取涨跌停价格
        upper_limit, lower_limit = self.strategy._get_price_limits(
            yesterday_close, stock_code, is_st
        )

        if is_buy:
            if target_price >= upper_limit * 0.995:  # 接近涨停
                return "涨停风险"
            elif target_price >= upper_limit * 0.98:  # 接近涨停
                return "接近涨停"
        else:
            if target_price <= lower_limit * 1.005:  # 接近跌停
                return "跌停风险"
            elif target_price <= lower_limit * 1.02:  # 接近跌停
                return "接近跌停"

        return "正常"

    def generate_buy_signals(self, capital=1000000, max_positions=5):
        """生成买入信号清单"""
        buy_list = []

        if not hasattr(self.strategy, 'rs_scores') or self.strategy.rs_scores.empty:
            print("未找到相对强度评分数据，请先运行策略")
            return buy_list

        # 选择候选股票
        candidates = []
        for _, row in self.strategy.rs_scores.head(20).iterrows():
            stock = row['stock_code']
            if stock in self.strategy.price_data:
                df = self.strategy.price_data[stock]
                metrics = self.strategy.risk_metrics.get(stock, {})

                # 技术条件过滤
                if (len(df) > 0 and
                    'trend_signal' in df.columns and
                    df['trend_signal'].iloc[-1] == 1 and  # 趋势向上
                    'RSI' in df.columns and
                    25 < df['RSI'].iloc[-1] < 75 and      # RSI合理区间
                    metrics.get('volatility', 1) < self.strategy.volatility_threshold):
                    candidates.append(stock)

        # 相关性过滤
        if len(candidates) > 1:
            candidates = self.strategy._filter_by_correlation(candidates)

        # 生成买入计划
        for stock in candidates[:max_positions]:
            position_info = self.calculate_precise_position_size(stock, capital)
            if position_info is None:
                continue

            df = self.strategy.price_data[stock]
            current_price = df['close'].iloc[-1]

            # 建议执行价格（开盘价或VWAP）
            entry_hint = "开盘价"  # 简化为开盘价，实际可加入VWAP逻辑

            # 检查涨跌停风险
            limit_risk = self.check_price_limit_risk(stock, current_price, is_buy=True)

            # 流动性风险标记
            adv_risk = "流动性风险" if self._check_adv_constraint(
                stock, position_info['shares'], current_price) else ""

            notes = [risk for risk in [limit_risk, adv_risk] if risk and risk != "正常"]

            buy_list.append({
                'date': self.trade_date,
                'code': stock,
                'name': self.strategy.get_stock_name(stock),
                'signal': f"RS_{self.strategy.rs_scores[self.strategy.rs_scores['stock_code']==stock]['rs_score'].iloc[0]:.1f}",
                'plan_action': 'buy',
                'plan_shares': position_info['shares'],
                'plan_weight': position_info['position_value'] / capital * 100,
                'entry_hint': entry_hint,
                'stop_loss': position_info['stop_loss'],
                'atr': position_info['atr'],
                'risk_used': position_info['risk_amount'],
                'notes': '; '.join(notes) if notes else '正常'
            })

        return buy_list

    def generate_watchlist(self, threshold_ratio=0.8):
        """生成观察清单 - 接近信号阈值但未通过筛选的股票"""
        watchlist = []

        if not hasattr(self.strategy, 'rs_scores') or self.strategy.rs_scores.empty:
            return watchlist

        # 找到买入信号的阈值
        buy_candidates = set()
        for _, row in self.strategy.rs_scores.head(10).iterrows():
            stock = row['stock_code']
            if stock in self.strategy.price_data:
                df = self.strategy.price_data[stock]
                if ('trend_signal' in df.columns and
                    df['trend_signal'].iloc[-1] == 1):
                    buy_candidates.add(stock)

        min_buy_score = min([self.strategy.rs_scores[
            self.strategy.rs_scores['stock_code']==stock]['rs_score'].iloc[0]
            for stock in buy_candidates]) if buy_candidates else 0

        watch_threshold = min_buy_score * threshold_ratio

        # 寻找接近阈值的股票
        for _, row in self.strategy.rs_scores.iterrows():
            stock = row['stock_code']
            rs_score = row['rs_score']

            if (stock not in buy_candidates and
                stock in self.strategy.price_data and
                rs_score >= watch_threshold):

                df = self.strategy.price_data[stock]
                current_price = df['close'].iloc[-1]

                # 分析接近突破的原因
                reasons = []
                if 'trend_signal' in df.columns:
                    if df['trend_signal'].iloc[-1] == 0:
                        reasons.append("趋势中性")
                    elif df['trend_signal'].iloc[-1] == -1:
                        reasons.append("趋势向下")

                if 'RSI' in df.columns:
                    rsi = df['RSI'].iloc[-1]
                    if rsi >= 75:
                        reasons.append("RSI超买")
                    elif rsi <= 25:
                        reasons.append("RSI超卖")

                metrics = self.strategy.risk_metrics.get(stock, {})
                if metrics.get('volatility', 0) > self.strategy.volatility_threshold:
                    reasons.append("波动率过高")

                watchlist.append({
                    'date': self.trade_date,
                    'code': stock,
                    'name': self.strategy.get_stock_name(stock),
                    'rs_score': rs_score,
                    'current_price': current_price,
                    'watch_reason': '; '.join(reasons) if reasons else '接近信号阈值',
                    'distance_to_signal': min_buy_score - rs_score
                })

        return sorted(watchlist, key=lambda x: x['rs_score'], reverse=True)[:10]

    def generate_risk_control_signals(self, current_holdings):
        """生成风控信号 - 减仓/清仓清单"""
        reduce_list = []

        for stock, shares in current_holdings.items():
            if stock not in self.strategy.price_data:
                continue

            df = self.strategy.price_data[stock]
            current_price = df['close'].iloc[-1]
            position_value = shares * current_price

            risk_flags = []

            # ATR止损检查
            if 'ATR' in df.columns and len(df) > 1:
                atr = df['ATR'].iloc[-1]
                stop_loss = current_price - (atr * self.strategy.atr_multiplier)

                # 假设持仓成本为前20日均价（简化处理）
                avg_cost = df['close'].iloc[-20:].mean() if len(df) >= 20 else current_price

                if current_price <= stop_loss:
                    risk_flags.append("ATR止损触发")

            # 最大回撤检查
            metrics = self.strategy.risk_metrics.get(stock, {})
            if metrics.get('max_drawdown_60d', 0) > self.strategy.max_drawdown_threshold:
                risk_flags.append("最大回撤超限")

            # 波动率检查
            if metrics.get('volatility', 0) > self.strategy.volatility_threshold:
                risk_flags.append("波动率超阈值")

            # 趋势反转检查
            if 'trend_signal' in df.columns and df['trend_signal'].iloc[-1] == -1:
                risk_flags.append("趋势反转向下")

            if risk_flags:
                # 计算建议减仓比例
                reduce_ratio = 1.0  # 默认全部清仓
                if "波动率超阈值" in risk_flags and len(risk_flags) == 1:
                    reduce_ratio = 0.5  # 波动率问题只减一半

                reduce_shares = int(shares * reduce_ratio // 100) * 100

                reduce_list.append({
                    'date': self.trade_date,
                    'code': stock,
                    'name': self.strategy.get_stock_name(stock),
                    'signal': '; '.join(risk_flags),
                    'plan_action': 'exit' if reduce_ratio == 1.0 else 'reduce',
                    'current_shares': shares,
                    'reduce_shares': reduce_shares,
                    'current_price': current_price,
                    'position_value': position_value,
                    'notes': f"风险等级: {'高' if len(risk_flags) > 2 else '中' if len(risk_flags) > 1 else '低'}"
                })

        return reduce_list

    def export_daily_plan_csv(self, buy_signals, add_signals, reduce_signals, watchlist, filepath=None):
        """导出标准化交易计划CSV文件"""
        if filepath is None:
            filepath = f"daily_trading_plan_{self.trade_date}.csv"

        all_plans = []

        # 买入信号
        for signal in buy_signals:
            all_plans.append(signal)

        # 加仓信号（这里简化为空，实际可根据持仓添加）
        for signal in add_signals:
            all_plans.append(signal)

        # 减仓信号
        for signal in reduce_signals:
            plan = {
                'date': signal['date'],
                'code': signal['code'],
                'name': signal['name'],
                'signal': signal['signal'],
                'plan_action': signal['plan_action'],
                'plan_shares': signal.get('reduce_shares', 0),
                'plan_weight': 0,  # 减仓不计算权重
                'entry_hint': '市价',
                'stop_loss': 0,  # 减仓无止损
                'atr': 0,
                'risk_used': 0,
                'notes': signal['notes']
            }
            all_plans.append(plan)

        # 转换为DataFrame并保存
        if all_plans:
            df = pd.DataFrame(all_plans)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"交易计划已导出到: {filepath}")

        # 同时导出观察清单
        if watchlist:
            watch_filepath = f"watchlist_{self.trade_date}.csv"
            watch_df = pd.DataFrame(watchlist)
            watch_df.to_csv(watch_filepath, index=False, encoding='utf-8-sig')
            print(f"观察清单已导出到: {watch_filepath}")

        return filepath

    def generate_complete_daily_plan(self, capital=1000000, current_holdings=None, max_positions=5):
        """生成完整的每日交易计划"""
        print(f"\n=== 生成 {self.trade_date} 交易计划 ===")

        # 设置随机种子确保可复现
        self.set_random_seed(self.trade_date)

        current_holdings = current_holdings or {}

        # 1. 买入信号
        print("正在生成买入信号...")
        buy_signals = self.generate_buy_signals(capital, max_positions)
        print(f"生成 {len(buy_signals)} 个买入信号")

        # 2. 加仓信号（简化实现，实际需要基于持仓分析）
        add_signals = []  # 这里可以根据需要添加加仓逻辑

        # 3. 减仓/清仓信号
        print("正在生成风控信号...")
        reduce_signals = self.generate_risk_control_signals(current_holdings)
        print(f"生成 {len(reduce_signals)} 个风控信号")

        # 4. 观察清单
        print("正在生成观察清单...")
        watchlist = self.generate_watchlist()
        print(f"生成 {len(watchlist)} 只观察股票")

        # 5. 导出CSV文件
        csv_path = self.export_daily_plan_csv(
            buy_signals, add_signals, reduce_signals, watchlist
        )

        # 6. 打印计划摘要
        print(f"\n=== 交易计划摘要 ===")
        print(f"买入信号: {len(buy_signals)} 只")
        print(f"减仓信号: {len(reduce_signals)} 只")
        print(f"观察清单: {len(watchlist)} 只")

        total_risk = sum([signal['risk_used'] for signal in buy_signals])
        total_value = sum([signal['plan_shares'] * signal.get('entry_price', 0) for signal in buy_signals])

        print(f"计划投入资金: ¥{total_value:,.0f}")
        print(f"风险占用: ¥{total_risk:,.0f} ({total_risk/capital*100:.1f}%)")

        if buy_signals:
            print(f"\n买入清单:")
            for signal in buy_signals:
                print(f"  {signal['code']} - {signal['name']}: {signal['plan_shares']}股 (风险: ¥{signal['risk_used']:,.0f}) [{signal['notes']}]")

        if reduce_signals:
            print(f"\n风控清单:")
            for signal in reduce_signals:
                print(f"  {signal['code']} - {signal['name']}: {signal['plan_action']} {signal.get('reduce_shares', 0)}股 [{signal['signal']}]")

        return {
            'buy_signals': buy_signals,
            'add_signals': add_signals,
            'reduce_signals': reduce_signals,
            'watchlist': watchlist,
            'csv_path': csv_path,
            'summary': {
                'total_positions': len(buy_signals),
                'total_value': total_value,
                'total_risk': total_risk,
                'risk_utilization': total_risk / capital * 100
            }
        }


def run_daily_trading_engine(start_date='20230101', end_date=None, max_stocks=200,
                           capital=1000000, max_positions=5, current_holdings=None, filter_st=False):
    """运行每日交易引擎 - 一键生成交易计划"""
    print("=== 启动每日交易引擎 ===")

    # 1. 初始化策略
    strategy = RiskSensitiveTrendStrategy(
        start_date=start_date,
        end_date=end_date,
        stock_pool_mode='auto',
        filter_st=filter_st
    )
    strategy.max_stocks = max_stocks

    # 2. 运行策略获取数据
    print("正在运行策略分析...")
    selected_stocks, position_sizes = strategy.run_strategy(use_concurrent=True)

    # 3. 初始化交易计划生成器
    trading_plan = DailyTradingPlan(strategy)

    # 4. 生成完整交易计划
    daily_plan = trading_plan.generate_complete_daily_plan(
        capital=capital,
        current_holdings=current_holdings,
        max_positions=max_positions
    )

    return daily_plan, strategy


if __name__ == "__main__":
    main()
