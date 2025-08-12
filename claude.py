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
from functools import partial
import multiprocessing as mp
import random
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
        self.price_limit_pct = 0.10        # 沪深涨跌停幅度（10%）
        self.st_limit_pct = 0.05           # ST股涨跌停幅度（5%）
        self.bj_limit_pct = 0.30           # 北交所涨跌停幅度（30%）
        self.transaction_cost = 0.003      # 双边交易成本（0.3%）
        self.slippage_bps = 5              # 滑点（5个基点）

        # ST股票缓存
        self._st_stocks_cache = {}
        self._st_cache_date = None
        self._st_api_failed = False  # 标记API是否已失败，避免重复尝试
        
        # 流动性过滤参数
        self.min_adv_20d = 20_000_000      # 20日平均成交额阈值：2000万元
        self.max_suspend_days_60d = 10     # 60日内最大停牌天数
        
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
                disk_cache=0,
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

    def get_stock_name(self, stock_code: str) -> str:
        """使用akshare获取股票名称（兼容 SH/SZ/BJ 前缀与北证）"""
        code = str(stock_code).strip().upper()
        # 提取用于 AkShare 的纯 6 位代码
        numeric = code[2:] if len(code) > 6 and code[:2] in ("SH", "SZ", "BJ") else code

        # 1) 首选：东财个股信息接口（包含“股票简称/证券简称”）
        try:
            info = ak.stock_individual_info_em(symbol=numeric)
            if info is not None and not info.empty and {"item", "value"}.issubset(set(info.columns)):
                row = info.loc[info["item"].isin(["股票简称", "证券简称"])]
                if not row.empty:
                    name_val = str(row["value"].iloc[0]).strip()
                    if name_val:
                        return name_val
        except Exception:
            pass

        # 2) 回退：若是北交所代码，使用北证代码-简称映射
        try:
            if code.startswith("BJ"):
                bj_df = ak.stock_info_bj_name_code()
                if bj_df is not None and not bj_df.empty:
                    # 兼容不同版本的列名
                    cols = {c: c for c in bj_df.columns}
                    code_col = "证券代码" if "证券代码" in cols else ("代码" if "代码" in cols else list(cols)[0])
                    name_col = "证券简称" if "证券简称" in cols else ("名称" if "名称" in cols else list(cols)[1])
                    hit = bj_df[bj_df[code_col].astype(str).str.endswith(numeric)]
                    if not hit.empty:
                        return str(hit.iloc[0][name_col]).strip()
        except Exception:
            pass

        # 3) 最后回退：全 A 股代码-简称映射（包含北证）
        try:
            all_df = ak.stock_info_a_code_name()
            if all_df is not None and not all_df.empty:
                cols = {c: c for c in all_df.columns}
                # 常见列名兼容
                code_candidates = [c for c in ["证券代码", "代码", "code", "股票代码"] if c in cols] or [list(cols)[0]]
                name_candidates = [c for c in ["证券简称", "名称", "name"] if c in cols] or [list(cols)[1]]
                code_col = code_candidates[0]
                name_col = name_candidates[0]

                # 去掉可能的交易所前缀后匹配
                series_code = all_df[code_col].astype(str).str.upper()
                series_code = (
                    series_code.str.replace("^SH", "", regex=True)
                               .str.replace("^SZ", "", regex=True)
                               .str.replace("^BJ", "", regex=True)
                )
                hit = all_df[series_code == numeric]
                if not hit.empty:
                    return str(hit.iloc[0][name_col]).strip()
        except Exception:
            pass

        # 兜底：返回原始代码
        return stock_code

    def _fetch_st_stocks_list(self) -> set:
        """
        获取当前ST/风险警示股票名单
        使用AkShare API而非字符串判断，增强错误处理
        """
        today = datetime.now().strftime('%Y-%m-%d')
        
        # 检查缓存
        if self._st_cache_date == today and self._st_stocks_cache:
            return self._st_stocks_cache
            
        # 如果API已标记为失败，直接返回空集合避免重复尝试
        if self._st_api_failed:
            return set()
            
        st_stocks = set()
        
        # 方法1：尝试获取风险警示板块股票（静默失败避免过多错误信息）
        try:
            import time
            time.sleep(0.1)  # 简单的API限流
            risk_warning_stocks = ak.stock_board_concept_cons_em(symbol="风险警示")
            if risk_warning_stocks is not None and not risk_warning_stocks.empty and '代码' in risk_warning_stocks.columns:
                codes = risk_warning_stocks['代码'].astype(str).str.zfill(6)
                if len(codes) > 0:
                    st_stocks.update(codes.tolist())
                    print(f"通过风险警示板块获取到{len(codes)}只ST股票")
        except Exception as e:
            # 静默处理，避免过多错误日志
            pass
            
        # 方法2：通过股票名称匹配ST（更鲁棒的实现）
        try:
            import time
            time.sleep(0.1)  # 简单的API限流
            all_stocks = ak.stock_info_a_code_name()
            if all_stocks is not None and not all_stocks.empty and '名称' in all_stocks.columns and '代码' in all_stocks.columns:
                # 查找名称包含ST的股票
                name_col = all_stocks['名称']
                st_mask = name_col.str.contains('ST|\\*ST|S\\*ST', na=False, regex=True)
                if st_mask.any():
                    st_names = all_stocks[st_mask]
                    codes = st_names['代码'].astype(str).str.zfill(6)
                    if len(codes) > 0:
                        new_st_count = len(codes)
                        st_stocks.update(codes.tolist())
                        print(f"通过名称匹配新增{new_st_count}只ST股票")
        except Exception as e:
            # 静默处理，避免过多错误日志
            pass
            
        # 如果两种方法都失败，标记API失败避免重复尝试
        if len(st_stocks) == 0:
            self._st_api_failed = True
            print("ST股票API获取失败，后续将使用保守策略（不区分ST股票）")
        else:
            print(f"成功识别{len(st_stocks)}只ST/风险警示股票")
            
        # 更新缓存
        self._st_stocks_cache = st_stocks
        self._st_cache_date = today
            
        return st_stocks
        
    def _is_st_stock(self, stock_code: str) -> bool:
        """
        判断是否为ST股票（带后备机制）
        
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
        
        # 首先尝试API方法
        st_stocks = self._fetch_st_stocks_list()
        if len(st_stocks) > 0:
            return numeric_code in st_stocks
        
        # 如果API失败，返回False（保守处理）
        # 在交易约束层面，将ST股票当作普通股票处理，虽然不够精确，
        # 但避免了API调用失败导致的程序中断
        return False

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

            # 批量过滤：检查数据可用性和基本质量
            filtered_stocks = []
            start_date_qlib = self._convert_date_format(self.start_date)
            end_date_qlib = self._convert_date_format(self.end_date)

            # 使用并发处理批量筛选
            batch_size = 20
            batches = [candidate_pool[i:i+batch_size] for i in range(0, len(candidate_pool), batch_size)]

            # 确定并发数
            max_workers = max(1, int(mp.cpu_count() * 0.5))  # 使用50%CPU核心，避免过载
            print(f"股票池筛选使用{max_workers}个并发进程处理{len(batches)}个批次")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
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

                        print(f"批次进度: {batch_count}/{len(batches)}, 已筛选: {len(filtered_stocks)}")
                    except Exception as e:
                        print(f"处理批次时出错: {e}")

            print(f"从{len(candidate_pool)}个候选股票中筛选出{len(filtered_stocks)}只合格股票")

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
                    
            # 3. 日均成交额过滤：ADV20要求
            if 'amount' in stock_data.columns and len(stock_data) >= 20:
                # amount是成交额，单位通常是万元，需要转换为元
                amount_20d = stock_data['amount'].iloc[-20:]
                # 假设amount单位是万元，转换为元进行比较
                avg_amount = amount_20d.mean() * 10000  # 万元转元
                if avg_amount < self.min_adv_20d:
                    return False

            # 去除价格异常股票
            if 'close' in stock_data.columns:
                recent_prices = stock_data['close'].iloc[-10:]
                if recent_prices.std() / recent_prices.mean() > 2:  # 价格波动过大
                    return False
                if recent_prices.iloc[-1] < 1:  # 股价过低
                    return False

            # 去除ST股票（使用API识别）
            if self._is_st_stock(stock_code):
                return False

            return True

        except Exception:
            return False

    def _get_price_limits(self, yesterday_close, stock_code=None, is_st=None):
        """
        计算涨跌停价格限制（增强版：自动识别股票类型）

        Parameters:
        -----------
        yesterday_close : float
            昨日收盘价
        stock_code : str, optional
            股票代码，用于自动判断类型
        is_st : bool, optional
            是否为ST股票，如果提供则直接使用
        """
        if is_st is None and stock_code is not None:
            is_st = self._is_st_stock(stock_code)
            
        # 判断股票类型并设置限价幅度
        if stock_code and stock_code.startswith('BJ'):
            limit_pct = self.bj_limit_pct  # 北交所30%
        elif is_st:
            limit_pct = self.st_limit_pct  # ST股5%
        elif stock_code and any(prefix in stock_code for prefix in ['68']):  # 科创板
            limit_pct = 0.20  # 科创板20%
        else:
            limit_pct = self.price_limit_pct  # 普通股10%
            
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
        upper_limit, lower_limit = self._get_price_limits(yesterday_close, stock_code=None, is_st=is_st)

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
        upper_limit, lower_limit = self._get_price_limits(yesterday_close, stock_code=None, is_st=is_st)

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
                disk_cache=0
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
        print(f"正在并发获取股票历史数据并计算风险指标...")
        print(f"系统信息: CPU核心数={cpu_count}, 使用并发线程数={max_workers}")

        successful_count = 0
        total_count = len(self.stock_pool)
        completed_count = 0

        # 使用ThreadPoolExecutor处理I/O密集型任务（Qlib数据获取主要是I/O操作）
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
                        self.price_data[stock_code] = df
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
        
        # 9. 交易成本
        turnover = w_active.diff().abs().sum(axis=1).fillna(0.0)
        port_ret_net = port_ret - turnover * self.transaction_cost
        
        # 10. 处理NaN：若当日无任何有效标的→延续前值而非强制0
        valid_ret_mask = port_ret_net.notna()
        if not valid_ret_mask.all():
            print(f"发现{(~valid_ret_mask).sum()}个无效收益日，将延续前值")
            port_ret_net = port_ret_net.ffill()
            
        # 11. 累计净值
        equity = (1.0 + port_ret_net.fillna(0.0)).cumprod()
        
        # 12. 诊断信息
        nonzero_w_days = int((w_active.abs().sum(axis=1) > 1e-12).sum())
        nonzero_ret_days = int((rets_active.abs().sum(axis=1, skipna=True) > 1e-12).sum())
        print(f"[诊断] 活跃权重日={nonzero_w_days}, 有效收益日={nonzero_ret_days}, 回测周期={len(equity)}")
        print(f"[诊断] 净值区间: {equity.iloc[0]:.6f} → {equity.iloc[-1]:.6f}")
        
        # 暴露给外部
        self.daily_return = port_ret_net
        self.equity_curve = equity
        return equity

    def _build_tradable_mask(self, prices: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
        """
        构建可交易性掩码，处理涨跌停、停牌等不可交易情况
        
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
        
        # 涨跌停掩码：检查是否触及价格限制
        prev_close = prices.shift(1)
        
        for stock in prices.columns:
            stock_prices = prices[stock]
            stock_prev = prev_close[stock]
            
            # 判断股票类型（北交所、ST股、普通股）
            # 提取股票代码（去掉SH/SZ前缀）
            stock_code = stock.replace('SH', '').replace('SZ', '') if len(stock) > 6 else stock
            is_st = self._is_st_stock(stock_code)
            
            if stock.startswith('BJ'):
                limit_pct = self.bj_limit_pct  # 北交所30%
            elif is_st:
                limit_pct = self.st_limit_pct  # ST股5%  
            else:
                limit_pct = self.price_limit_pct  # 普通股10%
                
            # 计算涨跌停价格
            upper_limit = stock_prev * (1 + limit_pct)
            lower_limit = stock_prev * (1 - limit_pct)
            
            # 触及涨跌停的不可交易（买不到/卖不出）
            # 注意：这里简化处理，实际中可能需要更精细的流动性判断
            limit_hit = (stock_prices >= upper_limit * 0.999) | (stock_prices <= lower_limit * 1.001)
            tradable[stock] = tradable[stock] & ~limit_hit
            
        # 成交量过滤：过滤流动性不足的标的
        # 这里简化处理，实际可以加入成交量/换手率判断
        
        return tradable.fillna(False)

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

    def run_strategy(self, use_concurrent=True, max_workers=None):
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
            
        # 基础指标
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
        # 使用几何年化（复合收益）
        periods = len(returns)
        annual_return = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / periods) - 1) * 100
        volatility = returns.std() * np.sqrt(252) * 100
        
        # 夏普比率（统一口径：日频超额均值 × √252 / 日频波动率）
        # 假设无风险利率为2.5%（当前中国1年期国债收益率）
        risk_free_rate = 0.025
        daily_rf_rate = risk_free_rate / 252
        excess_returns = returns - daily_rf_rate
        
        if returns.std() > 0:
            sharpe_ratio = (excess_returns.mean() * 252) / (returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        cumulative = equity_curve
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # 胜率和盈亏比（基于日度收益）
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        win_rate = len(positive_returns) / len(returns) * 100 if len(returns) > 0 else 0
        profit_factor = positive_returns.sum() / abs(negative_returns.sum()) if len(negative_returns) > 0 and negative_returns.sum() < 0 else float('inf')
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
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
                       help='auto模式下的最大股票数量，设置为0表示不限制 (默认: 200)')

    # 性能选项
    parser.add_argument('--no-concurrent', action='store_true',
                       help='禁用并发处理，使用顺序处理(默认使用并发)')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='最大并发线程数(默认为CPU核心数的75%%)')

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
            
            # 显示绩效统计
            print(f"组合绩效指标（统一口径）:")
            print(f"  - 总收益率: {performance_stats.get('total_return', 0):.2f}%")
            print(f"  - 年化收益率: {performance_stats.get('annual_return', 0):.2f}%")
            print(f"  - 年化波动率: {performance_stats.get('volatility', 0):.2f}%")
            print(f"  - 夏普比率: {performance_stats.get('sharpe_ratio', 0):.3f}")
            print(f"  - 最大回撤: {performance_stats.get('max_drawdown', 0):.2f}%")
            print(f"  - 胜率: {performance_stats.get('win_rate', 0):.1f}%")
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
    else:
        print("没有符合风险条件的股票")


if __name__ == "__main__":
    main()