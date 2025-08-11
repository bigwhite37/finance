# 导入所需的库
import argparse
import qlib
from qlib.data import D
import pandas as pd
import os


# 默认字段（兼容 cn_data）：
DEFAULT_FIELDS = ['$open', '$high', '$low', '$close', '$volume', '$amount', '$factor']

def _parse_fields_arg(fields_arg: str | None) -> list[str]:
    """Parse --fields into a list of Qlib expression fields.
    Notes:
      - If `fields_arg` is None/empty/'ALL' (case-insensitive), return DEFAULT_FIELDS.
      - Users SHOULD avoid $ in shell unless using single quotes; double quotes can expand $VAR.
      - We therefore accept names without '$' and will prefix automatically.
    """
    if not fields_arg or fields_arg.strip() == "" or fields_arg.strip().upper() == 'ALL':
        return DEFAULT_FIELDS.copy()
    raw = fields_arg.strip()
    # If shell expanded "$open" to empty, the string may become commas only. Guard it:
    if set(raw) <= {',', ' ', '\t'}:
        return DEFAULT_FIELDS.copy()
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    out: list[str] = []
    for p in parts:
        # remove any accidental leading '$' leftover markers but keep if present
        if not p.startswith('$'):
            p = '$' + p
        out.append(p)
    return out

# Helper: fetch a single field safely, return DataFrame or None if fails
def _safe_features_single(instruments, field: str, start_time, end_time, freq):
    """Try fetching a single field; return DataFrame or None if fails."""
    try:
        return D.features(
            instruments=instruments,
            fields=[field],
            start_time=start_time,
            end_time=end_time,
            freq=freq,
            disk_cache=0,
        )
    except Exception:
        return None

# Helper: compute vwap if possible from turnover/volume
def _compute_vwap_if_possible(df: pd.DataFrame) -> pd.DataFrame:
    # df has columns without '$'. If 'turnover' and 'volume' present, compute vwap = turnover/volume where volume>0
    if 'turnover' in df.columns and 'volume' in df.columns:
        v = df['volume']
        a = df['turnover']
        with pd.option_context('mode.use_inf_as_na', True):
            df['vwap'] = (a / v).where(v > 0)
    return df

# 工具函数：规范股票代码为 Qlib 标准格式，如 600000->SH600000, 000001->SZ000001
def _normalize_instrument(code: str) -> str:
    c = str(code).strip().upper()
    if len(c) == 6 and c.isdigit():
        if c[0] == '6':
            return 'SH' + c
        elif c[0] in ('0', '3'):
            return 'SZ' + c
    return c

# 仅初始化一次 Qlib
_QINIT_DONE = False

def _init_qlib_once(qlib_dir_expanded: str):
    global _QINIT_DONE
    if _QINIT_DONE:
        return
    qlib.init(provider_uri=qlib_dir_expanded, region="cn")
    print(f"Qlib 初始化成功，数据路径: {qlib_dir_expanded}")
    print("提示: --fields 建议写成 open,close,volume 这种不带$的形式；或使用单引号防止 $ 被 shell 展开。")
    _QINIT_DONE = True

def print_stock_details_for_day(stock_code: str, date: str, qlib_dir: str = "~/.qlib/qlib_data/cn_data"):
    """
    使用 Qlib API 读取并打印指定股票在特定日期的详细行情数据。

    此版本经过修正，以兼容可能存在解析器BUG的Qlib版本。

    参数:
    ----------
    stock_code : str
        要查询的股票代码，格式需遵循Qlib规范，例如 'SH600919' 或 'SZ000001'。
    date : str
        要查询的日期，格式为 'YYYY-MM-DD'。
    qlib_dir : str, optional
        本地Qlib数据的存储路径。默认为 '~/.qlib/qlib_data/cn_data'。
    """
    # 展开用户目录路径
    qlib_dir_expanded = os.path.expanduser(qlib_dir)

    # 检查数据目录是否存在
    if not os.path.exists(qlib_dir_expanded):
        print(f"错误：Qlib数据目录不存在于 '{qlib_dir_expanded}'")
        print("请检查 `qlib_dir` 参数是否正确，或者您是否已经下载了数据。")
        return

    # --- 步骤 1: 初始化 Qlib ---
    try:
        qlib.init(provider_uri=qlib_dir_expanded, region="cn")
        print(f"Qlib 初始化成功，数据路径: {qlib_dir_expanded}")
    except Exception as e:
        print(f"Qlib 初始化失败: {e}")
        return

    # 使用 Qlib 表达式字段（均为前复权/按因子调整后的字段）
    fields = ['$open', '$high', '$low', '$close', '$volume', '$amount', '$factor']

    # --- 步骤 3: 直接一次性获取所需字段（全部为表达式字段） ---
    try:
        data = D.features(
            instruments=[stock_code],
            start_time=date,
            end_time=date,
            fields=fields,
            disk_cache=0,
        )
    except Exception as e:
        print(f"\n从Qlib获取数据时出错: {e}")
        print("请检查股票代码和日期是否正确。")
        return

    # 规范列名：去掉 `$` 前缀，并把 amount 重命名为 turnover，随后计算未复权价格（raw = adjusted / factor）
    data = data.copy()
    data.columns = [c.replace('$', '') for c in data.columns]
    if 'amount' in data.columns:
        data = data.rename(columns={'amount': 'turnover'})
    if all(col in data.columns for col in ['open', 'high', 'low', 'close', 'factor']):
        data['open']  = data['open']  / data['factor']
        data['high']  = data['high']  / data['factor']
        data['low']   = data['low']   / data['factor']
        data['close'] = data['close'] / data['factor']
        # 只保留原始（未复权）价格列用于展示
        data = data[['open', 'high', 'low', 'close']]
    else:
        print("警告：缺少计算未复权价格所需的列（open/high/low/close/factor），将按现有列原样打印。")

    # --- 步骤 4: 检查数据并打印 ---
    if data.empty:
        print(f"\n未能找到股票 {stock_code} 在 {date} 的数据。")
        print("可能原因：该日期为非交易日（周末或节假日），或股票代码有误。")
    else:
        print(f"\n\033[1m--- 股票 {stock_code} 在 {date} 的详细数据 ---\033[0m")

        stock_day_details = data.loc[(stock_code, pd.Timestamp(date))]
        # 为了更美观且避免 dtype 警告，构造 object 类型的显示副本
        display_details = stock_day_details.copy()
        display_details = display_details.astype("object")
        if 'volume' in display_details and pd.notna(display_details['volume']):
            display_details['volume'] = "{:,.0f}".format(float(display_details['volume']))
        if 'turnover' in display_details and pd.notna(display_details['turnover']):
            display_details['turnover'] = "{:,.2f}".format(float(display_details['turnover']))
        print(display_details.to_string())

        print("\n\033[1m--- 字段含义说明 ---\033[0m")
        print("open, high, low, close:  \033[35m未复权\033[0m价格（依据 Qlib 规范：raw = adjusted / factor）。")


# 支持“多个股票 + 时间段”的详细行情获取并打印
def print_stocks_details_for_range(
    stocks,
    start_date: str,
    end_date: str,
    qlib_dir: str = "~/.qlib/qlib_data/cn_data",
    save_csv: str | None = None,
    freq: str = 'day',
    fields_arg: str | None = None,
    meta: bool = False
):
    """
    使用 Qlib API 读取并打印【多只股票】在【指定时间段】的详细行情数据。
    参数：
        stocks:  list[str]，股票代码列表，支持 600000/000001 或 SH600000/SZ000001
        start_date, end_date: 字符串 'YYYY-MM-DD'
        qlib_dir:  本地 Qlib 数据路径
        save_csv:  若提供文件路径，则将合并后的明细另存为 CSV
        freq: 数据频率，默认 'day'
        fields_arg: 字段参数（命令行 --fields），None 或 'ALL' 为默认字段
        meta: 是否打印元信息/数据健康
    输出：
        在控制台按股票分组打印全量日期行；如需保存，写入 CSV 文件。
    """
    qlib_dir_expanded = os.path.expanduser(qlib_dir)
    if not os.path.exists(qlib_dir_expanded):
        print(f"错误：Qlib数据目录不存在于 '{qlib_dir_expanded}'")
        return

    try:
        _init_qlib_once(qlib_dir_expanded)
    except Exception as e:
        print(f"Qlib 初始化失败: {e}")
        return

    # 规范化代码
    instruments = [_normalize_instrument(s) for s in stocks]

    requested_fields = _parse_fields_arg(fields_arg)

    # If user explicitly asked for $vwap, we will compute it from $amount/$volume to be robust across datasets.
    want_vwap = any(f.lstrip('$').lower() == 'vwap' for f in requested_fields)

    # Build a candidate fetch list: remove $vwap (will compute later), ensure $amount and $volume exist if VWAP is wanted.
    candidate_fields = [f for f in requested_fields if f.lstrip('$').lower() != 'vwap']
    if want_vwap:
        if '$amount' not in candidate_fields:
            candidate_fields.append('$amount')
        if '$volume' not in candidate_fields:
            candidate_fields.append('$volume')
    # Always ensure fields required for raw (unadjusted) prices are present
    for f in ['$open', '$high', '$low', '$close', '$factor']:
        if f not in candidate_fields:
            candidate_fields.append(f)

    # If user didn't pass any (e.g., shell expanded all $vars), fall back to DEFAULT_FIELDS
    if not candidate_fields:
        candidate_fields = DEFAULT_FIELDS.copy()

    # Probe each field individually to build a supported list (some datasets may not provide all columns)
    supported_fields: list[str] = []
    for f in candidate_fields:
        got = _safe_features_single(instruments, f, start_date, end_date, freq)
        if got is not None and not got.empty:
            supported_fields.append(f)
    if not supported_fields:
        print("未能获取到任何字段：请检查 --fields 的写法（尽量不要在 shell 中使用 $，或使用单引号包裹），以及本地数据是否包含这些列。")
        print("已尝试字段: " + ",".join(candidate_fields))
        return

    # 一次性拉取所有已验证支持的字段
    try:
        df = D.features(
            instruments=instruments,
            fields=supported_fields,
            start_time=start_date,
            end_time=end_date,
            freq=freq,
            disk_cache=0,
        )
    except Exception as e:
        print(f"\n从Qlib获取数据时出错: {e}")
        print("请检查股票代码与时间范围是否正确（交易日、代码前缀 SH/SZ）。")
        return

    if df.empty:
        print("未获取到任何数据；请确认时间区间内有交易日，且代码在本地数据集中可用。")
        return

    # 规范列名，并计算未复权价格（raw = adjusted / factor）
    df = df.copy()
    df.columns = [c.replace('$', '') for c in df.columns]
    if 'amount' in df.columns:
        df = df.rename(columns={'amount': 'turnover'})

    # Compute raw OHLC if possible
    if all(col in df.columns for col in ['open', 'high', 'low', 'close', 'factor']):
        df['open']  = df['open']  / df['factor']
        df['high']  = df['high']  / df['factor']
        df['low']   = df['low']   / df['factor']
        df['close'] = df['close'] / df['factor']
        # 只保留原始（未复权）价格列用于展示
        df = df[['open', 'high', 'low', 'close']]
    else:
        print("警告：缺少计算未复权价格所需的列（open/high/low/close/factor），将按现有列原样打印。")

    # （如果用户显式要求 vwap，这里不再展示；但保留计算逻辑可按需恢复）
    if want_vwap and ('turnover' in df.columns and 'volume' in df.columns):
        df = _compute_vwap_if_possible(df)

    # 排序并按股票打印
    df = df.sort_index()

    # 可选：打印元信息/数据健康
    if meta:
        cal = pd.DatetimeIndex(D.calendar(start_time=start_date, end_time=end_date, future=False))
        print("\n\033[1m=== 数据概览 / Meta ===\033[0m")
        print(f"时间区间: {start_date} ~ {end_date}, 频率: {freq}")
        print(f"交易日总数: {len(cal)}  字段: {', '.join([c for c in df.columns])}")

    # 可选保存 CSV（采用多级索引：instrument, date）
    if save_csv:
        # 输出为普通两级列，便于后续处理
        out = df.reset_index()
        out.rename(columns={'datetime': 'date', 'instrument': 'code'}, inplace=True)
        out.to_csv(save_csv, index=False, encoding='utf-8-sig')
        print(f"已保存合并明细至: {save_csv}")

    instruments_found = sorted(list({idx[0] for idx in df.index}))
    for code in instruments_found:
        sub = df.xs(code, level=0)
        # 显示副本，避免 dtype 警告
        disp = sub.copy().astype('object')
        if 'open' in disp:   disp['open']   = disp['open'].map(lambda x: f"{float(x):.4f}" if pd.notna(x) else "")
        if 'high' in disp:   disp['high']   = disp['high'].map(lambda x: f"{float(x):.4f}" if pd.notna(x) else "")
        if 'low' in disp:    disp['low']    = disp['low'].map(lambda x: f"{float(x):.4f}" if pd.notna(x) else "")
        if 'close' in disp:  disp['close']  = disp['close'].map(lambda x: f"{float(x):.4f}" if pd.notna(x) else "")
        if 'volume' in disp: disp['volume'] = disp['volume'].map(lambda x: f"{float(x):,.0f}" if pd.notna(x) else "")
        if 'turnover' in disp: disp['turnover'] = disp['turnover'].map(lambda x: f"{float(x):,.2f}" if pd.notna(x) else "")
        print(f"\n\033[1m=== {code} @ [{start_date} ~ {end_date}] 共 {len(disp)} 个交易日 ===\033[0m")
        if meta:
            # 与交易日历对齐，统计缺失
            cal = pd.DatetimeIndex(D.calendar(start_time=start_date, end_time=end_date, future=False))
            sub_full = sub.reindex(cal)
            valid_days = len(sub.dropna(how='all'))
            missing_days = len(sub_full) - valid_days
            first_valid = sub.dropna(how='all').index.min()
            last_valid = sub.dropna(how='all').index.max()
            print(f"[Meta] 交易日={len(sub_full)}  有效日={valid_days}  缺失日={missing_days}")
            if pd.notna(first_valid):
                print(f"[Meta] 首个有效交易日={first_valid.date()}  末个有效交易日={last_valid.date()}")
        print(disp.to_string())

    print("\n\033[1m--- 字段含义说明 ---\033[0m")
    print("open, high, low, close:  \033[35m未复权\033[0m价格（依据 Qlib 规范：raw = adjusted / factor，即 open = $open / $factor 等）。")


def main():
    parser = argparse.ArgumentParser(description="CLI for qlib_tool: fetch details for multiple stocks over a date range")
    parser.add_argument("--codes", "-c", nargs="+", help="股票代码列表 (支持带/不带前缀)", required=True)
    parser.add_argument("--start", "-s", help="开始日期 (格式 YYYY-MM-DD)", required=True)
    parser.add_argument("--end", "-e", help="结束日期 (格式 YYYY-MM-DD)", required=True)
    parser.add_argument("--save", help="可选，保存结果到 CSV 文件")
    parser.add_argument("--freq", default="day", choices=["day", "1min", "5min", "15min", "30min", "60min"], help="数据频率，默认 day")
    parser.add_argument("--fields", help="自定义字段列表，逗号分隔；建议不要写$以避免shell变量展开，如: open,close,volume,amount,factor,vwap；不传或写 ALL 则尽可能多地打印可用字段")
    parser.add_argument("--meta", action="store_true", help="打印每只股票的元信息/数据健康统计")

    args = parser.parse_args()

    codes = args.codes
    start_date = args.start
    end_date = args.end
    save_csv = args.save

    print_stocks_details_for_range(
        stocks=codes,
        start_date=start_date,
        end_date=end_date,
        qlib_dir="~/.qlib/qlib_data/cn_data",
        save_csv=save_csv,
        freq=args.freq,
        fields_arg=args.fields,
        meta=args.meta,
    )

# --- 主程序入口 ---
if __name__ == "__main__":
    main()