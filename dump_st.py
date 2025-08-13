import akshare as ak
import json

def get_st_stocks_with_akshare_and_save():
    """
    使用AKShare获取当前所有A股ST股票信息并保存为JSON文件。
    """
    try:
        # AKShare的stock_info_a_code_name接口可以获取所有A股的股票代码和名称
        # 它返回一个pandas DataFrame，包含'code'和'name'两列
        all_stocks_df = ak.stock_info_a_code_name()

        # 筛选出股票名称'name'列中包含'ST'的股票
        # .copy()可以避免后续操作中可能出现的SettingWithCopyWarning
        st_stocks_df = all_stocks_df[all_stocks_df['name'].str.contains('ST')].copy()

        # 如果筛选结果不为空
        if not st_stocks_df.empty:
            print(f"成功通过AKShare找到 {len(st_stocks_df)} 只ST股票。正在保存...")

            # 将筛选后的DataFrame转换为JSON格式
            # orient='records'使得每个股票信息成为一个独立的JSON对象，易于解析
            # force_ascii=False 确保中文字符能正常显示而不是被转义
            st_stocks_json = st_stocks_df.to_json(orient='records', force_ascii=False, indent=4)

            # 将JSON字符串写入本地文件
            with open('st_stocks_akshare.json', 'w', encoding='utf-8') as f:
                f.write(st_stocks_json)

            print("文件 st_stocks_akshare.json 已成功保存。")

        else:
            print("当前市场未发现ST股票。")

    except Exception as e:
        print(f"在通过AKShare获取或处理数据时发生错误: {e}")

# 主程序入口
if __name__ == "__main__":
    print("正在尝试使用AKShare获取A股ST股票列表...")
    get_st_stocks_with_akshare_and_save()