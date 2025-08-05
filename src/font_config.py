"""
字体配置模块
统一配置matplotlib中文字体显示
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import logging

logger = logging.getLogger(__name__)


def setup_chinese_font():
    """
    配置matplotlib中文字体
    根据不同操作系统选择合适的中文字体
    """
    system = platform.system()
    
    if system == "Darwin":  # macOS
        # macOS系统可用的中文字体
        chinese_fonts = [
            'PingFang SC',      # 苹方-简
            'PingFang HK',      # 苹方-港
            'Heiti TC',         # 黑体-繁
            'STHeiti',          # 华文黑体
            'Hei',              # 黑体
            'Arial Unicode MS'  # Arial Unicode MS
        ]
    elif system == "Windows":  # Windows
        chinese_fonts = [
            'SimHei',           # 黑体
            'Microsoft YaHei',  # 微软雅黑
            'SimSun',           # 宋体
            'KaiTi',            # 楷体
            'Arial Unicode MS'
        ]
    else:  # Linux
        chinese_fonts = [
            'DejaVu Sans',
            'WenQuanYi Micro Hei',  # 文泉驿微米黑
            'WenQuanYi Zen Hei',    # 文泉驿正黑
            'Noto Sans CJK SC',     # 思源黑体
            'Arial Unicode MS'
        ]
    
    # 检查系统中可用的字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 选择第一个可用的中文字体
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        logger.info(f"使用中文字体: {selected_font}")
        plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
    else:
        logger.warning("未找到合适的中文字体，将使用默认字体")
        # 使用系统默认字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] + plt.rcParams['font.sans-serif']
    
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置字体大小
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    
    logger.info("中文字体配置完成")


def test_chinese_display():
    """
    测试中文字体显示效果
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 创建测试图表
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    ax.plot(x, y, label='正弦曲线')
    ax.set_title('中文字体测试图表')
    ax.set_xlabel('横轴标签')
    ax.set_ylabel('纵轴标签')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("如果能正确显示中文，说明字体配置成功")


if __name__ == "__main__":
    setup_chinese_font()
    test_chinese_display()