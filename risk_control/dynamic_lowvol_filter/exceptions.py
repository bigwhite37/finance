"""
动态低波筛选器异常类定义

定义了筛选器模块中使用的所有异常类，提供清晰的错误分类和处理机制。
"""


class FilterException(Exception):
    """筛选器基础异常
    
    所有筛选器相关异常的基类。
    """
    pass


class DataQualityException(FilterException):
    """数据质量异常
    
    当输入数据不符合质量要求时抛出，例如：
    - 数据为空
    - 缺失值比例过高
    - 数据包含异常值
    - 数据格式不正确
    """
    pass


class ModelFittingException(FilterException):
    """模型拟合异常
    
    当GARCH或其他统计模型拟合失败时抛出，例如：
    - 模型收敛失败
    - 参数估计异常
    - 模型验证失败
    """
    pass


class RegimeDetectionException(FilterException):
    """状态检测异常
    
    当市场状态检测过程失败时抛出，例如：
    - HMM模型拟合失败
    - 状态转换异常
    - 状态识别错误
    """
    pass


class InsufficientDataException(FilterException):
    """数据不足异常
    
    当历史数据不足以进行计算时抛出，例如：
    - 数据长度小于最小要求
    - 有效观测值不足
    - 时间窗口覆盖不足
    """
    pass


class ConfigurationException(FilterException):
    """配置异常
    
    当配置参数不正确或不合理时抛出，例如：
    - 参数值超出有效范围
    - 参数类型错误
    - 配置组合不兼容
    """
    pass