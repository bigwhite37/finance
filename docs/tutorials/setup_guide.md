# 环境搭建指南

本指南将引导您完成A股强化学习量化交易系统的所有环境搭建和配置步骤。请严格按照以下步骤操作，以确保系统能够顺利运行。

## 1. 系统要求

-   **操作系统**: Linux (推荐 Ubuntu 20.04+) 或 macOS
-   **Python版本**: 3.8 或更高版本
-   **硬件**: 
    -   CPU: 4核或以上
    -   内存: 16GB或以上
    -   GPU: NVIDIA显卡，支持CUDA 11.0+ (强烈推荐，能极大加速模型训练)

## 2. 安装Python依赖

我们建议使用`conda`来管理Python环境，以避免包版本冲突。

### 2.1 创建conda环境
```bash
# 创建一个新的conda环境
conda create -n finance_rl python=3.8

# 激活环境
conda activate finance_rl
```

### 2.2 安装依赖包

项目的所有依赖都记录在`requirements.txt`文件中。

```bash
# 在项目根目录下运行
pip install -r requirements.txt
```

这将安装`torch`, `qlib`, `pandas`, `numpy`, `gymnasium`, `pyyaml`等所有必需的库。

**GPU用户请注意**: `requirements.txt`默认安装CPU版本的PyTorch。如果您的机器上有NVIDIA GPU，请根据您的CUDA版本从[PyTorch官网](https://pytorch.org/get-started/locally/)获取对应的安装命令，以获得GPU加速能力。例如：
```bash
# 示例：安装支持CUDA 11.8的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 3. 配置Qlib数据源

本系统使用`qlib`作为主要的数据源。您需要下载A股市场的日线数据并配置`qlib`。

### 3.1 下载数据

`qlib`提供了自动下载数据的脚本。请在终端中运行以下Python代码：

```python
# 运行这个Python脚本来下载数据
import qlib

# 设置您希望的A股数据存储区域，例如 'cn' (中国) 或 'us' (美国)
# provider_uri参数指定了数据存储的路径，默认是用户主目录下的.qlib/qlib_data/cn_data
# 如果您想更改存储位置，可以修改这个路径，例如 "/path/to/your/data"
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

from qlib.data.auto_updater import AutoUpdater

# 创建一个AutoUpdater实例，用于更新数据
# check_data_legality设置为True，可以在数据更新后检查数据的合法性
updater = AutoUpdater(check_data_legality=True)

# 使用update_data方法来更新数据
# interval参数可以设置为 "day" (日度) 或 "1min" (分钟)
# end_time参数指定了数据更新的结束时间，默认是当前时间
updater.update_data(interval="day")

print("Qlib数据下载和更新完成！")

```

这个过程可能需要一些时间，具体取决于您的网络状况。

### 3.2 验证数据

下载完成后，请在您的项目配置文件（如`config.yaml`）中，确保`data.provider_uri`指向您刚刚下载数据的路径。

```yaml
data:
  provider_uri: "~/.qlib/qlib_data/cn_data" # 确保此路径正确
  # ... 其他配置
```

## 4. 初始化验证

完成以上所有步骤后，运行项目提供的快速演示脚本来验证所有配置是否正确。

```bash
# 在项目根目录下运行
python examples/quick_start.py
```

如果您在终端看到类似以下的输出，并且没有报错，那么恭喜您，环境搭建成功！

```
[INFO] 系统初始化完成
[INFO] 数据加载中...
[INFO] 因子计算完成，共计算20个因子
[INFO] 强化学习环境初始化完成
[INFO] 开始简化训练...
[INFO] 训练完成，开始回测
[INFO] 回测完成，年化收益率: 6.8%, 最大回撤: 8.2%
```

## 5. 常见问题

-   **`ModuleNotFoundError: No module named 'qlib'`**: 说明`qlib`没有正确安装。请确保您已经激活了正确的`conda`环境，并执行了`pip install -r requirements.txt`。
-   **`FileNotFoundError`**: 很可能是`qlib`数据路径配置错误。请仔细检查`config.yaml`中的`provider_uri`是否指向了您存放数据的真实路径。
-   **GPU未被使用**: 运行`python -c "import torch; print(torch.cuda.is_available())"`。如果返回`False`，说明PyTorch的GPU版本没有正确安装。请参考第2.2节重新安装。

--- 

现在您的环境已经准备就绪，可以开始学习[快速开始教程](quick_start.md)来运行您的第一个策略了。
