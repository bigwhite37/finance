# 分层RL量化交易系统

基于O3设计文档和Claude实现方案的分层强化学习量化交易系统，采用信息论驱动多样性和MF-PBT训练机制。

## 系统架构

### 核心组件

1. **TimesNet特征编码器** (`models/trans_encoder.py`)
   - 基于Transformer的金融时序特征提取
   - 支持LoRA微调以控制参数量
   - 多尺度时序卷积特征融合

2. **专家策略系统** (`models/expert_policy.py`)
   - 基于SAC的低层专家策略
   - DIAYN互信息奖励机制促进技能多样性
   - 专家种群管理和性能跟踪

3. **元路由器** (`models/meta_router.py`)
   - 基于MaskablePPO的高层决策器
   - KL散度多样性奖励鼓励专家轮换
   - 动态专家选择和风险控制

4. **共享经验池** (`replay/shared_buffer.py`)
   - V-Trace重要性采样校正
   - 多专家经验共享和离策略学习
   - 高效的数据存储和采样机制

5. **交易环境** (`envs/trading_env.py`)
   - 真实市场约束建模（交易成本、滑点、涨跌停）
   - 风险管理和组合监控
   - 基于Qlib的高质量金融数据

6. **MF-PBT训练器** (`trainers/mfpbt_trainer.py`)
   - 多频率基于种群的训练
   - 防止策略坍塌的多样性管理
   - 交替冻结和种群进化机制

## 技术特点

### 信息论驱动多样性
- **高层奖励**: 相对收益 + λ·KL(π_selected ‖ π_mean) 鼓励轮换差异化专家
- **低层目标**: 外部收益 + β·I(a; z) 促进每个策略形成可区分技能
- **V-Trace校正**: 解决离策略漂移问题，确保训练稳定性

### 分层决策架构
- **高层路由器**: 低频决策选择专家，降低动作维度
- **低层专家**: 高频执行具体交易动作，专注特定技能
- **技能分化**: 每个专家学习独特的交易策略和市场适应性

### 多频率训练机制
- **路由器更新**: 每10步，学习专家选择策略
- **专家更新**: 每5步，优化个体交易策略  
- **种群进化**: 每20步，淘汰表现差的专家并注入多样性
- **交替冻结**: 更新专家时冻结路由器，确保训练稳定

## 安装和使用

### 环境配置

```bash
# 克隆项目
git clone <repository_url>
cd finance

# 安装依赖
pip install -r requirements.txt

# 初始化Qlib数据
python -c "import qlib; qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region='cn')"
```

### 数据准备

系统支持多种数据源：

1. **Qlib数据** (主要)
   - 沪深股票日线行情
   - 技术指标和因子数据
   - 高质量的金融数据管道

2. **扩展数据源** (可选)
   - AkShare: 开源财经数据
   - TuShare: 专业金融数据
   - Wind: 高频Tick数据
   - 宏观经济和舆情数据

### 运行训练

```bash
# 使用默认配置训练
python train.py --config configs/default_config.yaml

# 指定输出目录和日志级别
python train.py --config configs/default_config.yaml \
                --output_dir ./outputs/experiment_1 \
                --log_level DEBUG

# 从检查点恢复训练
python train.py --config configs/default_config.yaml \
                --resume ./outputs/experiment_1/checkpoint_50000
```

### 配置说明

系统配置通过YAML文件管理，主要参数：

```yaml
# 专家策略配置
experts:
  n_experts: 5                  # 专家数量
  skill_dim: 10                 # 技能向量维度
  mi_reward_weight: 0.5         # 互信息奖励权重

# 元路由器配置
meta_router:
  kl_lambda: 0.1                # KL散度权重
  ent_coef: 0.01                # 熵系数

# 训练器配置
trainer:
  diversity_threshold: 0.25     # 多样性阈值
  training_schedule:
    router: {freq: 10, steps: 10000}
    experts: {freq: 5, steps: 5000}
    population: {freq: 20, mutation_rate: 0.1}
```

## 系统性能

### 理论优势

1. **解决策略坍塌**: 信息论多样性奖励防止专家收敛为同一策略
2. **克服探索锁死**: KL奖励鼓励高层探索不常用专家
3. **处理离策略漂移**: V-Trace校正确保专家间经验共享的稳定性
4. **维持长期韧性**: MF-PBT种群进化适应市场环境变化

### 预期指标

- **多样性得分** (MAPD): ≥ 0.25 维持专家行为差异化
- **年化收益率**: 目标超越基准指数
- **最大回撤**: 控制在15%以内
- **夏普比率**: 追求风险调整后的稳定收益

## 文件结构

```
finance/
├── src/
│   └── data_loader.py          # 数据加载器
├── models/
│   ├── trans_encoder.py        # TimesNet编码器
│   ├── expert_policy.py        # 专家策略
│   └── meta_router.py          # 元路由器
├── envs/
│   └── trading_env.py          # 交易环境
├── replay/
│   └── shared_buffer.py        # 共享经验池
├── trainers/
│   └── mfpbt_trainer.py        # MF-PBT训练器
├── configs/
│   └── default_config.yaml     # 默认配置
├── train.py                    # 主训练脚本
├── requirements.txt            # 依赖包
└── README.md                   # 项目说明
```

## 引用和致谢

本项目基于以下研究工作：

1. **分层强化学习**: HRPM框架用于金融交易的层级决策
2. **DIAYN算法**: 基于互信息的技能多样性学习
3. **V-Trace**: DeepMind的离策略校正算法
4. **MF-PBT**: 多频率基于种群的训练方法
5. **TimesNet**: 时序数据的Transformer架构

## 许可证

本项目仅供学术研究和教育用途。商业使用请联系作者获得授权。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目Issues
- 邮件联系
- 学术讨论

---

*SuperClaude v2.0.1 | 分层RL量化交易系统 | 基于信息论驱动多样性的专家协作框架*