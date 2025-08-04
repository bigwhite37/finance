# SB3 重构任务列表

## 任务概述
根据 sb3.md 中的代码审查结果，需要对 SAC 智能体迁移到 Stable-Baselines3 框架进行完善和修复。

## 重构任务清单

### 1. SACAgent 实现修复
- [x] **Task 1.1**: 修复参数注入与 policy_kwargs 同步问题 ✅
  - 修复 `_inject_training_params` 与 `_create_model()` 之间的参数同步
  - 确保 training_config 修改后能正确更新模型参数
  - 优化 activation_fn 传递方式（传递类而非实例）

- [x] **Task 1.2**: 优化 Transformer 特征提取器 ✅
  - 修复 forward() 中张量维度处理的激进操作
  - 优化 DictObs 处理逻辑
  - 添加维度检查和错误处理

- [x] **Task 1.3**: 改进向量化环境创建 ✅
  - 移除多余的 test_env 创建步骤
  - 优化 VecTransposeDict 的使用条件
  - 添加 GPU/IO 资源管理

- [x] **Task 1.4**: 统一模型保存/加载路径 ✅
  - 确保保存和加载路径的一致性
  - 修复文件扩展名处理

### 2. Callback 体系完善
- [x] **Task 2.1**: 修复 TrainingProgressCallback ✅
  - 修复多进程环境下的 log_freq 缩放问题
  - 确保 num_timesteps 计算正确

- [x] **Task 2.2**: 集成 EarlyStopping 到 SB3 callback 链 ✅
  - 将 EarlyStopping 和 DrawdownEarlyStopping 改写为继承 BaseCallback
  - 集成到 _create_callbacks() 中

### 3. RLTrainer 修复
- [x] **Task 3.1**: 实现缺失的 evaluate() 方法 ✅
  - 添加 evaluate() 方法实现
  - 集成 stable_baselines3.common.evaluation.evaluate_policy

- [x] **Task 3.2**: 修复多进程配置 ✅
  - 优化 mp.set_start_method 调用
  - 统一 parallel_environments 与 n_envs 配置

- [ ] **Task 3.3**: 优化指标历史内存管理
  - 实现指标历史的窗口统计或定期落盘
  - 防止长训练中的内存溢出

### 4. 驱动脚本修复
- [x] **Task 4.1**: 修复 trainer.evaluate() 调用 ✅
  - 替换不存在的 trainer.evaluate() 调用
  - 使用正确的评估方法

- [x] **Task 4.2**: 统一配置字段访问 ✅
  - 统一 model_config["training"] vs model_config["model"]["training"] 的使用
  - 添加默认值处理

- [x] **Task 4.3**: 修复模型保存路径 ✅
  - 修正 final_model 路径处理
  - 确保与 agent.save() 的兼容性

- [ ] **Task 4.4**: 优化日志输出
  - 修复多进程环境下的 ColorFormatter 线程安全问题

### 5. 性能优化和改进
- [ ] **Task 5.1**: 添加数据归一化
  - 集成 VecNormalize 进行自动归一化
  - 替换手动标准化逻辑

- [x] **Task 5.2**: 优化 Replay Buffer 大小 ✅
  - 根据数据集大小调整 buffer 容量
  - 避免内存浪费

- [ ] **Task 5.3**: 添加 Transformer 输入掩码
  - 为变长序列添加 padding mask
  - 优化 token 处理效率

- [ ] **Task 5.4**: 集成性能监控
  - 使用 SB3 内置的 tensorboard scalars
  - 优化 EnhancedMetricsCallback

- [ ] **Task 5.5**: 性能加速优化
  - 考虑集成 torch.compile 或 Flash-Attention
  - 替换原生 nn.MultiheadAttention

### 6. 代码质量保证
- [x] **Task 6.1**: 异常处理审查 ✅
  - 检查并移除所有不当的异常捕获
  - 确保异常正确传播或处理

- [ ] **Task 6.2**: 测试用例完善
  - 为修复的功能添加单元测试
  - 确保测试覆盖率

- [ ] **Task 6.3**: 文档更新
  - 更新 API 文档
  - 添加使用示例

## 执行状态总结

### 已完成的关键修复 ✅
1. **运行时错误修复**: Task 3.1, 4.1 - 修复了 trainer.evaluate() 缺失问题
2. **配置同步问题**: Task 1.1, 4.2 - 统一了参数注入和配置访问
3. **核心功能优化**: Task 1.2, 1.3, 1.4, 2.1, 2.2, 3.2 - 优化了 Transformer、环境创建、回调、模型保存等
4. **路径和异常处理**: Task 4.3, 6.1 - 修复了模型保存路径和异常处理
5. **性能优化**: Task 5.2 - 智能调整 Replay Buffer 大小

### 完成统计
- ✅ **已完成**: 11/17 个任务 (65%)
- ⏳ **待完成**: 6/17 个任务 (35%)

### 核心问题已解决
所有在 sb3.md 中指出的**关键运行时错误**和**配置同步问题**都已修复，系统现在可以正常运行。剩余任务主要是性能优化和功能增强。

### 执行顺序（已按此顺序完成）
1. ✅ 首先修复关键的运行时错误（Task 3.1, 4.1）
2. ✅ 然后修复配置和参数同步问题（Task 1.1, 4.2）
3. ✅ 接着优化核心功能（Task 1.2, 1.3, 2.1, 3.2）
4. ⏳ 最后进行性能优化和改进（Task 5.x - 待完成）

## 注意事项
- 严格遵守异常处理规则，不得吞掉异常
- 不使用临时补丁或硬编码
- 保持测试用例的完整性
- 无法获取数据时立即抛出 RuntimeError
- 所有注释和提交信息使用中文