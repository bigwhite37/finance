# 基于强化学习与Transformer的A股量化交易智能体设计方案

## 一、项目概述

### 1.1 核心技术架构
- **强化学习框架**：采用SAC（Soft Actor-Critic）/PPO（Proximal Policy Optimization）作为决策引擎
- **时序建模**：使用Transformer/Informer架构捕捉长期时序依赖
- **环境设计**：基于OpenAI Gym规范构建Portfolio Environment
- **目标**：在考虑交易成本的情况下，实现年化收益8%-12%，最大回撤控制在15%以内

### 1.2 系统架构图

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    强化学习量化交易智能体系统架构                             │
├────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────────┐    │
│  │   数据处理层     │  │  特征工程层       │  │   时序编码层           │    │
│  │                 │  │                  │  │                        │    │
│  │ • Qlib Data    │─▶│ • 技术指标       │─▶│ • Transformer Encoder  │    │
│  │ • Akshare API  │  │ • 基本面因子     │  │ • Positional Encoding  │    │
│  │ • 实时行情     │  │ • 市场微观结构   │  │ • Multi-Head Attention │    │
│  └─────────────────┘  └──────────────────┘  └────────────────────────┘    │
│                                                         │                    │
│  ┌─────────────────────────────────────────────────────▼────────────────┐  │
│  │                        强化学习决策层                                  │  │
│  │  ┌────────────────┐  ┌─────────────────┐  ┌───────────────────┐    │  │
│  │  │ Portfolio Env  │  │  Actor Network   │  │  Critic Network    │    │  │
│  │  │                │  │                  │  │                    │    │  │
│  │  │ State Space    │◀▶│ Transformer Base │  │ Transformer Base   │    │  │
│  │  │ Action Space   │  │ Policy Head      │  │ Value Head         │    │  │
│  │  │ Reward Function│  │ (SAC/PPO)        │  │ Q-Function         │    │  │
│  │  └────────────────┘  └─────────────────┘  └───────────────────┘    │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                     │                                        │
│  ┌──────────────────────────────────▼────────────────────────────────────┐ │
│  │                    执行与监控层                                         │ │
│  │  • 交易成本模型  • 风险控制  • 实时监控  • 审计日志                   │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
```

## 二、强化学习环境设计

### 2.1 Portfolio Environment定义

```python
import gym
import numpy as np
from gym import spaces
import qlib
from qlib.data import D

class PortfolioEnv(gym.Env):
    """A股投资组合环境，符合OpenAI Gym规范"""
    
    def __init__(self, config):
        super().__init__()
        
        # 市场配置
        self.stock_pool = config['stock_pool']  # 股票池
        self.lookback_window = config['lookback_window']  # 历史窗口
        self.n_stocks = len(self.stock_pool)
        
        # 交易成本参数
        self.commission_rate = 0.001  # 手续费率 0.1%
        self.stamp_tax_rate = 0.001   # 印花税率 0.1%（仅卖出）
        self.slippage_model = AlmgrenChrissModel()  # 市场冲击模型
        
        # 状态空间：[历史特征窗口, 当前持仓, 市场状态]
        self.observation_space = spaces.Dict({
            'features': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.lookback_window, self.n_stocks, config['n_features'])
            ),
            'positions': spaces.Box(
                low=0, high=1, shape=(self.n_stocks,)
            ),
            'market_state': spaces.Box(
                low=-np.inf, high=np.inf, shape=(config['market_features'],)
            )
        })
        
        # 动作空间：目标权重（连续动作）
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_stocks,)
        )
        
        # 风险参数
        self.risk_aversion = config['risk_aversion']
        self.max_drawdown_penalty = config['max_drawdown_penalty']
        
    def step(self, action):
        """执行交易动作"""
        # 1. 标准化动作（确保权重和为1）
        target_weights = action / (action.sum() + 1e-8)
        
        # 2. 计算交易量
        current_weights = self._get_current_weights()
        trade_weights = target_weights - current_weights
        
        # 3. 计算交易成本
        transaction_cost = self._calculate_transaction_cost(
            current_weights, target_weights
        )
        
        # 4. 执行交易（考虑A股规则）
        executed_weights = self._execute_trades(target_weights)
        
        # 5. 计算下一期收益
        next_returns = self._get_next_returns()
        portfolio_return = np.dot(executed_weights, next_returns)
        
        # 6. 计算风险调整后的奖励
        reward = self._calculate_reward(
            portfolio_return, 
            transaction_cost,
            executed_weights
        )
        
        # 7. 更新状态
        self.current_step += 1
        next_state = self._get_observation()
        done = self.current_step >= self.max_steps
        
        # 8. 记录信息（用于监控和审计）
        info = {
            'portfolio_return': portfolio_return,
            'transaction_cost': transaction_cost,
            'executed_weights': executed_weights,
            'sharpe_ratio': self._calculate_sharpe(),
            'max_drawdown': self._calculate_max_drawdown(),
            'timestamp': self.current_date
        }
        
        return next_state, reward, done, info
    
    def _calculate_transaction_cost(self, current_weights, target_weights):
        """计算交易成本（包括手续费、印花税、滑点）"""
        trade_weights = np.abs(target_weights - current_weights)
        
        # 手续费（双边）
        commission = np.sum(trade_weights) * self.commission_rate
        
        # 印花税（仅卖出）
        sell_weights = np.maximum(current_weights - target_weights, 0)
        stamp_tax = np.sum(sell_weights) * self.stamp_tax_rate
        
        # 市场冲击成本（使用Almgren-Chriss模型）
        volumes = self._get_current_volumes()
        slippage = self.slippage_model.calculate_impact(
            trade_weights, volumes, self.total_value
        )
        
        return commission + stamp_tax + slippage
    
    def _calculate_reward(self, portfolio_return, transaction_cost, weights):
        """计算风险调整后的奖励"""
        # 基础收益
        net_return = portfolio_return - transaction_cost
        
        # 风险惩罚
        portfolio_volatility = self._calculate_portfolio_volatility(weights)
        risk_penalty = self.risk_aversion * portfolio_volatility
        
        # 最大回撤惩罚
        current_drawdown = self._calculate_current_drawdown()
        drawdown_penalty = self.max_drawdown_penalty * max(0, current_drawdown - 0.1)
        
        # 最终奖励
        reward = net_return - risk_penalty - drawdown_penalty
        
        return reward
```

### 2.2 Almgren-Chriss市场冲击模型

```python
class AlmgrenChrissModel:
    """市场冲击成本模型"""
    
    def __init__(self, permanent_impact=0.1, temporary_impact=0.05):
        self.gamma = permanent_impact  # 永久冲击系数
        self.eta = temporary_impact    # 临时冲击系数
        
    def calculate_impact(self, trade_weights, volumes, total_value):
        """
        计算市场冲击成本
        参考：Almgren & Chriss (2001)
        """
        # 计算交易占市场成交量的比例
        market_participation = trade_weights * total_value / (volumes + 1e-8)
        
        # 永久冲击（线性）
        permanent_cost = self.gamma * np.sum(market_participation * trade_weights)
        
        # 临时冲击（平方根）
        temporary_cost = self.eta * np.sum(np.sqrt(market_participation) * trade_weights)
        
        return permanent_cost + temporary_cost
```

## 三、Transformer强化学习模型

### 3.1 时序Transformer编码器

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeSeriesTransformer(nn.Module):
    """时序数据的Transformer编码器"""
    
    def __init__(self, config):
        super().__init__()
        
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        
        # 输入嵌入层
        self.input_projection = nn.Linear(
            config['n_features'], self.d_model
        )
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(
            self.d_model, config['max_seq_len']
        )
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=config['d_ff'],
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.n_layers,
            norm=nn.LayerNorm(self.d_model)
        )
        
        # 时间注意力机制
        self.temporal_attention = TemporalAttention(self.d_model)
        
    def forward(self, x, mask=None):
        """
        x: [batch_size, seq_len, n_stocks, n_features]
        """
        batch_size, seq_len, n_stocks, n_features = x.shape
        
        # 重塑为 [batch_size * n_stocks, seq_len, n_features]
        x = x.view(batch_size * n_stocks, seq_len, n_features)
        
        # 输入投影
        x = self.input_projection(x)
        
        # 添加位置编码
        x = self.positional_encoding(x)
        
        # Transformer编码
        x = self.transformer_encoder(x, mask=mask)
        
        # 时间注意力聚合
        x = self.temporal_attention(x)
        
        # 重塑回 [batch_size, n_stocks, d_model]
        x = x.view(batch_size, n_stocks, self.d_model)
        
        return x

class TemporalAttention(nn.Module):
    """时间维度的注意力聚合"""
    
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, num_heads=8, batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, x):
        """聚合时间序列为单一表示"""
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)
        
        # 使用学习的query向量聚合时间信息
        out, _ = self.attention(query, x, x)
        return out.squeeze(1)
```

### 3.2 SAC Actor-Critic网络

```python
class TransformerSACAgent(nn.Module):
    """基于Transformer的SAC智能体"""
    
    def __init__(self, config):
        super().__init__()
        
        # 共享的Transformer编码器
        self.encoder = TimeSeriesTransformer(config['transformer'])
        
        # Actor网络（策略网络）
        self.actor = Actor(
            input_dim=config['transformer']['d_model'],
            n_stocks=config['n_stocks']
        )
        
        # Critic网络（Q函数）
        self.critic_1 = Critic(
            state_dim=config['transformer']['d_model'],
            action_dim=config['n_stocks']
        )
        self.critic_2 = Critic(
            state_dim=config['transformer']['d_model'],
            action_dim=config['n_stocks']
        )
        
        # 目标网络
        self.target_critic_1 = Critic(
            state_dim=config['transformer']['d_model'],
            action_dim=config['n_stocks']
        )
        self.target_critic_2 = Critic(
            state_dim=config['transformer']['d_model'],
            action_dim=config['n_stocks']
        )
        
        # 温度参数（用于熵正则化）
        self.log_alpha = nn.Parameter(torch.zeros(1))
        
    def get_action(self, state, deterministic=False):
        """获取动作（投资组合权重）"""
        # 编码状态
        encoded_state = self.encoder(state['features'])
        
        # 结合其他状态信息
        full_state = torch.cat([
            encoded_state,
            state['positions'].unsqueeze(1).expand(-1, encoded_state.size(1), -1),
            state['market_state'].unsqueeze(1).expand(-1, encoded_state.size(1), -1)
        ], dim=-1)
        
        # 生成动作
        action, log_prob = self.actor(full_state, deterministic)
        
        return action, log_prob

class Actor(nn.Module):
    """策略网络"""
    
    def __init__(self, input_dim, n_stocks):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * n_stocks, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
        # 输出均值和标准差
        self.mean_head = nn.Linear(256, n_stocks)
        self.log_std_head = nn.Linear(256, n_stocks)
        
        # 权重约束（确保和为1）
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, state, deterministic=False):
        """生成投资组合权重"""
        x = state.flatten(start_dim=1)
        x = self.mlp(x)
        
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -20, 2)
        
        if deterministic:
            # 确定性动作
            action = self.softmax(mean)
            log_prob = None
        else:
            # 随机动作（重参数化技巧）
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = self.softmax(x_t)
            
            # 计算对数概率
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob
```

## 四、训练与验证流程

### 4.1 数据集划分策略

```python
class DataSplitStrategy:
    """时序数据的训练/验证/测试划分"""
    
    def __init__(self):
        # 固定划分点
        self.train_period = ('2009-01-01', '2017-12-31')
        self.valid_period = ('2018-01-01', '2021-12-31')
        self.test_period = ('2022-01-01', '2025-12-31')
        
        # 滚动窗口参数
        self.rolling_window = 252 * 4  # 4年训练窗口
        self.retraining_freq = 63      # 每季度重训练
        
    def get_rolling_splits(self, start_date, end_date):
        """生成滚动窗口的数据划分"""
        splits = []
        current_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        
        while current_date < end_date:
            train_start = current_date
            train_end = train_start + pd.Timedelta(days=self.rolling_window)
            valid_start = train_end
            valid_end = valid_start + pd.Timedelta(days=63)  # 3个月验证
            
            if valid_end <= end_date:
                splits.append({
                    'train': (train_start, train_end),
                    'valid': (valid_start, valid_end),
                    'test': (valid_end, valid_end + pd.Timedelta(days=63))
                })
            
            current_date += pd.Timedelta(days=self.retraining_freq)
            
        return splits
```

### 4.2 训练流程与早停机制

```python
class RLTrainer:
    """强化学习训练器"""
    
    def __init__(self, agent, env, config):
        self.agent = agent
        self.env = env
        self.config = config
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(
            agent.actor.parameters(), lr=config['actor_lr']
        )
        self.critic_optimizer = torch.optim.Adam(
            list(agent.critic_1.parameters()) + list(agent.critic_2.parameters()),
            lr=config['critic_lr']
        )
        self.alpha_optimizer = torch.optim.Adam(
            [agent.log_alpha], lr=config['alpha_lr']
        )
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(config['buffer_size'])
        
        # 早停机制
        self.early_stopping = EarlyStopping(
            patience=config['patience'],
            min_delta=config['min_delta']
        )
        
        # 检查点管理
        self.checkpoint_manager = CheckpointManager(config['checkpoint_dir'])
        
    def train(self, n_episodes, validation_env):
        """训练主循环"""
        best_validation_score = -np.inf
        
        for episode in range(n_episodes):
            # 收集经验
            episode_reward = self._collect_experience()
            
            # 更新网络
            if len(self.replay_buffer) > self.config['batch_size']:
                losses = self._update_networks()
            
            # 验证集评估
            if episode % self.config['eval_freq'] == 0:
                validation_score = self._evaluate(validation_env)
                
                # 早停检查
                if self.early_stopping(validation_score):
                    print(f"Early stopping triggered at episode {episode}")
                    break
                
                # 保存最佳模型
                if validation_score > best_validation_score:
                    best_validation_score = validation_score
                    self.checkpoint_manager.save_checkpoint(
                        self.agent, episode, validation_score
                    )
                
            # 记录训练指标
            self._log_metrics(episode, episode_reward, losses, validation_score)
    
    def _update_networks(self):
        """SAC算法更新"""
        batch = self.replay_buffer.sample(self.config['batch_size'])
        
        with torch.no_grad():
            # 计算目标Q值
            next_action, next_log_prob = self.agent.get_action(batch.next_state)
            target_q1 = self.agent.target_critic_1(batch.next_state, next_action)
            target_q2 = self.agent.target_critic_2(batch.next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.agent.log_alpha.exp() * next_log_prob
            target_value = batch.reward + self.config['gamma'] * (1 - batch.done) * target_q
        
        # 更新Critic
        q1 = self.agent.critic_1(batch.state, batch.action)
        q2 = self.agent.critic_2(batch.state, batch.action)
        critic_loss = F.mse_loss(q1, target_value) + F.mse_loss(q2, target_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor
        action, log_prob = self.agent.get_action(batch.state)
        q1_new = self.agent.critic_1(batch.state, action)
        q2_new = self.agent.critic_2(batch.state, action)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.agent.log_alpha.exp() * log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新温度参数
        alpha_loss = -(self.agent.log_alpha * (log_prob + self.config['target_entropy']).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # 软更新目标网络
        self._soft_update_target_networks()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item()
        }
```

## 五、细粒度回测与评估

### 5.1 多频率回测引擎

```python
class MultiFrequencyBacktest:
    """支持日频和分钟频的回测引擎"""
    
    def __init__(self, config):
        self.freq = config['freq']  # '1d' or '1min'
        self.rebalance_freq = config['rebalance_freq']  # 调仓频率
        
        # 初始化Qlib回测引擎
        if self.freq == '1min':
            self.engine = qlib.backtest.HighFreqTradeSim(
                trade_calendar=config['calendar'],
                deal_price='vwap',  # 使用VWAP成交
                min_cost=5,  # 最小手续费5元
            )
        else:
            self.engine = qlib.backtest.Backtest(
                trade_calendar=config['calendar'],
                benchmark=config['benchmark']
            )
    
    def run_backtest(self, agent, start_date, end_date):
        """运行回测"""
        results = {
            'dates': [],
            'portfolio_values': [],
            'positions': [],
            'transactions': [],
            'metrics': {}
        }
        
        # 初始化环境
        env = PortfolioEnv(self._get_env_config())
        state = env.reset(start_date)
        
        current_date = start_date
        while current_date <= end_date:
            # 检查是否需要调仓
            if self._should_rebalance(current_date):
                # 获取动作
                action, _ = agent.get_action(state, deterministic=True)
                
                # 执行交易
                next_state, reward, done, info = env.step(action)
                
                # 记录交易信息
                results['transactions'].append({
                    'date': current_date,
                    'action': action,
                    'cost': info['transaction_cost'],
                    'executed_weights': info['executed_weights']
                })
            
            # 更新投资组合价值
            portfolio_value = self._calculate_portfolio_value(env)
            results['portfolio_values'].append(portfolio_value)
            results['dates'].append(current_date)
            results['positions'].append(env.positions.copy())
            
            # 下一步
            current_date += pd.Timedelta(days=1 if self.freq == '1d' else minutes=1)
            state = next_state
        
        # 计算评估指标
        results['metrics'] = self._calculate_metrics(results)
        
        return results
    
    def _calculate_metrics(self, results):
        """计算详细的评估指标"""
        returns = pd.Series(results['portfolio_values']).pct_change().dropna()
        
        metrics = {
            # 收益指标
            'total_return': (results['portfolio_values'][-1] / results['portfolio_values'][0] - 1),
            'annual_return': self._annualized_return(returns),
            'monthly_returns': returns.resample('M').apply(lambda x: (1 + x).prod() - 1),
            
            # 风险指标
            'volatility': returns.std() * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(results['portfolio_values']),
            'var_95': np.percentile(returns, 5),
            'cvar_95': returns[returns <= np.percentile(returns, 5)].mean(),
            
            # 风险调整指标
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'calmar_ratio': self._calculate_calmar_ratio(returns, metrics['max_drawdown']),
            
            # 交易指标
            'turnover_rate': self._calculate_turnover(results['transactions']),
            'avg_transaction_cost': np.mean([t['cost'] for t in results['transactions']]),
            'trade_count': len(results['transactions']),
            
            # 因子暴露分析
            'factor_exposures': self._analyze_factor_exposures(results['positions'])
        }
        
        return metrics
```

## 六、实时监控与自动化运维

### 6.1 Prometheus监控集成

```python
from prometheus_client import Counter, Gauge, Histogram, start_http_server
import time

class TradingSystemMonitor:
    """交易系统监控"""
    
    def __init__(self):
        # 定义Prometheus指标
        self.metrics = {
            # 性能指标
            'daily_return': Gauge('trading_daily_return', 'Daily portfolio return'),
            'sharpe_ratio': Gauge('trading_sharpe_ratio', 'Rolling 30-day Sharpe ratio'),
            'ic': Gauge('trading_ic', 'Information coefficient'),
            'max_drawdown': Gauge('trading_max_drawdown', 'Current maximum drawdown'),
            
            # 风险指标
            'var_95': Gauge('trading_var_95', '95% Value at Risk'),
            'position_concentration': Gauge('trading_position_concentration', 'Herfindahl index'),
            'sector_exposure': Gauge('trading_sector_exposure', 'Maximum sector exposure', ['sector']),
            
            # 系统指标
            'model_inference_time': Histogram('model_inference_seconds', 'Model inference time'),
            'data_latency': Histogram('data_latency_seconds', 'Data update latency'),
            'error_count': Counter('trading_errors_total', 'Total error count', ['error_type']),
            
            # 交易指标
            'transaction_cost': Gauge('trading_transaction_cost', 'Daily transaction cost'),
            'turnover_rate': Gauge('trading_turnover_rate', 'Portfolio turnover rate'),
            'execution_slippage': Gauge('trading_execution_slippage', 'Execution slippage')
        }
        
        # 启动HTTP服务器
        start_http_server(8000)
        
        # 动态阈值管理
        self.threshold_manager = DynamicThresholdManager()
        
    def update_metrics(self, trading_data):
        """更新监控指标"""
        # 更新性能指标
        self.metrics['daily_return'].set(trading_data['daily_return'])
        self.metrics['sharpe_ratio'].set(trading_data['sharpe_ratio'])
        self.metrics['ic'].set(trading_data['ic'])
        self.metrics['max_drawdown'].set(trading_data['max_drawdown'])
        
        # 更新风险指标
        self.metrics['var_95'].set(trading_data['var_95'])
        self.metrics['position_concentration'].set(trading_data['herfindahl_index'])
        
        # 更新行业暴露
        for sector, exposure in trading_data['sector_exposures'].items():
            self.metrics['sector_exposure'].labels(sector=sector).set(exposure)
        
        # 检查告警
        self._check_alerts(trading_data)
    
    def _check_alerts(self, trading_data):
        """基于动态阈值的告警检查"""
        alerts = []
        
        # 获取动态阈值
        thresholds = self.threshold_manager.get_thresholds(trading_data)
        
        # 检查各项指标
        if trading_data['max_drawdown'] > thresholds['max_drawdown_threshold']:
            alerts.append({
                'level': 'critical',
                'metric': 'max_drawdown',
                'value': trading_data['max_drawdown'],
                'threshold': thresholds['max_drawdown_threshold'],
                'message': f"Maximum drawdown {trading_data['max_drawdown']:.2%} exceeds threshold"
            })
        
        if trading_data['sharpe_ratio'] < thresholds['min_sharpe_threshold']:
            alerts.append({
                'level': 'warning',
                'metric': 'sharpe_ratio',
                'value': trading_data['sharpe_ratio'],
                'threshold': thresholds['min_sharpe_threshold'],
                'message': f"Sharpe ratio {trading_data['sharpe_ratio']:.2f} below threshold"
            })
        
        # 发送告警
        if alerts:
            self._send_alerts(alerts)

class DynamicThresholdManager:
    """动态阈值管理器"""
    
    def __init__(self, lookback_days=90):
        self.lookback_days = lookback_days
        self.historical_data = []
        
    def get_thresholds(self, current_data):
        """基于历史分位数计算动态阈值"""
        self.historical_data.append(current_data)
        
        if len(self.historical_data) < self.lookback_days:
            # 使用默认阈值
            return {
                'max_drawdown_threshold': 0.15,
                'min_sharpe_threshold': 0.5,
                'max_var_threshold': 0.03
            }
        
        # 计算历史分位数
        historical_df = pd.DataFrame(self.historical_data[-self.lookback_days:])
        
        return {
            'max_drawdown_threshold': historical_df['max_drawdown'].quantile(0.95),
            'min_sharpe_threshold': historical_df['sharpe_ratio'].quantile(0.05),
            'max_var_threshold': historical_df['var_95'].quantile(0.95)
        }
```

### 6.2 Canary部署与灰度发布

```python
class CanaryDeployment:
    """金丝雀部署管理"""
    
    def __init__(self, config):
        self.canary_ratio = config['initial_canary_ratio']  # 初始5%资金
        self.evaluation_period = config['evaluation_period']  # 评估期14天
        self.promotion_criteria = config['promotion_criteria']
        
    def deploy_new_model(self, new_model, production_model):
        """部署新模型"""
        deployment = {
            'model_version': new_model.version,
            'start_date': datetime.now(),
            'status': 'canary',
            'metrics': [],
            'canary_portfolio': Portfolio(self.canary_ratio),
            'production_portfolio': Portfolio(1 - self.canary_ratio)
        }
        
        # 启动并行运行
        self._run_parallel_evaluation(new_model, production_model, deployment)
        
        return deployment
    
    def _run_parallel_evaluation(self, new_model, production_model, deployment):
        """并行评估新旧模型"""
        for day in range(self.evaluation_period):
            # 获取当日数据
            market_data = self._get_market_data()
            
            # 运行两个模型
            canary_action = new_model.predict(market_data)
            production_action = production_model.predict(market_data)
            
            # 执行交易
            canary_return = deployment['canary_portfolio'].execute(canary_action)
            production_return = deployment['production_portfolio'].execute(production_action)
            
            # 记录指标
            daily_metrics = {
                'date': datetime.now(),
                'canary_return': canary_return,
                'production_return': production_return,
                'canary_sharpe': self._calculate_sharpe(deployment['canary_portfolio']),
                'production_sharpe': self._calculate_sharpe(deployment['production_portfolio'])
            }
            deployment['metrics'].append(daily_metrics)
            
            # 检查是否需要回滚
            if self._should_rollback(deployment['metrics']):
                self._rollback_deployment(deployment)
                return
        
        # 评估是否推广
        if self._should_promote(deployment['metrics']):
            self._promote_to_production(new_model, deployment)
    
    def _should_promote(self, metrics):
        """判断是否推广到生产环境"""
        metrics_df = pd.DataFrame(metrics)
        
        # 计算评估指标
        canary_avg_return = metrics_df['canary_return'].mean()
        production_avg_return = metrics_df['production_return'].mean()
        canary_sharpe = metrics_df['canary_sharpe'].mean()
        production_sharpe = metrics_df['production_sharpe'].mean()
        
        # 推广条件
        return (
            canary_avg_return > production_avg_return * 0.95 and  # 收益不低于95%
            canary_sharpe > production_sharpe * 0.9 and  # Sharpe不低于90%
            metrics_df['canary_return'].min() > -0.02  # 单日最大亏损不超过2%
        )
    
    def _promote_to_production(self, new_model, deployment):
        """逐步推广到生产环境"""
        promotion_schedule = [0.1, 0.25, 0.5, 0.75, 1.0]  # 逐步增加比例
        
        for ratio in promotion_schedule:
            deployment['canary_portfolio'].resize(ratio)
            deployment['production_portfolio'].resize(1 - ratio)
            
            # 继续监控7天
            additional_metrics = self._monitor_for_days(7)
            
            if not self._is_stable(additional_metrics):
                self._rollback_deployment(deployment)
                return
        
        deployment['status'] = 'production'
        print(f"Model {deployment['model_version']} successfully promoted to production")
```

## 七、合规审计与可解释性

### 7.1 决策审计日志

```python
class AuditLogger:
    """审计日志系统"""
    
    def __init__(self, retention_years=5):
        self.retention_years = retention_years
        self.log_storage = TimeSeries Database()  # 使用时序数据库
        
    def log_trading_decision(self, decision_data):
        """记录交易决策"""
        audit_record = {
            'timestamp': datetime.now().isoformat(),
            'model_version': decision_data['model_version'],
            'market_snapshot': {
                'index_level': decision_data['index_level'],
                'vix': decision_data['vix'],
                'market_turnover': decision_data['market_turnover']
            },
            'features': decision_data['features'].to_dict(),
            'model_output': {
                'raw_scores': decision_data['raw_scores'],
                'final_weights': decision_data['final_weights'],
                'confidence': decision_data['confidence']
            },
            'risk_metrics': {
                'portfolio_var': decision_data['var'],
                'max_drawdown': decision_data['max_drawdown'],
                'concentration': decision_data['concentration']
            },
            'execution': {
                'target_weights': decision_data['target_weights'],
                'executed_weights': decision_data['executed_weights'],
                'transaction_cost': decision_data['transaction_cost'],
                'slippage': decision_data['slippage']
            },
            'compliance_checks': decision_data['compliance_checks']
        }
        
        # 存储到时序数据库
        self.log_storage.insert(audit_record)
        
        # 生成审计报告
        if decision_data.get('generate_report', False):
            self._generate_audit_report(audit_record)
    
    def query_historical_decisions(self, start_date, end_date, filters=None):
        """查询历史决策记录"""
        query = {
            'time_range': (start_date, end_date),
            'filters': filters or {}
        }
        
        return self.log_storage.query(query)
```

### 7.2 模型可解释性

```python
import shap
from lime import lime_tabular

class ModelExplainer:
    """模型决策解释器"""
    
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        
        # 初始化SHAP解释器
        self.shap_explainer = shap.DeepExplainer(
            model, 
            background_data=self._get_background_data()
        )
        
        # 初始化LIME解释器
        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=self._get_training_data(),
            feature_names=feature_names,
            mode='regression'
        )
    
    def explain_prediction(self, input_data, stock_id):
        """解释单个预测"""
        explanation = {
            'stock_id': stock_id,
            'prediction': self.model.predict(input_data),
            'shap_values': self._get_shap_explanation(input_data),
            'lime_explanation': self._get_lime_explanation(input_data),
            'feature_importance': self._get_feature_importance(input_data)
        }
        
        # 生成可视化报告
        self._generate_explanation_report(explanation)
        
        return explanation
    
    def _get_shap_explanation(self, input_data):
        """SHAP解释"""
        shap_values = self.shap_explainer.shap_values(input_data)
        
        # 获取前10个最重要的特征
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': shap_values[0],
            'abs_shap_value': np.abs(shap_values[0])
        }).sort_values('abs_shap_value', ascending=False).head(10)
        
        return {
            'shap_values': shap_values,
            'top_features': feature_importance.to_dict('records')
        }
    
    def _get_feature_importance(self, input_data):
        """特征重要性分析"""
        # 使用注意力权重（如果是Transformer模型）
        if hasattr(self.model, 'get_attention_weights'):
            attention_weights = self.model.get_attention_weights(input_data)
            
            # 聚合时间维度的注意力
            temporal_importance = attention_weights.mean(axis=0)
            
            # 聚合特征维度的注意力
            feature_importance = attention_weights.mean(axis=1)
            
            return {
                'temporal_importance': temporal_importance,
                'feature_importance': feature_importance,
                'attention_heatmap': attention_weights
            }
        
        return {}
    
    def generate_compliance_report(self, decisions, period):
        """生成合规报告"""
        report = {
            'period': period,
            'total_decisions': len(decisions),
            'model_versions': list(set(d['model_version'] for d in decisions)),
            'risk_violations': self._check_risk_violations(decisions),
            'concentration_analysis': self._analyze_concentration(decisions),
            'turnover_analysis': self._analyze_turnover(decisions),
            'cost_analysis': self._analyze_costs(decisions),
            'performance_attribution': self._attribute_performance(decisions)
        }
        
        # 保存报告
        self._save_compliance_report(report)
        
        return report
```

## 八、项目实施计划

### 8.1 项目结构

```
rl_trading_system/
├── config/
│   ├── model_config.yaml
│   ├── trading_config.yaml
│   ├── monitoring_config.yaml
│   └── compliance_config.yaml
├── data/
│   ├── collectors/
│   │   ├── qlib_collector.py
│   │   └── akshare_collector.py
│   ├── processors/
│   │   ├── feature_engineering.py
│   │   └── data_validation.py
│   └── cache/
├── models/
│   ├── transformer/
│   │   ├── encoder.py
│   │   └── attention.py
│   ├── rl_agents/
│   │   ├── sac_agent.py
│   │   └── ppo_agent.py
│   └── utils/
├── trading/
│   ├── environment/
│   │   ├── portfolio_env.py
│   │   └── market_simulator.py
│   ├── execution/
│   │   ├── order_manager.py
│   │   └── cost_model.py
│   └── risk/
├── backtest/
│   ├── engine/
│   ├── analysis/
│   └── visualization/
├── monitoring/
│   ├── metrics/
│   ├── alerts/
│   └── dashboards/
├── deployment/
│   ├── canary/
│   ├── rollback/
│   └── versioning/
├── audit/
│   ├── logger/
│   ├── explainer/
│   └── reports/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── scripts/
    ├── train.py
    ├── evaluate.py
    ├── deploy.py
    └── monitor.py
```

## 九、总结

本方案通过结合强化学习和Transformer架构，构建了一个完整的A股量化交易系统。关键创新点包括：

1. **强化学习决策**：使用SAC/PPO算法自动学习最优交易策略
2. **Transformer时序建模**：捕捉长期市场模式和复杂依赖关系
3. **精确成本建模**：考虑手续费、印花税和市场冲击
4. **严格的训练流程**：三段式数据划分和滚动窗口验证
5. **细粒度监控**：实时指标监控和动态阈值告警
6. **完整的合规体系**：审计日志和模型可解释性

系统预期在严格控制风险的前提下，实现8%-12%的年化收益目标，为投资者提供稳健可靠的量化投资工具。