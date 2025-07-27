#!/usr/bin/env python3
"""
增强系统快速测试
基于现有系统，应用关键改进，验证8%目标
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import logging
import torch
import warnings
import yaml
from datetime import datetime
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入基础组件
from data.data_manager import DataManager
from config.config_manager import ConfigManager
from factors.factor_engine import FactorEngine
from rl_agent.cvar_ppo_agent import CVaRPPOAgent
from rl_agent.trading_environment import TradingEnvironment
from rl_agent.safety_shield import SafetyShield
from risk_control.risk_controller import RiskController
from backtest.backtest_engine import BacktestEngine
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


class EnhancedTradingSystem:
    """增强交易系统"""
    
    def __init__(self, config_file: str):
        self.config = self._load_enhanced_config(config_file)
        self._initialize_systems()
    
    def _load_enhanced_config(self, config_file: str) -> dict:
        """加载并增强配置"""
        with open(config_file, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
        
        # 应用增强配置
        enhanced_config = {
            'data': base_config.get('data', {}),
            'agent': {
                'learning_rate': 1e-4,      # 降低学习率提升稳定性
                'hidden_dim': 256,
                'clip_epsilon': 0.2,
                'ppo_epochs': 8,             # 增加训练轮次
                'batch_size': 64,
                'gamma': 0.99,
                'lambda_gae': 0.95,
                'cvar_alpha': 0.05,
                'cvar_lambda': 0.1,
                'cvar_threshold': -0.02
            },
            'safety_shield': {
                'max_position': 0.25,        # 提升仓位限制
                'max_leverage': 2.0,         # 提升杠杆限制
                'var_threshold': 0.05,       # 放宽VaR阈值
                'max_drawdown_threshold': 0.12,
                'volatility_threshold': 0.30
            },
            'environment': {
                'lookback_window': 30,
                'transaction_cost': 0.001,
                'max_position': 0.25,
                'max_leverage': 2.0,
                'lambda1': 0.5,              # 降低回撤惩罚
                'lambda2': 0.3,              # 降低CVaR惩罚
                'reward_amplification': 25.0  # 提升奖励放大倍数
            },
            'training': {
                'total_episodes': 300,       # 适中的训练轮数
                'max_steps_per_episode': 300,
                'save_frequency': 50,
                'evaluation_frequency': 25
            },
            'model': base_config.get('model', {})
        }
        
        return enhanced_config
    
    def _initialize_systems(self):
        """初始化系统组件"""
        self.data_manager = DataManager(self.config['data'])
        self.factor_engine = FactorEngine(self.config)
        self.risk_controller = RiskController(self.config)
        self.backtest_engine = BacktestEngine(self.config)
        
        logger.info("增强交易系统初始化完成")
    
    def prepare_data(self):
        """准备训练数据"""
        logger.info("正在获取股票数据...")
        
        stock_data = self.data_manager.get_stock_data(
            start_time=self.config['data']['start_date'],
            end_time=self.config['data']['end_date'],
            include_fundamentals=True
        )
        
        if stock_data.empty:
            raise ValueError("未能获取到股票数据")
        
        # 获取价格和成交量数据
        price_data = stock_data['$close'].unstack().T
        volume_data = stock_data['$volume'].unstack().T
        
        logger.info(f"价格数据形状: {price_data.shape}")
        
        # 股票池筛选（使用现有逻辑或创建新的）
        if os.path.exists('./models/selected_stocks.pkl'):
            import pickle
            with open('./models/selected_stocks.pkl', 'rb') as f:
                selected_stocks = pickle.load(f)
            logger.info(f"已加载训练时的股票池，包含 {len(selected_stocks)} 只股票")
            
            available_stocks = list(set(selected_stocks) & set(price_data.columns))
            logger.info(f"当前期间可用股票: {len(available_stocks)} 只")
            
            price_data = price_data[available_stocks]
            volume_data = volume_data[available_stocks]
        else:
            # 首次训练，进行简单筛选
            logger.info("进行股票筛选...")
            returns = price_data.pct_change().dropna()
            volatility = returns.std()
            
            # 选择波动率适中的股票
            vol_threshold = volatility.quantile(0.7)
            selected_stocks = volatility[volatility <= vol_threshold].index.tolist()
            
            # 进一步筛选有效股票
            min_data_points = len(price_data) * 0.8
            valid_stocks = []
            for stock in selected_stocks:
                if price_data[stock].notna().sum() >= min_data_points:
                    valid_stocks.append(stock)
            
            selected_stocks = valid_stocks[:800]
            logger.info(f"筛选出 {len(selected_stocks)} 只股票")
            
            # 保存股票池
            os.makedirs('./models', exist_ok=True)
            import pickle
            with open('./models/selected_stocks.pkl', 'wb') as f:
                pickle.dump(selected_stocks, f)
            
            price_data = price_data[selected_stocks]
            volume_data = volume_data[selected_stocks]
        
        # 计算因子（使用增强的因子列表）
        logger.info("正在计算增强因子...")
        
        # 使用现有系统中可用的简单因子
        enhanced_factors = [
            "return_5d", "return_20d", "return_60d", "price_momentum",
            "rsi_14d", "ma_ratio_10d", "ma_ratio_20d", "price_reversal"
        ]
        
        factor_data = self.factor_engine.calculate_all_factors(
            price_data, volume_data, factors=enhanced_factors
        )
        
        logger.info(f"因子计算完成! 因子数据形状: {factor_data.shape}")
        
        return price_data, factor_data
    
    def create_enhanced_environment(self, price_data, factor_data):
        """创建增强交易环境"""
        # 修改环境配置以应用增强奖励函数
        env_config = self.config['environment'].copy()
        
        # 使用现有环境但修改奖励参数
        env = TradingEnvironment(
            factor_data=factor_data,
            price_data=price_data,
            config=env_config,
            train_universe=list(price_data.columns)
        )
        
        # 动态修改奖励函数
        original_calculate_reward = env._calculate_reward
        
        def enhanced_calculate_reward(returns: float, transaction_costs: float) -> float:
            """增强奖励函数"""
            # 基础收益奖励 - 大幅增强
            base_reward = returns * env_config.get('reward_amplification', 25.0)
            
            # 动量奖励
            momentum_bonus = 0.0
            if len(env.portfolio_returns) >= 5:
                recent_returns = np.array(env.portfolio_returns[-5:])
                if np.mean(recent_returns) > 0:
                    momentum_bonus = np.mean(recent_returns) * 8.0
            
            # 夏普比率奖励
            sharpe_bonus = 0.0
            if len(env.portfolio_returns) >= 20:
                returns_array = np.array(env.portfolio_returns)
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                if std_return > 1e-8 and mean_return > 0:
                    sharpe_ratio = mean_return / std_return * np.sqrt(252)
                    sharpe_bonus = sharpe_ratio * 3.0
            
            # 减少风险惩罚
            risk_penalty = 0.0
            current_drawdown = (env.peak_value - env.portfolio_value) / env.peak_value
            if current_drawdown > 0.15:  # 只在15%以上回撤时惩罚
                risk_penalty = env_config.get('lambda1', 0.5) * (current_drawdown - 0.15)
            
            # 轻微成本惩罚
            cost_penalty = transaction_costs * env_config.get('lambda2', 0.3)
            
            # 一致性奖励
            consistency_bonus = 0.0
            if len(env.portfolio_returns) >= 10:
                recent_returns = np.array(env.portfolio_returns[-10:])
                if np.mean(recent_returns) > 0:
                    win_rate = np.mean(recent_returns > 0)
                    consistency_bonus = win_rate * np.mean(recent_returns) * 5.0
            
            total_reward = (base_reward + momentum_bonus + sharpe_bonus + 
                           consistency_bonus - risk_penalty - cost_penalty)
            
            return total_reward
        
        # 替换奖励函数
        env._calculate_reward = enhanced_calculate_reward
        
        return env
    
    def create_enhanced_agent(self, env):
        """创建增强智能体"""
        agent_config = self.config['agent'].copy()
        
        # 计算状态和动作维度
        # 获取一个示例观察来确定维度
        try:
            sample_obs = env.reset()[0]
            state_dim = len(sample_obs) if isinstance(sample_obs, np.ndarray) else sample_obs.shape[0]
        except Exception as e:
            logger.warning(f"无法获取状态维度，使用默认值: {e}")
            state_dim = 100  # 默认维度
        
        action_dim = len(env.train_universe)
        
        # 创建智能体
        agent = CVaRPPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            config=agent_config
        )
        
        # 应用数值稳定性修复
        for param in agent.network.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param, gain=0.5)
            else:
                torch.nn.init.constant_(param, 0.0)
        
        return agent
    
    def create_enhanced_safety_shield(self):
        """创建增强安全保护层"""
        shield_config = self.config['safety_shield']
        
        safety_shield = SafetyShield(shield_config)
        
        # 增强安全检查方法
        original_shield_action = safety_shield.shield_action
        
        def enhanced_shield_action(action, state, price_data=None, current_portfolio=None):
            """增强安全检查"""
            safe_action = action.copy()
            
            # 更宽松的仓位限制
            max_pos = shield_config.get('max_position', 0.25)
            position_sizes = np.abs(safe_action)
            violations = position_sizes > max_pos
            
            if violations.any():
                # 渐进式调整而非硬截断
                scale_factor = max_pos / np.max(position_sizes[violations])
                scale_factor = max(scale_factor, 0.8)  # 保留80%的信号强度
                safe_action[violations] *= scale_factor
            
            # 更宽松的杠杆限制
            max_lev = shield_config.get('max_leverage', 2.0)
            total_leverage = np.sum(np.abs(safe_action))
            
            if total_leverage > max_lev:
                scale_factor = max_lev / total_leverage
                scale_factor = max(scale_factor, 0.7)  # 保留70%的信号强度
                safe_action *= scale_factor
            
            return safe_action
        
        safety_shield.shield_action = enhanced_shield_action
        
        return safety_shield
    
    def train(self):
        """训练增强系统"""
        logger.info("🚀 开始训练增强系统")
        
        # 准备数据
        price_data, factor_data = self.prepare_data()
        
        # 创建增强组件
        env = self.create_enhanced_environment(price_data, factor_data)
        agent = self.create_enhanced_agent(env)
        safety_shield = self.create_enhanced_safety_shield()
        
        # 记录动作维度以备后用
        action_dim = len(env.train_universe)
        
        # 训练参数
        training_config = self.config['training']
        total_episodes = training_config.get('total_episodes', 300)
        save_frequency = training_config.get('save_frequency', 50)
        
        logger.info(f"开始训练: 总共 {total_episodes} 个episode")
        
        best_performance = {
            'episode': 0,
            'annual_return': -np.inf,
            'sharpe_ratio': -np.inf,
            'model_path': None
        }
        
        for episode in range(total_episodes):
            state_info = env.reset()
            if isinstance(state_info, tuple):
                state = state_info[0]
            else:
                state = state_info
            
            # 确保状态是有效的numpy数组
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
            
            episode_reward = 0.0
            episode_steps = 0
            
            while True:
                # 获取动作
                try:
                    action_result = agent.get_action(state)
                    if len(action_result) == 3:
                        action, log_prob, value = action_result
                    else:
                        logger.warning(f"get_action返回了{len(action_result)}个值，期望3个")
                        action = action_result[0] if len(action_result) > 0 else np.zeros(action_dim)
                        log_prob = action_result[1] if len(action_result) > 1 else 0.0
                        value = action_result[2] if len(action_result) > 2 else 0.0
                except Exception as e:
                    logger.error(f"获取动作时出错: {e}")
                    action = np.zeros(action_dim)
                    log_prob = 0.0
                    value = 0.0
                
                # 应用安全保护
                safe_action = safety_shield.shield_action(action, {})
                
                # 执行动作
                step_result = env.step(safe_action)
                if len(step_result) == 4:
                    next_state, reward, done, info = step_result
                    truncated = False
                else:
                    next_state, reward, done, truncated, info = step_result
                
                # 确保下一个状态有效
                if not isinstance(next_state, np.ndarray):
                    next_state = np.array(next_state, dtype=np.float32)
                next_state = np.nan_to_num(next_state, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # 存储经验
                agent.store_experience(state, safe_action, reward, log_prob, value, done or truncated)
                
                episode_reward += reward
                episode_steps += 1
                state = next_state
                
                if done or truncated:
                    break
            
            # 更新智能体
            if len(agent.memory['states']) >= agent.batch_size:
                update_info = agent.update()
            
            # 定期评估和保存
            if (episode + 1) % save_frequency == 0:
                # 计算性能指标
                if len(env.portfolio_returns) > 10:
                    returns_array = np.array(env.portfolio_returns)
                    total_return = (env.portfolio_value / env.initial_capital) - 1.0
                    
                    # 年化收益
                    trading_days = len(returns_array)
                    annual_return = ((1 + total_return) ** (252 / trading_days)) - 1 if trading_days > 0 else 0.0
                    
                    # 夏普比率
                    if np.std(returns_array) > 1e-8:
                        sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
                    else:
                        sharpe_ratio = 0.0
                    
                    logger.info(
                        f"Episode {episode + 1}/{total_episodes}: "
                        f"奖励={episode_reward:.2f}, "
                        f"年化收益={annual_return:.2%}, "
                        f"夏普比率={sharpe_ratio:.2f}"
                    )
                    
                    # 更新最佳模型
                    if annual_return > best_performance['annual_return']:
                        best_performance.update({
                            'episode': episode + 1,
                            'annual_return': annual_return,
                            'sharpe_ratio': sharpe_ratio
                        })
                        
                        # 保存最佳模型
                        model_dir = self.config.get('model', {}).get('save_dir', './models/enhanced_csi300_2020_2022')
                        os.makedirs(model_dir, exist_ok=True)
                        model_path = f"{model_dir}/best_enhanced_agent_episode_{episode + 1}.pth"
                        agent.save_model(model_path)
                        best_performance['model_path'] = model_path
                        
                        logger.info(f"保存最佳模型: {model_path}, 年化收益: {annual_return:.2%}")
        
        logger.info("✅ 训练完成")
        logger.info(f"最佳性能: Episode {best_performance['episode']}, "
                   f"年化收益: {best_performance['annual_return']:.2%}, "
                   f"夏普比率: {best_performance['sharpe_ratio']:.2f}")
        
        return best_performance
    
    def backtest(self, model_path: str):
        """回测增强系统"""
        logger.info("📊 开始回测增强系统")
        
        # 准备回测数据
        price_data, factor_data = self.prepare_data()
        
        # 创建回测环境和智能体
        env = self.create_enhanced_environment(price_data, factor_data)
        agent = self.create_enhanced_agent(env)
        safety_shield = self.create_enhanced_safety_shield()
        
        # 加载模型
        logger.info(f"加载模型: {model_path}")
        agent.load_model(model_path)
        
        # 运行回测
        logger.info("开始运行回测...")
        state = env.reset()
        
        step_count = 0
        while True:
            # 获取动作（确定性）
            action, _, _ = agent.get_action(state, deterministic=True)
            
            # 应用安全保护
            safe_action = safety_shield.shield_action(action, {})
            
            # 执行动作
            next_state, reward, done, info = env.step(safe_action)
            
            state = next_state
            step_count += 1
            
            # 定期输出进度
            if step_count % 50 == 0:
                current_return = (env.portfolio_value / env.initial_capital) - 1.0
                current_drawdown = (env.peak_value - env.portfolio_value) / env.peak_value
                logger.info(f"回测进度: {step_count} 步, 累积收益: {current_return:.2%}, 回撤: {current_drawdown:.2%}")
            
            if done:
                break
        
        # 计算最终性能
        returns_array = np.array(env.portfolio_returns)
        total_return = (env.portfolio_value / env.initial_capital) - 1.0
        
        # 年化收益
        trading_days = len(returns_array)
        annual_return = ((1 + total_return) ** (252 / trading_days)) - 1 if trading_days > 0 else 0.0
        
        # 夏普比率
        if np.std(returns_array) > 1e-8:
            sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # 最大回撤
        max_drawdown = (env.peak_value - env.portfolio_value) / env.peak_value
        
        # 胜率
        win_rate = np.mean(returns_array > 0) if len(returns_array) > 0 else 0.0
        
        # 卡尔玛比率
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 1e-8 else 0.0
        
        # 输出结果
        logger.info("📈 回测完成")
        logger.info("=" * 60)
        logger.info("增强系统回测报告")
        logger.info("=" * 60)
        
        print(f"\n【回测概要】")
        print(f"回测期间: {self.config['data']['start_date']} 至 {self.config['data']['end_date']}")
        print(f"回测步数: {step_count}")
        print(f"最终组合价值: {env.portfolio_value:,.0f}")
        
        print(f"\n【核心指标】")
        print(f"年化收益率: {annual_return:.2%}")
        print(f"夏普比率: {sharpe_ratio:.2f}")
        print(f"最大回撤: {max_drawdown:.2%}")
        print(f"胜率: {win_rate:.1%}")
        print(f"卡尔玛比率: {calmar_ratio:.2f}")
        
        print(f"\n【目标达成评估】")
        target_return = 0.08
        target_achieved = annual_return >= target_return
        
        print(f"目标年化收益: {target_return:.1%}")
        print(f"实际年化收益: {annual_return:.2%}")
        print(f"目标达成: {'✅ 是' if target_achieved else '❌ 否'}")
        print(f"目标完成度: {annual_return / target_return:.1%}")
        
        if target_achieved:
            print(f"🎉 恭喜！增强系统成功达到8%年化收益目标！")
        else:
            gap = target_return - annual_return
            print(f"📈 距离目标还需提升: {gap:.2%}")
        
        return {
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'calmar_ratio': calmar_ratio,
            'target_achieved': target_achieved
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强系统快速测试')
    parser.add_argument('--mode', choices=['train', 'backtest', 'full'], 
                       default='full', help='运行模式')
    parser.add_argument('--train_config', default='config_enhanced_train.yaml',
                       help='训练配置文件')
    parser.add_argument('--backtest_config', default='config_enhanced_backtest.yaml',
                       help='回测配置文件')
    parser.add_argument('--model', help='模型路径（回测模式需要）')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logger()
    
    logger.info("🚀 增强系统快速测试开始")
    
    best_model_path = None
    
    if args.mode in ['train', 'full']:
        logger.info("=" * 60)
        logger.info("第一步：训练增强系统 (2020-2022)")
        logger.info("=" * 60)
        
        train_system = EnhancedTradingSystem(args.train_config)
        best_performance = train_system.train()
        best_model_path = best_performance['model_path']
        
        logger.info(f"训练完成，最佳模型: {best_model_path}")
    
    if args.mode in ['backtest', 'full']:
        logger.info("=" * 60)
        logger.info("第二步：回测增强系统 (2023)")
        logger.info("=" * 60)
        
        # 确定使用的模型路径
        if args.model:
            model_path = args.model
        elif best_model_path:
            model_path = best_model_path
        else:
            # 查找最新的模型
            model_dir = './models/enhanced_csi300_2020_2022'
            if os.path.exists(model_dir):
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
                if model_files:
                    model_path = os.path.join(model_dir, sorted(model_files)[-1])
                else:
                    raise ValueError("未找到可用的模型文件")
            else:
                raise ValueError("未找到模型目录，请先运行训练")
        
        logger.info(f"使用模型: {model_path}")
        
        # 运行回测
        backtest_system = EnhancedTradingSystem(args.backtest_config)
        backtest_results = backtest_system.backtest(model_path)
        
        logger.info("回测完成")
    
    logger.info("✅ 增强系统快速测试完成")


if __name__ == "__main__":
    main()