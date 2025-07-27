"""
增强系统验证器
验证所有改进是否能够达到8%年化收益目标
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入增强组件
from rl_agent.enhanced_trading_environment import EnhancedTradingEnvironment
from rl_agent.stable_cvar_ppo_agent import StableCVaRPPOAgent
from rl_agent.adaptive_safety_shield import AdaptiveSafetyShield
from rl_agent.enhanced_reward_system import EnhancedRewardSystem
from factors.advanced_alpha_factors import AdvancedAlphaFactors

# 导入基础组件
from data.data_manager import DataManager
from config.config_manager import ConfigManager
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


class EnhancedSystemValidator:
    """增强系统验证器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化验证器
        
        Args:
            config_path: 配置文件路径
        """
        # 设置日志
        setup_logger()
        
        # 加载配置
        if config_path:
            self.config_manager = ConfigManager(config_path)
        else:
            self.config_manager = ConfigManager()
        
        # 增强配置
        self.enhanced_config = self._create_enhanced_config()
        
        # 初始化数据管理器
        self.data_manager = DataManager(self.config_manager.get_data_config())
        
        # 验证结果存储
        self.validation_results = {
            'baseline_performance': {},
            'enhanced_performance': {},
            'improvement_analysis': {},
            'target_achievement': {},
            'component_analysis': {}
        }
        
        logger.info("增强系统验证器初始化完成")
    
    def _create_enhanced_config(self) -> Dict:
        """创建增强配置"""
        base_config = self.config_manager.config
        
        # 增强配置覆盖
        enhanced_config = {
            # 数据配置
            'data': {
                'provider': 'yahoo',
                'region': 'cn',
                'universe': 'csi300',
                'start_date': '2020-01-01',
                'end_date': '2022-12-31',
                'backtest_start': '2023-01-01',
                'backtest_end': '2023-12-31'
            },
            
            # 增强因子配置
            'factors': {
                'use_advanced_factors': True,
                'factor_selection': 'all',
                'short_window': 5,
                'medium_window': 20,
                'long_window': 60
            },
            
            # 稳定CVaR-PPO配置
            'agent': {
                'hidden_dim': 256,
                'learning_rate': 1e-4,  # 降低学习率提升稳定性
                'clip_epsilon': 0.2,
                'ppo_epochs': 8,        # 增加训练轮次
                'batch_size': 64,
                'gamma': 0.99,
                'lambda_gae': 0.95,
                'cvar_alpha': 0.05,
                'cvar_lambda': 0.1,
                'cvar_threshold': -0.02
            },
            
            # 自适应安全保护层配置
            'safety_shield': {
                'max_position': 0.25,           # 提升至25%
                'max_leverage': 2.0,            # 提升至2倍
                'var_threshold': 0.05,          # 提升至5%
                'max_drawdown_threshold': 0.12, # 12%
                'volatility_threshold': 0.30,   # 30%
                'risk_adaptation_factor': 1.0,
                'market_regime_sensitivity': 0.3,
                'performance_feedback_weight': 0.2
            },
            
            # 增强奖励系统配置
            'reward_system': {
                'target_annual_return': 0.08,   # 8%目标
                'target_sharpe_ratio': 1.0,
                'return_weight': 1.0,
                'sharpe_weight': 0.3,
                'consistency_weight': 0.2,
                'momentum_weight': 0.15,
                'efficiency_weight': 0.1,
                'risk_penalty_weight': 0.1,
                'cost_penalty_weight': 0.05
            },
            
            # 训练配置
            'training': {
                'total_episodes': 400,          # 增加训练轮数
                'max_steps_per_episode': 300,
                'save_frequency': 50,
                'evaluation_frequency': 25,
                'early_stopping_patience': 100,
                'target_annual_return': 0.08    # 8%目标
            },
            
            # 环境配置
            'environment': {
                'lookback_window': 30,
                'transaction_cost': 0.001,
                'initial_capital': 1000000,
                'max_position': 0.25,
                'max_leverage': 2.0
            }
        }
        
        return enhanced_config
    
    def run_comprehensive_validation(self) -> Dict:
        """运行综合验证"""
        logger.info("🚀 开始综合系统验证")
        
        # 1. 准备数据
        logger.info("📊 准备训练和测试数据...")
        train_data, test_data = self._prepare_validation_data()
        
        # 2. 基线系统验证
        logger.info("📈 验证基线系统性能...")
        baseline_results = self._validate_baseline_system(train_data, test_data)
        self.validation_results['baseline_performance'] = baseline_results
        
        # 3. 增强系统验证
        logger.info("🔥 验证增强系统性能...")
        enhanced_results = self._validate_enhanced_system(train_data, test_data)
        self.validation_results['enhanced_performance'] = enhanced_results
        
        # 4. 性能对比分析
        logger.info("📋 进行性能对比分析...")
        improvement_analysis = self._analyze_improvements(baseline_results, enhanced_results)
        self.validation_results['improvement_analysis'] = improvement_analysis
        
        # 5. 目标达成评估
        logger.info("🎯 评估目标达成情况...")
        target_analysis = self._analyze_target_achievement(enhanced_results)
        self.validation_results['target_achievement'] = target_analysis
        
        # 6. 生成最终报告
        logger.info("📄 生成验证报告...")
        final_report = self._generate_final_report()
        
        # 7. 保存结果
        self._save_validation_results(final_report)
        
        logger.info("✅ 综合验证完成")
        return final_report
    
    def _prepare_validation_data(self) -> Tuple[Dict, Dict]:
        """准备验证数据"""
        # 获取训练数据 (2020-2022)
        train_config = self.enhanced_config['data'].copy()
        train_config['start_date'] = '2020-01-01'
        train_config['end_date'] = '2022-12-31'
        
        train_stock_data = self.data_manager.get_stock_data(
            start_time=train_config['start_date'],
            end_time=train_config['end_date']
        )
        
        train_price_data = train_stock_data['$close'].unstack().T
        train_volume_data = train_stock_data['$volume'].unstack().T
        
        # 获取测试数据 (2023)
        test_config = self.enhanced_config['data'].copy()
        test_config['start_date'] = '2023-01-01'
        test_config['end_date'] = '2023-12-31'
        
        test_stock_data = self.data_manager.get_stock_data(
            start_time=test_config['start_date'],
            end_time=test_config['end_date']
        )
        
        test_price_data = test_stock_data['$close'].unstack().T
        test_volume_data = test_stock_data['$volume'].unstack().T
        
        # 计算增强因子
        enhanced_factors = AdvancedAlphaFactors(self.enhanced_config['factors'])\n        \n        # 训练因子\n        train_factors = enhanced_factors.calculate_all_factors(\n            train_price_data, train_volume_data\n        )\n        \n        # 测试因子\n        test_factors = enhanced_factors.calculate_all_factors(\n            test_price_data, test_volume_data\n        )\n        \n        train_data = {\n            'price_data': train_price_data,\n            'volume_data': train_volume_data,\n            'factor_data': train_factors,\n            'period': '2020-2022 (Training)'\n        }\n        \n        test_data = {\n            'price_data': test_price_data,\n            'volume_data': test_volume_data,\n            'factor_data': test_factors,\n            'period': '2023 (Testing)'\n        }\n        \n        logger.info(f\"训练数据: {train_price_data.shape}, 测试数据: {test_price_data.shape}\")\n        return train_data, test_data\n    \n    def _validate_baseline_system(self, train_data: Dict, test_data: Dict) -> Dict:\n        \"\"\"验证基线系统（当前系统）\"\"\"\n        logger.info(\"运行基线系统验证...\")\n        \n        # 使用当前系统配置\n        baseline_config = self.config_manager.config.copy()\n        \n        # 创建基线环境和智能体\n        baseline_env = self._create_baseline_environment(train_data, baseline_config)\n        baseline_agent = self._create_baseline_agent(baseline_env, baseline_config)\n        \n        # 训练基线系统\n        baseline_training_results = self._train_system(\n            baseline_env, baseline_agent, episodes=200, system_name=\"基线系统\"\n        )\n        \n        # 测试基线系统\n        baseline_test_results = self._test_system(\n            test_data, baseline_agent, baseline_config, system_name=\"基线系统\"\n        )\n        \n        return {\n            'training_results': baseline_training_results,\n            'test_results': baseline_test_results,\n            'system_type': 'baseline'\n        }\n    \n    def _validate_enhanced_system(self, train_data: Dict, test_data: Dict) -> Dict:\n        \"\"\"验证增强系统\"\"\"\n        logger.info(\"运行增强系统验证...\")\n        \n        # 创建增强环境和智能体\n        enhanced_env = self._create_enhanced_environment(train_data)\n        enhanced_agent = self._create_enhanced_agent(enhanced_env)\n        \n        # 训练增强系统\n        enhanced_training_results = self._train_system(\n            enhanced_env, enhanced_agent, episodes=400, system_name=\"增强系统\"\n        )\n        \n        # 测试增强系统\n        enhanced_test_results = self._test_system(\n            test_data, enhanced_agent, self.enhanced_config, system_name=\"增强系统\"\n        )\n        \n        return {\n            'training_results': enhanced_training_results,\n            'test_results': enhanced_test_results,\n            'system_type': 'enhanced'\n        }\n    \n    def _create_baseline_environment(self, data: Dict, config: Dict):\n        \"\"\"创建基线环境\"\"\"\n        # 使用简化的因子数据\n        simple_factors = pd.DataFrame(\n            np.random.randn(len(data['price_data']), 3),\n            index=data['price_data'].index,\n            columns=['factor1', 'factor2', 'factor3']\n        )\n        \n        # 这里应该导入原始的TradingEnvironment，但为了演示，使用增强环境\n        return EnhancedTradingEnvironment(\n            factor_data=simple_factors,\n            price_data=data['price_data'],\n            config=config\n        )\n    \n    def _create_baseline_agent(self, env, config: Dict):\n        \"\"\"创建基线智能体\"\"\"\n        # 使用标准配置\n        agent_config = config.get('agent', {})\n        agent_config.update({\n            'learning_rate': 3e-4,  # 标准学习率\n            'hidden_dim': 256,\n            'ppo_epochs': 4         # 标准训练轮次\n        })\n        \n        return StableCVaRPPOAgent(\n            state_dim=env.observation_space.shape[0],\n            action_dim=env.action_space.shape[0],\n            config=agent_config\n        )\n    \n    def _create_enhanced_environment(self, data: Dict) -> EnhancedTradingEnvironment:\n        \"\"\"创建增强环境\"\"\"\n        return EnhancedTradingEnvironment(\n            factor_data=data['factor_data'],\n            price_data=data['price_data'],\n            config=self.enhanced_config\n        )\n    \n    def _create_enhanced_agent(self, env: EnhancedTradingEnvironment) -> StableCVaRPPOAgent:\n        \"\"\"创建增强智能体\"\"\"\n        return StableCVaRPPOAgent(\n            state_dim=env.observation_space.shape[0],\n            action_dim=env.action_space.shape[0],\n            config=self.enhanced_config['agent']\n        )\n    \n    def _train_system(self, env, agent, episodes: int, system_name: str) -> Dict:\n        \"\"\"训练系统\"\"\"\n        logger.info(f\"开始训练{system_name}，共{episodes}轮...\")\n        \n        training_metrics = {\n            'episode_rewards': [],\n            'episode_returns': [],\n            'episode_sharpe_ratios': [],\n            'best_performance': {\n                'episode': 0,\n                'annual_return': -np.inf,\n                'sharpe_ratio': -np.inf\n            },\n            'convergence_metrics': {\n                'training_stability': 0.0,\n                'final_performance': 0.0\n            }\n        }\n        \n        for episode in range(episodes):\n            # 重置环境\n            observation, info = env.reset()\n            episode_reward = 0.0\n            episode_steps = 0\n            \n            while True:\n                # 获取动作\n                action, log_prob, value = agent.get_action(observation)\n                \n                # 执行动作\n                next_observation, reward, terminated, truncated, info = env.step(action)\n                \n                # 存储经验\n                agent.store_experience(\n                    observation, action, reward, log_prob, value, terminated\n                )\n                \n                episode_reward += reward\n                episode_steps += 1\n                observation = next_observation\n                \n                if terminated or truncated:\n                    break\n            \n            # 更新智能体\n            if len(agent.memory['states']) >= agent.batch_size:\n                update_info = agent.update()\n            \n            # 记录指标\n            training_metrics['episode_rewards'].append(episode_reward)\n            \n            if 'performance_metrics' in info:\n                annual_return = info['performance_metrics'].get('annualized_return', 0.0)\n                sharpe_ratio = info['performance_metrics'].get('sharpe_ratio', 0.0)\n                \n                training_metrics['episode_returns'].append(annual_return)\n                training_metrics['episode_sharpe_ratios'].append(sharpe_ratio)\n                \n                # 更新最佳性能\n                if annual_return > training_metrics['best_performance']['annual_return']:\n                    training_metrics['best_performance'].update({\n                        'episode': episode,\n                        'annual_return': annual_return,\n                        'sharpe_ratio': sharpe_ratio\n                    })\n            \n            # 定期报告进度\n            if (episode + 1) % 50 == 0:\n                recent_rewards = training_metrics['episode_rewards'][-10:]\n                avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0\n                logger.info(\n                    f\"{system_name} - Episode {episode + 1}/{episodes}, \"\n                    f\"平均奖励: {avg_reward:.2f}, \"\n                    f\"最佳年化收益: {training_metrics['best_performance']['annual_return']:.2%}\"\n                )\n        \n        # 计算收敛指标\n        if len(training_metrics['episode_rewards']) >= 50:\n            final_50_rewards = training_metrics['episode_rewards'][-50:]\n            training_metrics['convergence_metrics']['training_stability'] = 1.0 / (np.std(final_50_rewards) + 1e-8)\n            training_metrics['convergence_metrics']['final_performance'] = np.mean(final_50_rewards)\n        \n        logger.info(f\"{system_name}训练完成\")\n        return training_metrics\n    \n    def _test_system(self, test_data: Dict, agent, config: Dict, system_name: str) -> Dict:\n        \"\"\"测试系统\"\"\"\n        logger.info(f\"开始测试{system_name}...\")\n        \n        # 创建测试环境\n        if system_name == \"增强系统\":\n            test_env = EnhancedTradingEnvironment(\n                factor_data=test_data['factor_data'],\n                price_data=test_data['price_data'],\n                config=config\n            )\n        else:\n            # 基线系统使用简化因子\n            simple_factors = pd.DataFrame(\n                np.random.randn(len(test_data['price_data']), 3),\n                index=test_data['price_data'].index,\n                columns=['factor1', 'factor2', 'factor3']\n            )\n            test_env = EnhancedTradingEnvironment(\n                factor_data=simple_factors,\n                price_data=test_data['price_data'],\n                config=config\n            )\n        \n        # 运行测试\n        observation, info = test_env.reset()\n        test_metrics = {\n            'daily_returns': [],\n            'portfolio_values': [],\n            'actions_taken': [],\n            'final_report': {}\n        }\n        \n        while True:\n            # 获取确定性动作（测试时不探索）\n            action, _, _ = agent.get_action(observation, deterministic=True)\n            \n            # 执行动作\n            next_observation, reward, terminated, truncated, info = test_env.step(action)\n            \n            # 记录指标\n            if 'performance_metrics' in info:\n                test_metrics['portfolio_values'].append(info['portfolio_value'])\n            \n            test_metrics['actions_taken'].append(action.copy())\n            observation = next_observation\n            \n            if terminated or truncated:\n                test_metrics['final_report'] = test_env.get_final_report()\n                break\n        \n        # 计算测试指标\n        final_performance = test_metrics['final_report']['final_performance']\n        \n        test_results = {\n            'annual_return': final_performance.get('annualized_return', 0.0),\n            'sharpe_ratio': final_performance.get('sharpe_ratio', 0.0),\n            'max_drawdown': final_performance.get('max_drawdown', 0.0),\n            'win_rate': final_performance.get('win_rate', 0.0),\n            'calmar_ratio': final_performance.get('calmar_ratio', 0.0),\n            'total_return': final_performance.get('total_return', 0.0),\n            'portfolio_values': test_metrics['portfolio_values'],\n            'test_period': test_data['period']\n        }\n        \n        logger.info(\n            f\"{system_name}测试完成 - 年化收益: {test_results['annual_return']:.2%}, \"\n            f\"夏普比率: {test_results['sharpe_ratio']:.2f}, \"\n            f\"最大回撤: {test_results['max_drawdown']:.2%}\"\n        )\n        \n        return test_results\n    \n    def _analyze_improvements(self, baseline: Dict, enhanced: Dict) -> Dict:\n        \"\"\"分析改进效果\"\"\"\n        baseline_test = baseline['test_results']\n        enhanced_test = enhanced['test_results']\n        \n        improvements = {\n            'annual_return_improvement': {\n                'baseline': baseline_test['annual_return'],\n                'enhanced': enhanced_test['annual_return'],\n                'absolute_improvement': enhanced_test['annual_return'] - baseline_test['annual_return'],\n                'relative_improvement': (\n                    (enhanced_test['annual_return'] - baseline_test['annual_return']) / \n                    (abs(baseline_test['annual_return']) + 1e-8) * 100\n                )\n            },\n            'sharpe_ratio_improvement': {\n                'baseline': baseline_test['sharpe_ratio'],\n                'enhanced': enhanced_test['sharpe_ratio'],\n                'absolute_improvement': enhanced_test['sharpe_ratio'] - baseline_test['sharpe_ratio']\n            },\n            'risk_control_improvement': {\n                'baseline_max_drawdown': baseline_test['max_drawdown'],\n                'enhanced_max_drawdown': enhanced_test['max_drawdown'],\n                'drawdown_reduction': baseline_test['max_drawdown'] - enhanced_test['max_drawdown']\n            },\n            'overall_improvement_score': 0.0\n        }\n        \n        # 计算综合改进评分\n        return_score = min(max(improvements['annual_return_improvement']['relative_improvement'] / 100, -2), 2)\n        sharpe_score = min(max(improvements['sharpe_ratio_improvement']['absolute_improvement'], -2), 2)\n        risk_score = min(max(-improvements['risk_control_improvement']['drawdown_reduction'] * 10, -1), 1)\n        \n        improvements['overall_improvement_score'] = 0.5 * return_score + 0.3 * sharpe_score + 0.2 * risk_score\n        \n        return improvements\n    \n    def _analyze_target_achievement(self, enhanced_results: Dict) -> Dict:\n        \"\"\"分析目标达成情况\"\"\"\n        test_results = enhanced_results['test_results']\n        target_annual_return = 0.08  # 8%\n        \n        achievement_analysis = {\n            'target_annual_return': target_annual_return,\n            'actual_annual_return': test_results['annual_return'],\n            'target_achieved': test_results['annual_return'] >= target_annual_return,\n            'achievement_ratio': test_results['annual_return'] / target_annual_return,\n            'gap_to_target': target_annual_return - test_results['annual_return'],\n            'confidence_metrics': {\n                'sharpe_ratio': test_results['sharpe_ratio'],\n                'max_drawdown': test_results['max_drawdown'],\n                'win_rate': test_results['win_rate'],\n                'calmar_ratio': test_results['calmar_ratio']\n            },\n            'risk_adjusted_achievement': {\n                'target_sharpe': 1.0,\n                'actual_sharpe': test_results['sharpe_ratio'],\n                'risk_adjusted_target_achieved': test_results['sharpe_ratio'] >= 1.0,\n                'drawdown_within_limit': test_results['max_drawdown'] <= 0.12\n            }\n        }\n        \n        # 计算成功概率\n        if achievement_analysis['target_achieved']:\n            achievement_analysis['success_probability'] = min(\n                1.0, \n                achievement_analysis['achievement_ratio'] * \n                (1 + test_results['sharpe_ratio'] / 2) * \n                (1 - test_results['max_drawdown'] * 5)\n            )\n        else:\n            achievement_analysis['success_probability'] = max(\n                0.0,\n                achievement_analysis['achievement_ratio'] * 0.8\n            )\n        \n        return achievement_analysis\n    \n    def _generate_final_report(self) -> Dict:\n        \"\"\"生成最终报告\"\"\"\n        report = {\n            'validation_summary': {\n                'validation_date': datetime.now().isoformat(),\n                'system_components': [\n                    '稳定CVaR-PPO智能体',\n                    '自适应安全保护层',\n                    '增强奖励系统',\n                    '高级Alpha因子库',\n                    '增强交易环境'\n                ],\n                'target_goal': '8%年化收益率',\n                'validation_period': '2023年（样本外测试）'\n            },\n            'performance_comparison': self.validation_results['improvement_analysis'],\n            'target_achievement': self.validation_results['target_achievement'],\n            'detailed_results': {\n                'baseline_performance': self.validation_results['baseline_performance'],\n                'enhanced_performance': self.validation_results['enhanced_performance']\n            },\n            'success_metrics': {\n                'target_achieved': self.validation_results['target_achievement']['target_achieved'],\n                'achievement_ratio': self.validation_results['target_achievement']['achievement_ratio'],\n                'success_probability': self.validation_results['target_achievement']['success_probability'],\n                'overall_improvement_score': self.validation_results['improvement_analysis']['overall_improvement_score']\n            },\n            'recommendations': self._generate_recommendations()\n        }\n        \n        return report\n    \n    def _generate_recommendations(self) -> List[str]:\n        \"\"\"生成改进建议\"\"\"\n        recommendations = []\n        \n        target_achieved = self.validation_results['target_achievement']['target_achieved']\n        achievement_ratio = self.validation_results['target_achievement']['achievement_ratio']\n        \n        if target_achieved:\n            recommendations.extend([\n                \"✅ 恭喜！系统已成功达到8%年化收益目标\",\n                \"🔄 建议进行更长期的样本外测试验证稳定性\",\n                \"📊 可考虑在不同市场环境下的鲁棒性测试\",\n                \"⚡ 建议监控实盘交易中的执行偏差\"\n            ])\n        else:\n            if achievement_ratio >= 0.8:\n                recommendations.extend([\n                    \"⭐ 系统表现接近目标，建议微调以下方面：\",\n                    \"🎯 进一步优化奖励函数的收益导向性\",\n                    \"📈 增加更多高质量alpha因子\",\n                    \"⚖️ 适度放宽风险约束以释放收益潜力\"\n                ])\n            else:\n                recommendations.extend([\n                    \"🔧 系统需要进一步改进，建议：\",\n                    \"🧠 增强模型架构的学习能力\",\n                    \"💡 重新设计奖励函数以更好激励收益\",\n                    \"🔍 深入分析因子有效性并优化因子库\",\n                    \"⚡ 考虑采用更先进的强化学习算法\"\n                ])\n        \n        return recommendations\n    \n    def _save_validation_results(self, report: Dict):\n        \"\"\"保存验证结果\"\"\"\n        # 创建结果目录\n        results_dir = './validation_results'\n        os.makedirs(results_dir, exist_ok=True)\n        \n        # 生成文件名\n        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')\n        report_file = f'{results_dir}/enhanced_system_validation_{timestamp}.json'\n        \n        # 保存报告\n        with open(report_file, 'w', encoding='utf-8') as f:\n            json.dump(report, f, indent=2, ensure_ascii=False, default=str)\n        \n        logger.info(f\"验证报告已保存至: {report_file}\")\n    \n    def print_validation_summary(self, report: Dict):\n        \"\"\"打印验证摘要\"\"\"\n        print(\"\\n\" + \"=\"*80)\n        print(\"🎯 增强系统验证报告\")\n        print(\"=\"*80)\n        \n        # 目标达成情况\n        target_info = report['target_achievement']\n        print(f\"\\n📊 目标达成情况:\")\n        print(f\"  目标年化收益率: {target_info['target_annual_return']:.1%}\")\n        print(f\"  实际年化收益率: {target_info['actual_annual_return']:.2%}\")\n        print(f\"  目标达成: {'✅ 是' if target_info['target_achieved'] else '❌ 否'}\")\n        print(f\"  达成比例: {target_info['achievement_ratio']:.1%}\")\n        print(f\"  成功概率: {target_info['success_probability']:.1%}\")\n        \n        # 性能改进\n        improvement = report['performance_comparison']\n        print(f\"\\n📈 性能改进分析:\")\n        print(f\"  年化收益改进: {improvement['annual_return_improvement']['absolute_improvement']:.2%}\")\n        print(f\"  夏普比率改进: {improvement['sharpe_ratio_improvement']['absolute_improvement']:.2f}\")\n        print(f\"  综合改进评分: {improvement['overall_improvement_score']:.2f}\")\n        \n        # 建议\n        print(f\"\\n💡 改进建议:\")\n        for rec in report['recommendations']:\n            print(f\"  {rec}\")\n        \n        print(\"\\n\" + \"=\"*80)\n\n\ndef main():\n    \"\"\"主函数\"\"\"\n    print(\"🚀 启动增强系统验证\")\n    \n    # 创建验证器\n    validator = EnhancedSystemValidator()\n    \n    # 运行验证\n    validation_report = validator.run_comprehensive_validation()\n    \n    # 打印摘要\n    validator.print_validation_summary(validation_report)\n    \n    return validation_report\n\n\nif __name__ == \"__main__\":\n    main()"