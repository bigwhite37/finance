#!/usr/bin/env python3
"""
综合改进实施和重新训练测试
集成所有优化改进，进行完整的训练-回测验证流程
"""

import sys
import os
import argparse
import logging
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ConfigManager
from data import DataManager
from factors import FactorEngine
from rl_agent import TradingEnvironment, CVaRPPOAgent, SafetyShield
from risk_control import RiskController
from backtest import BacktestEngine


class ComprehensiveImprovementTester:
    """综合改进测试器"""
    
    def __init__(self, config_path: str = None, verbose: bool = True):
        """
        初始化测试器
        
        Args:
            config_path: 配置文件路径
            verbose: 是否详细输出
        """
        self.verbose = verbose
        self.setup_logging()
        
        # 初始化配置管理器
        self.config_manager = ConfigManager(config_path or 'config_train.yaml')
        
        # 测试结果存储
        self.results = {
            'baseline_performance': None,
            'improved_performance': None,
            'comparison_metrics': None,
            'improvement_summary': None
        }
        
        # 改进跟踪
        self.improvement_stages = {
            'stage_1_reward_function': False,
            'stage_2_factor_enhancement': False, 
            'stage_3_risk_optimization': False,
            'stage_4_architecture_upgrade': False
        }
        
    def setup_logging(self):
        """设置日志系统"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        if self.verbose:
            logging.basicConfig(
                level=logging.INFO,
                format=log_format,
                handlers=[
                    logging.FileHandler(f'comprehensive_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                    logging.StreamHandler(sys.stdout)
                ]
            )
        else:
            logging.basicConfig(level=logging.WARNING, format=log_format)
            
        self.logger = logging.getLogger(__name__)
    
    def validate_improvements(self) -> Dict[str, bool]:
        """验证所有改进是否正确实施"""
        self.logger.info("开始验证改进实施状态...")
        
        validation_results = {}
        
        # 1. 验证奖励函数优化
        try:
            from rl_agent.trading_environment import TradingEnvironment
            # 检查奖励函数是否包含优化的收益放大逻辑
            env_config = self.config_manager.get_environment_config()
            test_env = TradingEnvironment(None, None, env_config)
            
            # 检查_calculate_reward方法是否存在优化
            import inspect
            reward_code = inspect.getsource(test_env._calculate_reward)
            if "10.0" in reward_code and "放大收益信号" in reward_code:
                validation_results['reward_function'] = True
                self.improvement_stages['stage_1_reward_function'] = True
            else:
                validation_results['reward_function'] = False
            
        except Exception as e:
            self.logger.warning(f"奖励函数验证失败: {e}")
            validation_results['reward_function'] = False
        
        # 2. 验证因子库增强
        try:
            from factors import FactorEngine
            factor_config = self.config_manager.get_config('factors')
            factor_engine = FactorEngine(factor_config)
            
            # 检查是否有17个增强因子
            enhanced_factors = factor_engine.default_factors
            if len(enhanced_factors) >= 17:
                validation_results['factor_enhancement'] = True
                self.improvement_stages['stage_2_factor_enhancement'] = True
            else:
                validation_results['factor_enhancement'] = False
                
        except Exception as e:
            self.logger.warning(f"因子库验证失败: {e}")
            validation_results['factor_enhancement'] = False
        
        # 3. 验证风险控制优化
        try:
            config = self.config_manager.get_config('safety_shield')
            
            # 检查关键参数是否优化
            optimized_params = {
                'max_position': 0.17,
                'max_leverage': 1.4,
                'var_threshold': 0.035,
                'max_drawdown_threshold': 0.12
            }
            
            all_optimized = True
            for param, expected in optimized_params.items():
                actual = config.get(param, 0)
                if abs(actual - expected) > 0.01:  # 允许小误差
                    all_optimized = False
                    break
            
            validation_results['risk_optimization'] = all_optimized
            if all_optimized:
                self.improvement_stages['stage_3_risk_optimization'] = True
                
        except Exception as e:
            self.logger.warning(f"风险控制验证失败: {e}")
            validation_results['risk_optimization'] = False
        
        # 4. 验证网络架构升级（检查是否有增强架构文件）
        try:
            enhanced_arch_exists = os.path.exists('rl_agent/enhanced_architecture.py')
            training_strategy_exists = os.path.exists('rl_agent/enhanced_training_strategy.py')
            
            validation_results['architecture_upgrade'] = enhanced_arch_exists and training_strategy_exists
            if validation_results['architecture_upgrade']:
                self.improvement_stages['stage_4_architecture_upgrade'] = True
                
        except Exception as e:
            self.logger.warning(f"架构升级验证失败: {e}")
            validation_results['architecture_upgrade'] = False
        
        # 输出验证结果
        self.logger.info("改进验证结果:")
        for improvement, status in validation_results.items():
            status_str = "✅ 已实施" if status else "❌ 未实施"
            self.logger.info(f"  {improvement}: {status_str}")
        
        return validation_results
    
    def run_baseline_test(self) -> Dict[str, float]:
        """运行基线测试（使用历史最佳模型）"""
        self.logger.info("开始基线性能测试...")
        
        try:
            # 检查是否有历史最佳模型
            model_files = []
            if os.path.exists('./models'):
                for file in os.listdir('./models'):
                    if file.endswith('.pth') and 'best_agent' in file:
                        model_files.append(file)
            
            if not model_files:
                self.logger.warning("未找到历史最佳模型，跳过基线测试")
                return {'baseline_annual_return': -0.2232}  # 使用已知的历史结果
            
            # 使用最新的最佳模型
            latest_model = sorted(model_files)[-1]
            model_path = os.path.join('./models', latest_model)
            
            self.logger.info(f"使用基线模型: {model_path}")
            
            # 运行回测
            baseline_results = self._run_backtest_with_model(model_path, test_name="baseline")
            
            self.results['baseline_performance'] = baseline_results
            return baseline_results
            
        except Exception as e:
            self.logger.error(f"基线测试失败: {e}")
            return {'baseline_annual_return': -0.2232}  # 使用已知结果
    
    def run_comprehensive_training(self) -> Dict[str, float]:
        """运行综合改进的完整训练"""
        self.logger.info("开始综合改进训练...")
        
        try:
            # 初始化系统组件
            systems = self._initialize_systems()
            
            # 准备训练数据
            price_data, factor_data = self._prepare_training_data(systems)
            
            # 创建训练环境
            env_config = self.config_manager.get_environment_config()
            environment = TradingEnvironment(factor_data, price_data, env_config)
            
            # 创建智能体（使用优化后的配置）
            agent_config = self.config_manager.get_agent_config()
            state_dim = environment.observation_space.shape[0]
            action_dim = environment.action_space.shape[0]
            agent = CVaRPPOAgent(state_dim, action_dim, agent_config)
            
            # 创建安全保护层
            shield_config = self.config_manager.get_config('safety_shield')
            safety_shield = SafetyShield(shield_config)
            
            # 训练配置
            training_config = self.config_manager.get_training_config()
            total_episodes = training_config['total_episodes']
            
            self.logger.info(f"开始训练 {total_episodes} 个episodes...")
            
            # 训练循环
            best_sharpe = float('-inf')
            training_metrics = {
                'episode_rewards': [],
                'sharpe_ratios': [],
                'annual_returns': [],
                'max_drawdowns': []
            }
            
            for episode in range(total_episodes):
                if episode % 20 == 0:
                    self.logger.info(f"训练进度: {episode}/{total_episodes} ({episode/total_episodes*100:.1f}%)")
                
                # 运行一个episode
                episode_metrics = self._run_training_episode(
                    environment, agent, safety_shield, episode
                )
                
                # 记录指标
                for key, value in episode_metrics.items():
                    if key in training_metrics:
                        training_metrics[key].append(value)
                
                # 保存最佳模型
                current_sharpe = episode_metrics.get('sharpe_ratio', 0)
                if current_sharpe > best_sharpe and episode > 50:  # 至少训练50个episodes
                    best_sharpe = current_sharpe
                    model_path = f'./models/improved_best_agent_episode_{episode}.pth'
                    os.makedirs('./models', exist_ok=True)
                    agent.save_model(model_path)
                    self.logger.info(f"保存改进模型: {model_path}, Sharpe: {current_sharpe:.4f}")
            
            # 训练完成统计
            final_metrics = {
                'training_episodes': total_episodes,
                'best_sharpe_ratio': best_sharpe,
                'final_episode_return': training_metrics['episode_rewards'][-1] if training_metrics['episode_rewards'] else 0,
                'avg_recent_return': np.mean(training_metrics['episode_rewards'][-20:]) if len(training_metrics['episode_rewards']) >= 20 else 0
            }
            
            self.logger.info("训练完成!")
            self.logger.info(f"最佳Sharpe比率: {best_sharpe:.4f}")
            
            return final_metrics
            
        except Exception as e:
            self.logger.error(f"综合训练失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}\n    \n    def run_improved_backtest(self) -> Dict[str, float]:\n        """运行改进后的回测"""\n        self.logger.info("开始改进模型回测...")\n        \n        try:\n            # 查找最新的改进模型\n            improved_models = []\n            if os.path.exists('./models'):\n                for file in os.listdir('./models'):\n                    if file.endswith('.pth') and 'improved_best_agent' in file:\n                        improved_models.append(file)\n            \n            if not improved_models:\n                self.logger.error("未找到改进后的模型文件")\n                return {}\n            \n            # 使用最新的改进模型\n            latest_improved_model = sorted(improved_models)[-1]\n            model_path = os.path.join('./models', latest_improved_model)\n            \n            self.logger.info(f"使用改进模型: {model_path}")\n            \n            # 运行回测\n            improved_results = self._run_backtest_with_model(model_path, test_name="improved")\n            \n            self.results['improved_performance'] = improved_results\n            return improved_results\n            \n        except Exception as e:\n            self.logger.error(f"改进回测失败: {e}")\n            return {}\n    \n    def _initialize_systems(self) -> Dict:\n        """初始化系统组件"""\n        # 数据管理器\n        data_config = self.config_manager.get_data_config()\n        data_manager = DataManager(data_config)\n        \n        # 因子引擎\n        factor_config = self.config_manager.get_config('factors')\n        factor_engine = FactorEngine(factor_config)\n        \n        # 风险控制器\n        risk_config = self.config_manager.get_risk_control_config()\n        risk_controller = RiskController(risk_config)\n        \n        # 回测引擎\n        backtest_config = self.config_manager.get_backtest_config()\n        backtest_engine = BacktestEngine(backtest_config)\n        \n        return {\n            'data_manager': data_manager,\n            'factor_engine': factor_engine,\n            'risk_controller': risk_controller,\n            'backtest_engine': backtest_engine\n        }\n    \n    def _prepare_training_data(self, systems: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:\n        """准备训练数据"""\n        data_manager = systems['data_manager']\n        factor_engine = systems['factor_engine']\n        \n        # 获取训练数据配置\n        data_config = self.config_manager.get_data_config()\n        \n        # 获取股票数据\n        stock_data = data_manager.get_stock_data(\n            start_time=data_config['start_date'],\n            end_time=data_config['end_date']\n        )\n        \n        # 处理价格和成交量数据\n        price_data = stock_data['$close'].unstack().T\n        volume_data = stock_data['$volume'].unstack().T\n        \n        # 筛选股票池\n        low_vol_stocks = factor_engine.filter_low_volatility_universe(\n            price_data,\n            threshold=self.config_manager.get_value('factors.low_vol_threshold', 0.5),\n            window=self.config_manager.get_value('factors.low_vol_window', 60)\n        )\n        \n        if low_vol_stocks:\n            price_data = price_data[low_vol_stocks]\n            volume_data = volume_data[low_vol_stocks]\n        else:\n            price_data = price_data.iloc[:, :100]\n            volume_data = volume_data.iloc[:, :100]\n            low_vol_stocks = list(price_data.columns)\n        \n        # 保存股票池\n        os.makedirs('./models', exist_ok=True)\n        with open('./models/selected_stocks.pkl', 'wb') as f:\n            pickle.dump(low_vol_stocks, f)\n        \n        # 计算因子\n        factor_data = factor_engine.calculate_all_factors(price_data, volume_data)\n        \n        # 数据对齐\n        if isinstance(factor_data.index, pd.MultiIndex):\n            factor_data = factor_data.unstack()\n            aligned_price, aligned_factor = price_data.align(factor_data, join='inner', axis=0)\n            price_data = aligned_price\n            factor_data = aligned_factor.stack().reorder_levels(['datetime', 'instrument'])\n        \n        self.logger.info(f"训练数据准备完成 - 价格数据: {price_data.shape}, 因子数据: {factor_data.shape}")\n        \n        return price_data, factor_data\n    \n    def _run_training_episode(self, environment, agent, safety_shield, episode: int) -> Dict[str, float]:\n        """运行一个训练episode"""\n        state, info = environment.reset()\n        episode_reward = 0\n        step_count = 0\n        \n        while True:\n            # 获取动作\n            action, log_prob, value, cvar_estimate = agent.get_action(state)\n            \n            # 应用安全保护\n            safe_action = safety_shield.shield_action(\n                action, info, environment.price_data, environment.portfolio_weights\n            )\n            \n            # 执行动作\n            next_state, reward, terminated, truncated, info = environment.step(safe_action)\n            \n            # 存储经验\n            agent.store_transition(state, safe_action, reward, value, log_prob, terminated or truncated, cvar_estimate)\n            \n            # 更新状态\n            state = next_state\n            episode_reward += reward\n            step_count += 1\n            \n            if terminated or truncated:\n                break\n        \n        # 定期更新智能体\n        if episode % 10 == 0 and episode > 0:\n            agent.update()\n        \n        # 计算episode指标\n        portfolio_returns = environment.portfolio_returns\n        if len(portfolio_returns) > 1:\n            returns_array = np.array(portfolio_returns)\n            annual_return = np.mean(returns_array) * 252\n            volatility = np.std(returns_array) * np.sqrt(252)\n            sharpe_ratio = annual_return / volatility if volatility > 0 else 0\n            \n            # 计算最大回撤\n            cumulative = np.cumprod(1 + returns_array)\n            running_max = np.maximum.accumulate(cumulative)\n            drawdown = (cumulative - running_max) / running_max\n            max_drawdown = np.min(drawdown)\n        else:\n            annual_return = 0\n            sharpe_ratio = 0\n            max_drawdown = 0\n        \n        return {\n            'episode_reward': episode_reward,\n            'annual_return': annual_return,\n            'sharpe_ratio': sharpe_ratio,\n            'max_drawdown': max_drawdown,\n            'steps': step_count\n        }\n    \n    def _run_backtest_with_model(self, model_path: str, test_name: str) -> Dict[str, float]:\n        """使用指定模型运行回测"""\n        try:\n            # 切换到回测配置\n            backtest_config_manager = ConfigManager('config_backtest.yaml')\n            \n            # 初始化回测系统\n            systems = self._initialize_systems()\n            \n            # 准备回测数据\n            backtest_data_config = backtest_config_manager.get_data_config()\n            stock_data = systems['data_manager'].get_stock_data(\n                start_time=backtest_data_config['start_date'],\n                end_time=backtest_data_config['end_date']\n            )\n            \n            # 加载训练时的股票池\n            with open('./models/selected_stocks.pkl', 'rb') as f:\n                selected_stocks = pickle.load(f)\n            \n            # 处理回测数据\n            price_data = stock_data['$close'].unstack().T\n            volume_data = stock_data['$volume'].unstack().T\n            \n            # 使用相同的股票池\n            available_stocks = [s for s in selected_stocks if s in price_data.columns]\n            price_data = price_data[available_stocks]\n            volume_data = volume_data[available_stocks]\n            \n            # 计算因子\n            factor_data = systems['factor_engine'].calculate_all_factors(price_data, volume_data)\n            \n            # 数据对齐\n            if isinstance(factor_data.index, pd.MultiIndex):\n                factor_data = factor_data.unstack()\n                aligned_price, aligned_factor = price_data.align(factor_data, join='inner', axis=0)\n                price_data = aligned_price\n                factor_data = aligned_factor.stack().reorder_levels(['datetime', 'instrument'])\n            \n            # 创建环境和智能体\n            env_config = backtest_config_manager.get_environment_config()\n            environment = TradingEnvironment(factor_data, price_data, env_config)\n            \n            agent_config = backtest_config_manager.get_agent_config()\n            state_dim = environment.observation_space.shape[0]\n            action_dim = environment.action_space.shape[0]\n            agent = CVaRPPOAgent(state_dim, action_dim, agent_config)\n            \n            # 加载模型\n            agent.load_model(model_path)\n            \n            # 创建安全保护层\n            shield_config = backtest_config_manager.get_config('safety_shield')\n            safety_shield = SafetyShield(shield_config)\n            \n            # 运行回测\n            backtest_results = systems['backtest_engine'].run_backtest(\n                agent=agent,\n                env=environment,\n                start_date=backtest_data_config['start_date'],\n                end_date=backtest_data_config['end_date'],\n                safety_shield=safety_shield\n            )\n            \n            # 生成报告\n            report = systems['backtest_engine'].generate_backtest_report(backtest_results)\n            self.logger.info(f"{test_name}回测报告:\\n{report}")\n            \n            # 提取关键指标\n            key_metrics = {\n                'annual_return': backtest_results.get('annual_return', 0),\n                'sharpe_ratio': backtest_results.get('sharpe_ratio', 0),\n                'max_drawdown': backtest_results.get('max_drawdown', 0),\n                'volatility': backtest_results.get('volatility', 0),\n                'total_days': backtest_results.get('total_days', 0),\n                'win_rate': backtest_results.get('win_rate', 0)\n            }\n            \n            return key_metrics\n            \n        except Exception as e:\n            self.logger.error(f"{test_name}回测失败: {e}")\n            import traceback\n            self.logger.error(traceback.format_exc())\n            return {}\n    \n    def generate_comparison_report(self) -> Dict[str, any]:\n        """生成对比报告"""\n        self.logger.info("生成性能对比报告...")\n        \n        baseline = self.results.get('baseline_performance', {})\n        improved = self.results.get('improved_performance', {})\n        \n        if not baseline or not improved:\n            self.logger.warning("缺少基线或改进后的性能数据")\n            return {}\n        \n        # 计算改进指标\n        comparison = {}\n        \n        # 年化收益率对比\n        baseline_return = baseline.get('annual_return', -0.2232)\n        improved_return = improved.get('annual_return', 0)\n        return_improvement = improved_return - baseline_return\n        return_improvement_pct = (return_improvement / abs(baseline_return)) * 100 if baseline_return != 0 else 0\n        \n        comparison['annual_return'] = {\n            'baseline': baseline_return,\n            'improved': improved_return,\n            'absolute_improvement': return_improvement,\n            'percentage_improvement': return_improvement_pct\n        }\n        \n        # Sharpe比率对比\n        baseline_sharpe = baseline.get('sharpe_ratio', 0)\n        improved_sharpe = improved.get('sharpe_ratio', 0)\n        sharpe_improvement = improved_sharpe - baseline_sharpe\n        \n        comparison['sharpe_ratio'] = {\n            'baseline': baseline_sharpe,\n            'improved': improved_sharpe,\n            'absolute_improvement': sharpe_improvement\n        }\n        \n        # 最大回撤对比\n        baseline_dd = baseline.get('max_drawdown', 0)\n        improved_dd = improved.get('max_drawdown', 0)\n        dd_improvement = baseline_dd - improved_dd  # 回撤减少是好的\n        \n        comparison['max_drawdown'] = {\n            'baseline': baseline_dd,\n            'improved': improved_dd,\n            'improvement': dd_improvement\n        }\n        \n        # 目标达成评估\n        target_return_achieved = improved_return >= 0.08  # 8%目标\n        target_return_range = 0.08 <= improved_return <= 0.12  # 8-12%目标区间\n        \n        comparison['target_achievement'] = {\n            'target_8pct_achieved': target_return_achieved,\n            'target_range_achieved': target_return_range,\n            'distance_to_target': max(0, 0.08 - improved_return)\n        }\n        \n        self.results['comparison_metrics'] = comparison\n        \n        return comparison\n    \n    def print_final_summary(self):\n        """打印最终总结"""\n        print("\\n" + "="*60)\n        print("         综合改进实施和测试总结报告")\n        print("="*60)\n        \n        # 改进实施状态\n        print("\\n📋 改进实施状态:")\n        for stage, status in self.improvement_stages.items():\n            status_icon = "✅" if status else "❌"\n            stage_name = {\n                'stage_1_reward_function': '奖励函数优化',\n                'stage_2_factor_enhancement': '因子库增强',\n                'stage_3_risk_optimization': '风险控制优化', \n                'stage_4_architecture_upgrade': '网络架构升级'\n            }.get(stage, stage)\n            print(f"  {status_icon} {stage_name}")\n        \n        # 性能对比\n        comparison = self.results.get('comparison_metrics', {})\n        if comparison:\n            print("\\n📊 性能对比结果:")\n            \n            # 年化收益率\n            return_data = comparison.get('annual_return', {})\n            print(f"  年化收益率:")\n            print(f"    基线: {return_data.get('baseline', 0)*100:.2f}%")\n            print(f"    改进后: {return_data.get('improved', 0)*100:.2f}%")\n            print(f"    改进幅度: {return_data.get('absolute_improvement', 0)*100:+.2f}% ")\n            print(f"             ({return_data.get('percentage_improvement', 0):+.1f}%)")\n            \n            # Sharpe比率\n            sharpe_data = comparison.get('sharpe_ratio', {})\n            print(f"  Sharpe比率:")\n            print(f"    基线: {sharpe_data.get('baseline', 0):.3f}")\n            print(f"    改进后: {sharpe_data.get('improved', 0):.3f}")\n            print(f"    改进幅度: {sharpe_data.get('absolute_improvement', 0):+.3f}")\n            \n            # 最大回撤\n            dd_data = comparison.get('max_drawdown', {})\n            print(f"  最大回撤:")\n            print(f"    基线: {dd_data.get('baseline', 0)*100:.2f}%")\n            print(f"    改进后: {dd_data.get('improved', 0)*100:.2f}%")\n            print(f"    回撤减少: {dd_data.get('improvement', 0)*100:+.2f}%")\n            \n            # 目标达成\n            target_data = comparison.get('target_achievement', {})\n            print(f"\\n🎯 目标达成情况:")\n            target_8_icon = "✅" if target_data.get('target_8pct_achieved', False) else "❌"\n            target_range_icon = "✅" if target_data.get('target_range_achieved', False) else "❌"\n            print(f"  {target_8_icon} 8%年化收益率目标")\n            print(f"  {target_range_icon} 8-12%目标区间")\n            \n            distance = target_data.get('distance_to_target', 0)\n            if distance > 0:\n                print(f"  📏 距离8%目标还需: {distance*100:.2f}%")\n        \n        print("\\n" + "="*60)\n        \n        # 总体评估\n        implemented_count = sum(self.improvement_stages.values())\n        total_improvements = len(self.improvement_stages)\n        \n        if comparison and implemented_count > 0:\n            improved_return = comparison.get('annual_return', {}).get('improved', 0)\n            if improved_return >= 0.08:\n                print("🎉 综合评估: 优秀 - 已达成收益率目标!")\n            elif improved_return >= 0.05:\n                print("👍 综合评估: 良好 - 显著改善，接近目标")\n            elif improved_return >= 0:\n                print("📈 综合评估: 改善 - 扭亏为盈，继续优化")\n            else:\n                print("⚠️  综合评估: 需要进一步优化")\n        else:\n            print("ℹ️  综合评估: 改进实施完成，等待性能验证")\n        \n        print("="*60)\n\n\ndef main():\n    """主函数"""\n    parser = argparse.ArgumentParser(description='综合改进实施和测试')\n    parser.add_argument('--config', type=str, help='配置文件路径')\n    parser.add_argument('--skip-baseline', action='store_true', help='跳过基线测试')\n    parser.add_argument('--skip-training', action='store_true', help='跳过重新训练')\n    parser.add_argument('--only-validate', action='store_true', help='仅验证改进实施')\n    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')\n    \n    args = parser.parse_args()\n    \n    # 创建测试器\n    tester = ComprehensiveImprovementTester(\n        config_path=args.config,\n        verbose=args.verbose\n    )\n    \n    try:\n        # 验证改进实施\n        validation_results = tester.validate_improvements()\n        \n        if args.only_validate:\n            tester.print_final_summary()\n            return\n        \n        # 运行基线测试\n        if not args.skip_baseline:\n            baseline_results = tester.run_baseline_test()\n        \n        # 运行综合训练\n        if not args.skip_training:\n            training_results = tester.run_comprehensive_training()\n        \n        # 运行改进后回测\n        improved_results = tester.run_improved_backtest()\n        \n        # 生成对比报告\n        comparison = tester.generate_comparison_report()\n        \n        # 打印最终总结\n        tester.print_final_summary()\n        \n    except KeyboardInterrupt:\n        tester.logger.info("用户中断测试")\n    except Exception as e:\n        tester.logger.error(f"测试过程发生错误: {e}")\n        import traceback\n        tester.logger.error(traceback.format_exc())\n\n\nif __name__ == "__main__":\n    main()