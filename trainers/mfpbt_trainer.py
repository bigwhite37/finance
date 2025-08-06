"""
MF-PBT多频率训练控制器
防止策略坍塌的关键训练机制
参考claude.md中的MFPBTTrainer实现
"""
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Callable, Tuple
import logging
import time
import copy
from collections import deque, defaultdict
import random
import pickle
from pathlib import Path

from models.expert_policy import ExpertPolicy, ExpertPopulation
from models.meta_router import MetaRouter
from replay.shared_buffer import SharedReplayBuffer

logger = logging.getLogger(__name__)


class DiversityTracker:
    """
    多样性跟踪器
    监控专家策略的多样性指标
    """
    
    def __init__(self, history_length: int = 1000):
        self.history_length = history_length
        self.action_histories = defaultdict(lambda: deque(maxlen=history_length))
        self.performance_histories = defaultdict(lambda: deque(maxlen=history_length))
        
    def update(self, expert_id: int, actions: np.ndarray, performance: float):
        """更新专家的行为和性能历史"""
        self.action_histories[expert_id].extend(actions.flatten())
        self.performance_histories[expert_id].append(performance)
    
    def compute_mapd(self, experts: List[ExpertPolicy]) -> float:
        """
        计算平均动作成对距离 (Mean Action Pairwise Distance)
        用于衡量专家策略的行为多样性
        """
        if len(experts) < 2:
            return 0.0
        
        # 收集每个专家的最近动作
        expert_actions = []
        for expert in experts:
            actions = list(self.action_histories[expert.expert_id])[-100:]  # 最近100个动作
            if len(actions) > 0:
                expert_actions.append(np.array(actions))
            else:
                # 如果没有历史动作，使用随机动作
                expert_actions.append(np.random.randn(100))
        
        if len(expert_actions) < 2:
            return 0.0
        
        # 计算两两之间的平均距离
        total_distance = 0.0
        count = 0
        
        for i in range(len(expert_actions)):
            for j in range(i + 1, len(expert_actions)):
                # 确保两个动作序列长度相同
                min_len = min(len(expert_actions[i]), len(expert_actions[j]))
                if min_len > 0:
                    actions_i = expert_actions[i][-min_len:]
                    actions_j = expert_actions[j][-min_len:]
                    
                    # 计算欧氏距离
                    distance = np.mean(np.linalg.norm(
                        actions_i.reshape(-1, 1) - actions_j.reshape(-1, 1), axis=0
                    ))
                    total_distance += distance
                    count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def compute_performance_diversity(self, experts: List[ExpertPolicy]) -> float:
        """计算性能多样性"""
        performances = []
        for expert in experts:
            recent_perf = list(self.performance_histories[expert.expert_id])[-20:]  # 最近20次性能
            if recent_perf:
                performances.append(np.mean(recent_perf))
            else:
                performances.append(0.0)
        
        return np.std(performances) if len(performances) > 1 else 0.0
    
    def get_diversity_metrics(self, experts: List[ExpertPolicy]) -> Dict[str, float]:
        """获取所有多样性指标"""
        return {
            'mapd': self.compute_mapd(experts),
            'performance_diversity': self.compute_performance_diversity(experts),
            'skill_diversity': self._compute_skill_diversity(experts),
            'selection_entropy': self._compute_selection_entropy(experts)
        }
    
    def _compute_skill_diversity(self, experts: List[ExpertPolicy]) -> float:
        """计算技能向量多样性"""
        if len(experts) < 2:
            return 0.0
        
        skill_vectors = [expert.skill_z.cpu().numpy() for expert in experts]
        
        total_distance = 0.0
        count = 0
        
        for i in range(len(skill_vectors)):
            for j in range(i + 1, len(skill_vectors)):
                distance = np.linalg.norm(skill_vectors[i] - skill_vectors[j])
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def _compute_selection_entropy(self, experts: List[ExpertPolicy]) -> float:
        """计算专家选择的熵"""
        # 这里简化处理，实际应该从元路由器获取选择历史
        n_experts = len(experts)
        uniform_prob = 1.0 / n_experts
        return -n_experts * uniform_prob * np.log(uniform_prob + 1e-8)


class PopulationEvolver:
    """
    种群进化器
    实现基于性能的专家淘汰和变异
    """
    
    def __init__(self, 
                 mutation_rate: float = 0.1,
                 elite_ratio: float = 0.3,
                 crossover_prob: float = 0.7):
        
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self.crossover_prob = crossover_prob
        
    def evolve_population(self, experts: List[ExpertPolicy]) -> List[int]:
        """
        进化种群，返回被替换的专家索引
        
        Args:
            experts: 专家列表
            
        Returns:
            被替换的专家索引列表
        """
        if len(experts) < 3:
            return []
        
        # 获取专家性能
        performances = [expert.get_performance_metrics()['avg_reward'] for expert in experts]
        
        # 排序专家（按性能降序）
        sorted_indices = np.argsort(performances)[::-1]
        
        # 确定精英数量
        n_elites = max(1, int(len(experts) * self.elite_ratio))
        elite_indices = sorted_indices[:n_elites]
        weak_indices = sorted_indices[n_elites:]
        
        replaced_indices = []
        
        # 替换表现最差的专家
        for weak_idx in weak_indices[-1:]:  # 只替换最差的一个
            if performances[weak_idx] < np.mean(performances) * 0.5:  # 性能过低
                # 选择一个精英进行变异
                parent_idx = np.random.choice(elite_indices)
                self._mutate_expert(experts[weak_idx], experts[parent_idx])
                replaced_indices.append(weak_idx)
                logger.info(f"专家 {weak_idx} 被替换（基于专家 {parent_idx} 变异）")
        
        return replaced_indices
    
    def _mutate_expert(self, target_expert: ExpertPolicy, parent_expert: ExpertPolicy):
        """变异专家"""
        # 复制父专家的技能向量并添加噪声
        parent_skill = parent_expert.skill_z.cpu().numpy()
        noise = np.random.normal(0, self.mutation_rate, parent_skill.shape)
        new_skill = parent_skill + noise
        
        # 更新目标专家的技能向量
        target_expert.skill_z = torch.tensor(
            new_skill, dtype=torch.float32, device=target_expert.device
        )
        
        # 重置专家的经验和性能历史
        target_expert.performance_history.clear()
        target_expert.mi_loss_history.clear()
        target_expert.episode_count = 0
        target_expert.cumulative_reward = 0.0
        
        logger.info(f"专家 {target_expert.expert_id} 技能向量已变异")


class MFPBTTrainer:
    """
    多频率基于种群的训练器
    实现分层RL系统的协调训练，防止策略坍塌
    """
    
    def __init__(self,
                 expert_population: ExpertPopulation,
                 meta_router: MetaRouter,
                 shared_buffer: SharedReplayBuffer,
                 population_size: int = 5,
                 diversity_threshold: float = 0.25,
                 training_config: Optional[Dict] = None):
        
        self.expert_population = expert_population
        self.meta_router = meta_router
        self.shared_buffer = shared_buffer
        self.population_size = population_size
        self.diversity_threshold = diversity_threshold
        
        # 训练配置
        self.config = training_config or {
            'router': {'freq': 10, 'lr_range': (1e-4, 1e-3), 'steps': 10000},
            'experts': {'freq': 5, 'lr_range': (3e-4, 3e-3), 'steps': 5000},
            'population': {'freq': 20, 'mutation_rate': 0.1},
            'buffer_update': {'freq': 1},
            'evaluation': {'freq': 50, 'episodes': 10}
        }
        
        # 组件
        self.diversity_tracker = DiversityTracker()
        self.population_evolver = PopulationEvolver(
            mutation_rate=self.config['population']['mutation_rate']
        )
        
        # 训练状态
        self.global_step = 0
        self.training_history = []
        self.best_performance = float('-inf')
        self.best_models = {}
        
        # 性能监控
        self.episode_rewards = deque(maxlen=1000)
        self.step_times = deque(maxlen=1000)
        
        logger.info("MF-PBT训练器初始化完成")
    
    def train(self, 
              total_steps: int,
              save_path: Optional[str] = None,
              eval_callback: Optional[Callable] = None,
              checkpoint_freq: int = 10000) -> Dict[str, Any]:
        """
        执行MF-PBT训练
        
        Args:
            total_steps: 总训练步数
            save_path: 模型保存路径
            eval_callback: 评估回调函数
            checkpoint_freq: 检查点保存频率
            
        Returns:
            训练结果统计
        """
        logger.info(f"开始MF-PBT训练，总步数: {total_steps}")
        
        start_time = time.time()
        
        for step in range(total_steps):
            step_start_time = time.time()
            self.global_step = step
            
            # 1. 缓冲区更新（高频）
            if step % self.config['buffer_update']['freq'] == 0:
                self._update_shared_buffer()
            
            # 2. 专家更新（中频）
            if step % self.config['experts']['freq'] == 0:
                expert_results = self._update_experts()
                self._update_diversity_tracker(expert_results)
            
            # 3. 路由器更新（低频）
            if step % self.config['router']['freq'] == 0:
                router_results = self._update_router()
            
            # 4. 种群进化（最低频）
            if step % self.config['population']['freq'] == 0:
                self._evolve_population()
            
            # 5. 多样性检查和注入
            if step % (self.config['population']['freq'] // 2) == 0:
                self._check_and_inject_diversity()
            
            # 6. 评估（可选）
            if step % self.config['evaluation']['freq'] == 0 and eval_callback:
                eval_results = eval_callback(self)
                logger.info(f"步骤 {step} 评估结果: {eval_results}")
            
            # 7. 保存检查点
            if save_path and step % checkpoint_freq == 0 and step > 0:
                self._save_checkpoint(save_path, step)
            
            # 记录性能
            step_time = time.time() - step_start_time
            self.step_times.append(step_time)
            
            # 定期输出训练状态
            if step % 100 == 0:
                self._log_training_status(step, total_steps)
        
        # 训练完成
        total_time = time.time() - start_time
        final_results = self._get_final_results(total_time)
        
        # 保存最终模型
        if save_path:
            self._save_final_models(save_path)
        
        logger.info(f"MF-PBT训练完成，耗时: {total_time:.2f}秒")
        
        return final_results
    
    def _update_shared_buffer(self):
        """更新共享经验池"""
        # 这里简化处理，实际应该从环境交互中收集经验
        # 在实际应用中，这会在环境交互循环中调用
        pass
    
    def _update_experts(self) -> Dict[str, Any]:
        """更新专家策略"""
        logger.debug("更新专家策略...")
        
        expert_results = {}
        
        # 交替冻结：更新专家时冻结路由器
        self._freeze_router()
        
        try:
            # 更新所有专家
            results = self.expert_population.train_all_experts(
                total_timesteps=self.config['experts']['steps']
            )
            expert_results.update(results)
            
        finally:
            # 解冻路由器
            self._unfreeze_router()
        
        return expert_results
    
    def _update_router(self) -> Dict[str, Any]:
        """更新元路由器"""
        logger.debug("更新元路由器...")
        
        # 交替冻结：更新路由器时冻结专家
        self._freeze_experts()
        
        try:
            # 训练路由器
            router_results = self.meta_router.train(
                total_timesteps=self.config['router']['steps']
            )
            
            # 动态调整KL权重
            self._adjust_kl_lambda()
            
            return router_results
            
        finally:
            # 解冻专家
            self._unfreeze_experts()
    
    def _evolve_population(self):
        """执行种群进化"""
        logger.debug("执行种群进化...")
        
        # 基于性能进化种群
        replaced_indices = self.population_evolver.evolve_population(
            self.expert_population.experts
        )
        
        if replaced_indices:
            logger.info(f"种群进化完成，替换了 {len(replaced_indices)} 个专家")
    
    def _check_and_inject_diversity(self):
        """检查多样性并注入新的多样性"""
        diversity_metrics = self.diversity_tracker.get_diversity_metrics(
            self.expert_population.experts
        )
        
        mapd = diversity_metrics['mapd']
        
        if mapd < self.diversity_threshold:
            logger.warning(f"多样性过低 (MAPD: {mapd:.4f} < {self.diversity_threshold})")
            
            # 注入多样性：重置表现最差的专家
            worst_idx = self.expert_population.reset_worst_expert()
            logger.info(f"注入多样性：重置专家 {worst_idx}")
        
        # 记录多样性指标
        self._log_diversity_metrics(diversity_metrics)
    
    def _update_diversity_tracker(self, expert_results: Dict[str, Any]):
        """更新多样性跟踪器"""
        for expert in self.expert_population.experts:
            # 获取专家的最近动作和性能
            # 这里简化处理，实际应该从专家的训练历史中获取
            dummy_actions = np.random.randn(10, 5)  # 假设的动作数据
            performance = expert.get_performance_metrics()['avg_reward']
            
            self.diversity_tracker.update(expert.expert_id, dummy_actions, performance)
    
    def _freeze_router(self):
        """冻结路由器参数"""
        if hasattr(self.meta_router.policy, 'policy'):
            for param in self.meta_router.policy.policy.parameters():
                param.requires_grad = False
    
    def _unfreeze_router(self):
        """解冻路由器参数"""
        if hasattr(self.meta_router.policy, 'policy'):
            for param in self.meta_router.policy.policy.parameters():
                param.requires_grad = True
    
    def _freeze_experts(self):
        """冻结专家参数"""
        for expert in self.expert_population.experts:
            if hasattr(expert.policy, 'policy'):
                for param in expert.policy.policy.parameters():
                    param.requires_grad = False
    
    def _unfreeze_experts(self):
        """解冻专家参数"""
        for expert in self.expert_population.experts:
            if hasattr(expert.policy, 'policy'):
                for param in expert.policy.policy.parameters():
                    param.requires_grad = True
    
    def _adjust_kl_lambda(self):
        """动态调整KL散度权重"""
        # 基于多样性指标动态调整
        diversity_metrics = self.diversity_tracker.get_diversity_metrics(
            self.expert_population.experts
        )
        
        current_mapd = diversity_metrics['mapd']
        
        if current_mapd < self.diversity_threshold:
            # 多样性过低，增加KL权重
            new_lambda = min(0.5, self.meta_router.kl_lambda * 1.1)
        elif current_mapd > self.diversity_threshold * 2:
            # 多样性过高，降低KL权重
            new_lambda = max(0.05, self.meta_router.kl_lambda * 0.9)
        else:
            # 多样性适中，保持当前权重
            new_lambda = self.meta_router.kl_lambda
        
        if abs(new_lambda - self.meta_router.kl_lambda) > 0.01:
            self.meta_router.update_kl_lambda(new_lambda)
            logger.info(f"KL权重调整: {self.meta_router.kl_lambda:.4f} -> {new_lambda:.4f}")
    
    def _log_training_status(self, step: int, total_steps: int):
        """输出训练状态"""
        progress = step / total_steps * 100
        avg_step_time = np.mean(self.step_times) if self.step_times else 0
        eta = (total_steps - step) * avg_step_time
        
        # 获取性能指标
        population_metrics = self.expert_population.get_population_metrics()
        router_stats = self.meta_router.get_selection_statistics()
        
        logger.info(f"训练进度: {progress:.1f}% ({step}/{total_steps})")
        logger.info(f"ETA: {eta:.0f}秒, 平均步骤时间: {avg_step_time:.4f}秒")
        logger.info(f"种群平均奖励: {population_metrics.get('avg_population_reward', 0):.6f}")
        logger.info(f"种群多样性: {population_metrics.get('diversity_score', 0):.4f}")
        if router_stats:
            logger.info(f"路由器选择熵: {router_stats.get('normalized_entropy', 0):.4f}")
    
    def _log_diversity_metrics(self, diversity_metrics: Dict[str, float]):
        """记录多样性指标"""
        logger.debug(f"多样性指标: {diversity_metrics}")
        
        # 记录到训练历史
        self.training_history.append({
            'step': self.global_step,
            'diversity_metrics': diversity_metrics,
            'timestamp': time.time()
        })
    
    def _save_checkpoint(self, save_path: str, step: int):
        """保存训练检查点"""
        checkpoint_dir = Path(save_path) / f"checkpoint_{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存专家模型
        for i, expert in enumerate(self.expert_population.experts):
            expert.save_model(str(checkpoint_dir / f"expert_{i}.pt"))
        
        # 保存路由器
        self.meta_router.save_model(str(checkpoint_dir / "router"))
        
        # 保存共享缓冲区
        self.shared_buffer.save(str(checkpoint_dir / "shared_buffer.pkl"))
        
        # 保存训练状态
        training_state = {
            'global_step': self.global_step,
            'training_history': self.training_history,
            'config': self.config,
            'diversity_threshold': self.diversity_threshold
        }
        
        with open(checkpoint_dir / "training_state.pkl", 'wb') as f:
            pickle.dump(training_state, f)
        
        logger.info(f"检查点已保存到: {checkpoint_dir}")
    
    def _save_final_models(self, save_path: str):
        """保存最终模型"""
        final_dir = Path(save_path) / "final_models"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存最佳模型
        self._save_checkpoint(str(final_dir), self.global_step)
        
        # 保存训练摘要
        summary = self._get_training_summary()
        with open(final_dir / "training_summary.json", 'w') as f:
            import json
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"最终模型已保存到: {final_dir}")
    
    def _get_final_results(self, total_time: float) -> Dict[str, Any]:
        """获取最终训练结果"""
        # 专家种群指标
        population_metrics = self.expert_population.get_population_metrics()
        
        # 路由器指标
        router_stats = self.meta_router.get_selection_statistics()
        
        # 多样性指标
        diversity_metrics = self.diversity_tracker.get_diversity_metrics(
            self.expert_population.experts
        )
        
        # 训练效率指标
        efficiency_metrics = {
            'total_training_time': total_time,
            'avg_step_time': np.mean(self.step_times) if self.step_times else 0,
            'total_steps': self.global_step,
            'steps_per_second': self.global_step / total_time if total_time > 0 else 0
        }
        
        return {
            'population_metrics': population_metrics,
            'router_stats': router_stats,
            'diversity_metrics': diversity_metrics,
            'efficiency_metrics': efficiency_metrics,
            'training_history': self.training_history[-100:]  # 最近100条记录
        }
    
    def _get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        return {
            'config': self.config,
            'total_steps': self.global_step,
            'population_size': self.population_size,
            'diversity_threshold': self.diversity_threshold,
            'final_performance': self.expert_population.get_population_metrics(),
            'best_performance': self.best_performance
        }
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载训练检查点"""
        checkpoint_dir = Path(checkpoint_path)
        
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"检查点目录不存在: {checkpoint_dir}")
        
        # 加载专家模型
        for i, expert in enumerate(self.expert_population.experts):
            expert_path = checkpoint_dir / f"expert_{i}.pt"
            if expert_path.exists():
                expert.load_model(str(expert_path))
        
        # 加载路由器
        router_path = checkpoint_dir / "router"
        if router_path.exists():
            self.meta_router.load_model(str(router_path))
        
        # 加载共享缓冲区
        buffer_path = checkpoint_dir / "shared_buffer.pkl"
        if buffer_path.exists():
            self.shared_buffer.load(str(buffer_path))
        
        # 加载训练状态
        state_path = checkpoint_dir / "training_state.pkl"
        if state_path.exists():
            with open(state_path, 'rb') as f:
                training_state = pickle.load(f)
                self.global_step = training_state['global_step']
                self.training_history = training_state['training_history']
                self.config = training_state['config']
                self.diversity_threshold = training_state['diversity_threshold']
        
        logger.info(f"检查点已加载: {checkpoint_dir}")


if __name__ == "__main__":
    # 测试MF-PBT训练器
    print("测试MF-PBT训练器...")
    
    # 这里需要创建完整的测试环境
    # 由于依赖较多，实际测试需要在完整的项目环境中进行
    
    print("MF-PBT训练器模块定义完成")