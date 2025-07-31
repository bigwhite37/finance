"""
金丝雀部署系统集成测试
测试完整的金丝雀部署流程，包括模型部署、A/B测试、性能监控和自动回滚
"""
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.rl_trading_system.deployment.canary_deployment import (
    CanaryDeployment,
    ABTestFramework,
    ModelPerformanceComparator,
    DeploymentSafetyController,
    RollbackManager,
    PerformanceMetrics,
    DeploymentConfig,
    DeploymentStatus
)
from src.rl_trading_system.deployment.model_version_manager import (
    ModelVersionManager,
    ModelMetadata,
    ModelStatus
)


class DummyModel:
    """可序列化的虚拟模型类"""
    def __init__(self, version, weights=None):
        self.version = version
        self.weights = weights or np.random.random((10, 10))
    
    def predict(self, x):
        return np.random.random(3)


class TestCanaryDeploymentIntegration:
    """金丝雀部署系统集成测试"""
    
    @pytest.fixture
    def mock_models(self):
        """创建模拟模型"""
        baseline_model = Mock()
        baseline_model.predict = Mock(return_value=np.array([0.2, 0.3, 0.5]))
        baseline_model.version = "v1.0.0"
        
        canary_model = Mock()
        canary_model.predict = Mock(return_value=np.array([0.1, 0.2, 0.7]))
        canary_model.version = "v1.1.0"
        
        return baseline_model, canary_model
    
    @pytest.fixture
    def serializable_models(self):
        """创建可序列化的模拟模型"""
        baseline_model = DummyModel("v1.0.0")
        canary_model = DummyModel("v1.1.0")
        
        return baseline_model, canary_model
    
    @pytest.fixture
    def deployment_config(self):
        """创建部署配置"""
        return DeploymentConfig(
            canary_percentage=10.0,
            evaluation_period=300,  # 5分钟
            success_threshold=0.90,
            error_threshold=0.10,
            performance_threshold=0.05,
            rollback_threshold=0.15,
            max_canary_duration=1800  # 30分钟
        )
    
    def test_successful_canary_deployment_flow(self, mock_models, deployment_config):
        """测试成功的金丝雀部署流程"""
        baseline_model, canary_model = mock_models
        
        # 1. 创建金丝雀部署
        deployment = CanaryDeployment(
            canary_model=canary_model,
            baseline_model=baseline_model,
            config=deployment_config
        )
        
        # 2. 启动部署
        deployment.start_deployment()
        assert deployment.status == DeploymentStatus.ACTIVE
        assert deployment.traffic_percentage == 10.0
        
        # 3. 模拟良好的性能指标
        for i in range(5):
            metrics = PerformanceMetrics(
                success_rate=0.95 + i * 0.001,
                error_rate=0.02 - i * 0.001,
                avg_response_time=0.1 + i * 0.001,
                throughput=1000 + i * 10,
                accuracy=0.92 + i * 0.001,
                precision=0.90 + i * 0.001,
                recall=0.88 + i * 0.001,
                f1_score=0.89 + i * 0.001
            )
            deployment.update_canary_metrics(metrics)
        
        # 4. 验证成功标准
        assert deployment.evaluate_success_criteria() == True
        assert deployment.should_trigger_rollback() == False
        
        # 5. 逐步增加流量
        deployment.increase_traffic(20.0)
        assert deployment.traffic_percentage == 30.0
        
        deployment.increase_traffic(30.0)
        assert deployment.traffic_percentage == 60.0
        
        deployment.increase_traffic(40.0)
        assert deployment.traffic_percentage == 100.0
        
        # 6. 完成部署
        deployment.complete_deployment()
        assert deployment.status == DeploymentStatus.COMPLETED
        assert deployment.traffic_percentage == 100.0
    
    def test_failed_canary_deployment_with_rollback(self, mock_models, deployment_config):
        """测试失败的金丝雀部署和自动回滚"""
        baseline_model, canary_model = mock_models
        
        # 1. 创建金丝雀部署
        deployment = CanaryDeployment(
            canary_model=canary_model,
            baseline_model=baseline_model,
            config=deployment_config
        )
        
        # 2. 启动部署
        deployment.start_deployment()
        
        # 3. 模拟初期良好的性能
        for i in range(3):
            metrics = PerformanceMetrics(
                success_rate=0.92,
                error_rate=0.05,
                avg_response_time=0.12,
                throughput=950,
                accuracy=0.90,
                precision=0.88,
                recall=0.86,
                f1_score=0.87
            )
            deployment.update_canary_metrics(metrics)
        
        # 4. 增加流量
        deployment.increase_traffic(20.0)
        assert deployment.traffic_percentage == 30.0
        
        # 5. 模拟性能严重下降
        deployment.performance_comparator.latest_comparison = {
            'success_rate_diff': 0.20,  # 金丝雀模型更差
            'error_rate_diff': 0.15,
            'overall_performance_score': -0.25,  # 严重性能下降
            'statistical_significance': True
        }
        
        # 6. 验证触发回滚
        assert deployment.should_trigger_rollback() == True
        
        # 7. 执行回滚
        deployment.rollback_deployment("性能严重下降")
        assert deployment.status == DeploymentStatus.ROLLED_BACK
        assert deployment.traffic_percentage == 0.0
    
    def test_ab_test_integration(self, mock_models):
        """测试A/B测试集成"""
        baseline_model, canary_model = mock_models
        
        # 1. 创建A/B测试框架
        ab_test = ABTestFramework(
            model_a=baseline_model,
            model_b=canary_model,
            traffic_split=0.5,
            minimum_sample_size=50,
            confidence_level=0.95,
            test_duration=3600
        )
        
        # 2. 开始测试
        ab_test.start_test()
        
        # 3. 模拟用户流量和结果
        np.random.seed(42)
        
        # 模型A数据（基线性能）
        for i in range(60):
            user_id = f"user_a_{i}"
            model = ab_test.route_traffic(user_id)
            success = 1.0 if np.random.random() > 0.25 else 0.0  # 75%成功率
            ab_test.record_result(user_id, model, 0.75, success)
        
        # 模型B数据（更好性能）
        for i in range(60):
            user_id = f"user_b_{i}"
            model = ab_test.route_traffic(user_id)
            success = 1.0 if np.random.random() > 0.15 else 0.0  # 85%成功率
            ab_test.record_result(user_id, model, 0.85, success)
        
        # 4. 验证样本量充足
        assert ab_test.has_sufficient_sample_size() == True
        
        # 5. 计算统计显著性
        significance_result = ab_test.calculate_statistical_significance()
        assert isinstance(significance_result, dict)
        assert 'p_value' in significance_result
        assert 'is_significant' in significance_result
        
        # 6. 模拟测试完成
        ab_test.start_time = datetime.now() - timedelta(hours=2)
        assert ab_test.is_test_complete() == True
        
        # 7. 确定获胜者
        winner = ab_test.determine_winner()
        assert winner is not None
        assert winner in [baseline_model, canary_model]
    
    def test_model_version_manager_integration(self, serializable_models):
        """测试模型版本管理器集成"""
        baseline_model, canary_model = serializable_models
        
        # 1. 创建版本管理器（使用唯一路径）
        import tempfile
        import shutil
        temp_dir = tempfile.mkdtemp()
        version_manager = ModelVersionManager(storage_path=temp_dir)
        
        # 2. 注册基线模型
        baseline_metadata = ModelMetadata(
            model_id="trading_model",
            version="v1.0.0",
            name="基线交易模型",
            description="稳定的基线模型",
            created_at=datetime.now() - timedelta(days=30),
            created_by="system",
            model_type="rl_agent",
            framework="pytorch",
            metrics={
                "accuracy": 0.85,
                "precision": 0.83,
                "recall": 0.80,
                "f1_score": 0.81
            }
        )
        
        baseline_path = version_manager.register_model(baseline_model, baseline_metadata)
        assert baseline_path is not None
        
        # 3. 注册金丝雀模型
        canary_metadata = ModelMetadata(
            model_id="trading_model",
            version="v1.1.0",
            name="改进的交易模型",
            description="新的改进版本",
            created_at=datetime.now(),
            created_by="developer",
            model_type="rl_agent",
            framework="pytorch",
            metrics={
                "accuracy": 0.90,
                "precision": 0.88,
                "recall": 0.85,
                "f1_score": 0.86
            }
        )
        
        canary_path = version_manager.register_model(canary_model, canary_metadata)
        assert canary_path is not None
        
        # 4. 验证版本列表
        versions = version_manager.list_versions("trading_model")
        assert len(versions) == 2
        assert versions[0].version == "v1.1.0"  # 最新版本在前
        assert versions[1].version == "v1.0.0"
        
        # 5. 比较模型版本
        comparison = version_manager.compare_models(
            "trading_model", "trading_model",
            "v1.0.0", "v1.1.0"
        )
        assert isinstance(comparison, object)
        assert comparison.performance_diff['accuracy'] > 0  # 新版本更好
        
        # 6. 提升新版本为活跃版本
        success = version_manager.promote_model("trading_model", "v1.1.0")
        assert success == True
        
        # 7. 获取活跃模型
        active_metadata = version_manager.get_model_metadata("trading_model")
        assert active_metadata.version == "v1.1.0"
        
        # 清理临时目录
        shutil.rmtree(temp_dir)
    
    def test_end_to_end_deployment_scenario(self, serializable_models, deployment_config):
        """测试端到端部署场景"""
        baseline_model, canary_model = serializable_models
        
        # 1. 版本管理 - 注册模型（使用唯一路径）
        import tempfile
        import shutil
        temp_dir = tempfile.mkdtemp()
        version_manager = ModelVersionManager(storage_path=temp_dir)
        
        baseline_metadata = ModelMetadata(
            model_id="e2e_model_unique",
            version="v1.0.0",
            name="E2E基线模型",
            description="端到端测试基线模型",
            created_at=datetime.now() - timedelta(days=7),
            created_by="system",
            model_type="rl_agent",
            framework="pytorch"
        )
        version_manager.register_model(baseline_model, baseline_metadata)
        
        canary_metadata = ModelMetadata(
            model_id="e2e_model_unique",
            version="v1.1.0",
            name="E2E金丝雀模型",
            description="端到端测试金丝雀模型",
            created_at=datetime.now(),
            created_by="developer",
            model_type="rl_agent",
            framework="pytorch"
        )
        version_manager.register_model(canary_model, canary_metadata)
        
        # 2. 金丝雀部署 - 启动部署
        deployment = CanaryDeployment(
            canary_model=canary_model,
            baseline_model=baseline_model,
            config=deployment_config
        )
        deployment.start_deployment()
        
        # 3. A/B测试 - 并行运行
        ab_test = ABTestFramework(
            model_a=baseline_model,
            model_b=canary_model,
            traffic_split=0.5,
            minimum_sample_size=30,
            confidence_level=0.95,
            test_duration=1800
        )
        ab_test.start_test()
        
        # 4. 模拟运行期间的指标收集
        for i in range(10):
            # 金丝雀部署指标
            canary_metrics = PerformanceMetrics(
                success_rate=0.92 + i * 0.002,
                error_rate=0.05 - i * 0.002,
                avg_response_time=0.11 + i * 0.001,
                throughput=980 + i * 5,
                accuracy=0.91 + i * 0.001,
                precision=0.89 + i * 0.001,
                recall=0.87 + i * 0.001,
                f1_score=0.88 + i * 0.001
            )
            deployment.update_canary_metrics(canary_metrics)
            
            # A/B测试数据 - 确保每个模型都有足够的样本
            for j in range(10):  # 增加样本数量
                user_id = f"user_{i}_{j}"
                model = ab_test.route_traffic(user_id)
                success = 1.0 if np.random.random() > 0.2 else 0.0
                ab_test.record_result(user_id, model, 0.8, success)
        
        # 5. 验证系统状态
        assert deployment.status == DeploymentStatus.ACTIVE
        assert deployment.evaluate_success_criteria() == True
        assert ab_test.has_sufficient_sample_size() == True
        
        # 6. 模拟测试完成和部署推进
        ab_test.start_time = datetime.now() - timedelta(minutes=35)
        if ab_test.is_test_complete():
            winner = ab_test.determine_winner()
            if winner == canary_model:
                # 金丝雀模型获胜，继续部署
                deployment.increase_traffic(40.0)
                deployment.complete_deployment()
                
                # 提升为活跃版本
                version_manager.promote_model("e2e_model_unique", "v1.1.0")
                
                assert deployment.status == DeploymentStatus.COMPLETED
                assert version_manager.get_model_metadata("e2e_model_unique").version == "v1.1.0"
        
        # 7. 验证部署历史
        deployment_status = deployment.get_deployment_status()
        assert deployment_status['deployment_id'] is not None
        assert deployment_status['canary_model_version'] == "v1.1.0"
        assert deployment_status['baseline_model_version'] == "v1.0.0"
        
        # 清理临时目录
        shutil.rmtree(temp_dir)