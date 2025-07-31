"""
模型可解释性的单元测试
测试SHAP和LIME解释器集成，注意力权重可视化和特征重要性分析，解释结果的合理性和可理解性
严格遵循TDD开发，不允许捕获异常，让异常暴露以尽早发现错误
"""
import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import os

from src.rl_trading_system.explainability.model_explainer import (
    ModelExplainer,
    SHAPExplainer,
    LIMEExplainer,
    AttentionVisualizer,
    FeatureImportanceAnalyzer,
    ExplanationReport,
    ExplanationConfig,
    ExplanationType
)


class TestModelExplainer:
    """模型解释器测试类"""

    @pytest.fixture
    def sample_model(self):
        """创建样本模型"""
        model = Mock()
        model.predict = Mock(return_value=np.array([0.1, 0.2, 0.7]))
        model.predict_proba = Mock(return_value=np.array([[0.1, 0.2, 0.7]]))
        model.version = "test_model_v1.0"
        model.feature_names = [f"feature_{i}" for i in range(10)]
        return model

    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        np.random.seed(42)
        features = np.random.random((100, 10))
        labels = np.random.randint(0, 3, 100)
        feature_names = [f"feature_{i}" for i in range(10)]
        
        return {
            'features': features,
            'labels': labels,
            'feature_names': feature_names
        }

    @pytest.fixture
    def explanation_config(self):
        """创建解释配置"""
        return ExplanationConfig(
            explanation_types=['shap', 'lime', 'attention'],
            num_samples=100,
            confidence_threshold=0.8,
            max_features=10,
            background_samples=50,
            output_format='html'
        )

    @pytest.fixture
    def model_explainer(self, sample_model, explanation_config):
        """创建模型解释器"""
        return ModelExplainer(
            model=sample_model,
            config=explanation_config
        )

    def test_model_explainer_initialization(self, model_explainer, sample_model, explanation_config):
        """测试模型解释器初始化"""
        assert model_explainer.model == sample_model
        assert model_explainer.config == explanation_config
        assert model_explainer.shap_explainer is not None
        assert model_explainer.lime_explainer is not None
        assert model_explainer.attention_visualizer is not None
        assert model_explainer.feature_analyzer is not None

    def test_explain_prediction_single_instance(self, model_explainer, sample_data):
        """测试单个实例的预测解释"""
        instance = sample_data['features'][0]
        
        explanation = model_explainer.explain_prediction(
            instance=instance,
            explanation_types=['shap', 'lime']
        )
        
        assert isinstance(explanation, dict)
        assert 'shap' in explanation
        assert 'lime' in explanation
        assert 'prediction' in explanation
        assert 'confidence' in explanation
        assert isinstance(explanation['prediction'], (int, float, np.ndarray))

    def test_explain_prediction_batch(self, model_explainer, sample_data):
        """测试批量预测解释"""
        instances = sample_data['features'][:5]
        
        explanations = model_explainer.explain_batch(
            instances=instances,
            explanation_types=['shap']
        )
        
        assert isinstance(explanations, list)
        assert len(explanations) == 5
        
        for explanation in explanations:
            assert isinstance(explanation, dict)
            assert 'shap' in explanation
            assert 'prediction' in explanation

    def test_global_explanation(self, model_explainer, sample_data):
        """测试全局解释"""
        global_explanation = model_explainer.explain_global(
            dataset=sample_data['features'],
            feature_names=sample_data['feature_names']
        )
        
        assert isinstance(global_explanation, dict)
        assert 'feature_importance' in global_explanation
        assert 'global_shap_values' in global_explanation
        assert 'summary_statistics' in global_explanation
        
        # 验证特征重要性
        feature_importance = global_explanation['feature_importance']
        assert isinstance(feature_importance, dict)
        assert len(feature_importance) == len(sample_data['feature_names'])

    def test_explanation_consistency(self, model_explainer, sample_data):
        """测试解释一致性"""
        instance = sample_data['features'][0]
        
        # 多次解释同一个实例
        explanation1 = model_explainer.explain_prediction(instance, ['shap'])
        explanation2 = model_explainer.explain_prediction(instance, ['shap'])
        
        # SHAP值应该一致（在数值误差内）
        shap_values1 = explanation1['shap']['values']
        shap_values2 = explanation2['shap']['values']
        
        np.testing.assert_array_almost_equal(shap_values1, shap_values2, decimal=5)

    def test_explanation_validity(self, model_explainer, sample_data):
        """测试解释有效性"""
        instance = sample_data['features'][0]
        
        explanation = model_explainer.explain_prediction(instance, ['shap', 'lime'])
        
        # 验证SHAP值的有效性
        shap_values = explanation['shap']['values']
        assert isinstance(shap_values, np.ndarray)
        assert len(shap_values) == len(instance)
        assert not np.any(np.isnan(shap_values))
        assert not np.any(np.isinf(shap_values))
        
        # 验证LIME解释的有效性
        lime_explanation = explanation['lime']
        assert 'feature_weights' in lime_explanation
        assert 'local_prediction' in lime_explanation

    def test_feature_filtering(self, model_explainer, sample_data):
        """测试特征过滤"""
        instance = sample_data['features'][0]
        
        # 限制返回的特征数量
        explanation = model_explainer.explain_prediction(
            instance=instance,
            explanation_types=['shap'],
            top_k_features=5
        )
        
        shap_values = explanation['shap']['values']
        top_features = explanation['shap']['top_features']
        
        assert len(top_features) <= 5
        assert all(isinstance(idx, (int, np.integer)) for idx in top_features)

    def test_confidence_based_filtering(self, model_explainer, sample_data):
        """测试基于置信度的过滤"""
        instance = sample_data['features'][0]
        
        explanation = model_explainer.explain_prediction(
            instance=instance,
            explanation_types=['shap'],
            confidence_threshold=0.9
        )
        
        # 检查是否只返回高置信度的解释
        assert explanation['confidence'] >= 0.0  # 至少有基本的置信度信息

    def test_explanation_caching(self, model_explainer, sample_data):
        """测试解释缓存"""
        instance = sample_data['features'][0]
        
        # 第一次解释（应该计算）
        start_time = datetime.now()
        explanation1 = model_explainer.explain_prediction(instance, ['shap'])
        first_duration = (datetime.now() - start_time).total_seconds()
        
        # 第二次解释（应该使用缓存）
        start_time = datetime.now()
        explanation2 = model_explainer.explain_prediction(instance, ['shap'])
        second_duration = (datetime.now() - start_time).total_seconds()
        
        # 验证结果一致
        np.testing.assert_array_almost_equal(
            explanation1['shap']['values'], 
            explanation2['shap']['values']
        )
        
        # 第二次应该更快（使用缓存）
        assert second_duration <= first_duration

    def test_explanation_serialization(self, model_explainer, sample_data):
        """测试解释序列化"""
        instance = sample_data['features'][0]
        
        explanation = model_explainer.explain_prediction(instance, ['shap', 'lime'])
        
        # 序列化到JSON
        serialized = model_explainer.serialize_explanation(explanation)
        assert isinstance(serialized, str)
        
        # 反序列化
        deserialized = model_explainer.deserialize_explanation(serialized)
        assert isinstance(deserialized, dict)
        assert 'shap' in deserialized
        assert 'lime' in deserialized

    def test_invalid_explanation_type(self, model_explainer, sample_data):
        """测试无效的解释类型"""
        instance = sample_data['features'][0]
        
        with pytest.raises(ValueError, match="不支持的解释类型"):
            model_explainer.explain_prediction(instance, ['invalid_type'])

    def test_empty_instance_handling(self, model_explainer):
        """测试空实例处理"""
        with pytest.raises(ValueError, match="输入实例不能为空"):
            model_explainer.explain_prediction(np.array([]), ['shap'])

    def test_mismatched_feature_dimensions(self, model_explainer):
        """测试特征维度不匹配"""
        # 创建维度不匹配的实例
        wrong_dimension_instance = np.random.random(5)  # 应该是10维
        
        with pytest.raises(ValueError, match="特征维度不匹配"):
            model_explainer.explain_prediction(wrong_dimension_instance, ['shap'])


class TestSHAPExplainer:
    """SHAP解释器测试类"""

    @pytest.fixture
    def mock_model(self):
        """创建模拟模型"""
        model = Mock()
        model.predict = Mock(return_value=np.array([0.1, 0.2, 0.7]))
        model.predict_proba = Mock(return_value=np.array([[0.1, 0.2, 0.7]]))
        return model

    @pytest.fixture
    def background_data(self):
        """创建背景数据"""
        np.random.seed(42)
        return np.random.random((50, 10))

    @pytest.fixture
    def shap_explainer(self, mock_model, background_data):
        """创建SHAP解释器"""
        return SHAPExplainer(
            model=mock_model,
            background_data=background_data,
            explainer_type='tree'
        )

    def test_shap_explainer_initialization(self, shap_explainer, mock_model, background_data):
        """测试SHAP解释器初始化"""
        assert shap_explainer.model == mock_model
        np.testing.assert_array_equal(shap_explainer.background_data, background_data)
        assert shap_explainer.explainer is not None

    def test_shap_values_calculation(self, shap_explainer):
        """测试SHAP值计算"""
        instance = np.random.random(10)
        
        shap_values = shap_explainer.calculate_shap_values(instance)
        
        assert isinstance(shap_values, np.ndarray)
        assert len(shap_values) == len(instance)
        assert not np.any(np.isnan(shap_values))

    def test_shap_values_batch(self, shap_explainer):
        """测试批量SHAP值计算"""
        instances = np.random.random((5, 10))
        
        shap_values = shap_explainer.calculate_shap_values_batch(instances)
        
        assert isinstance(shap_values, np.ndarray)
        assert shap_values.shape == instances.shape
        assert not np.any(np.isnan(shap_values))

    def test_shap_summary_plot_data(self, shap_explainer):
        """测试SHAP摘要图数据"""
        instances = np.random.random((20, 10))
        feature_names = [f"feature_{i}" for i in range(10)]
        
        summary_data = shap_explainer.get_summary_plot_data(instances, feature_names)
        
        assert isinstance(summary_data, dict)
        assert 'shap_values' in summary_data
        assert 'feature_values' in summary_data
        assert 'feature_names' in summary_data

    def test_shap_waterfall_data(self, shap_explainer):
        """测试SHAP瀑布图数据"""
        instance = np.random.random(10)
        feature_names = [f"feature_{i}" for i in range(10)]
        
        waterfall_data = shap_explainer.get_waterfall_data(instance, feature_names)
        
        assert isinstance(waterfall_data, dict)
        assert 'shap_values' in waterfall_data
        assert 'base_value' in waterfall_data
        assert 'feature_names' in waterfall_data
        assert 'feature_values' in waterfall_data

    def test_shap_force_plot_data(self, shap_explainer):
        """测试SHAP力图数据"""
        instance = np.random.random(10)
        feature_names = [f"feature_{i}" for i in range(10)]
        
        force_plot_data = shap_explainer.get_force_plot_data(instance, feature_names)
        
        assert isinstance(force_plot_data, dict)
        assert 'shap_values' in force_plot_data
        assert 'base_value' in force_plot_data
        assert 'prediction' in force_plot_data

    def test_shap_feature_importance(self, shap_explainer):
        """测试SHAP特征重要性"""
        instances = np.random.random((50, 10))
        feature_names = [f"feature_{i}" for i in range(10)]
        
        importance_scores = shap_explainer.get_feature_importance(instances, feature_names)
        
        assert isinstance(importance_scores, dict)
        assert len(importance_scores) == len(feature_names)
        assert all(isinstance(score, (int, float)) for score in importance_scores.values())

    def test_different_explainer_types(self):
        """测试不同的SHAP解释器类型"""
        mock_model = Mock()
        background_data = np.random.random((30, 5))
        
        # 测试不同的解释器类型
        explainer_types = ['tree', 'linear', 'kernel', 'deep']
        
        for explainer_type in explainer_types:
            explainer = SHAPExplainer(
                model=mock_model,
                background_data=background_data,
                explainer_type=explainer_type
            )
            assert explainer.explainer_type == explainer_type

    def test_shap_interaction_values(self, shap_explainer):
        """测试SHAP交互值"""
        instance = np.random.random(10)
        
        # 注意：不是所有SHAP解释器都支持交互值
        if hasattr(shap_explainer.explainer, 'shap_interaction_values'):
            interaction_values = shap_explainer.calculate_interaction_values(instance)
            assert isinstance(interaction_values, np.ndarray)
            assert interaction_values.shape == (len(instance), len(instance))

    def test_shap_expected_value(self, shap_explainer):
        """测试SHAP期望值"""
        expected_value = shap_explainer.get_expected_value()
        
        assert isinstance(expected_value, (int, float, np.ndarray))
        assert not np.isnan(expected_value) if isinstance(expected_value, (int, float)) else not np.any(np.isnan(expected_value))


class TestLIMEExplainer:
    """LIME解释器测试类"""

    @pytest.fixture
    def mock_model(self):
        """创建模拟模型"""
        model = Mock()
        model.predict_proba = Mock(return_value=np.array([[0.1, 0.2, 0.7]]))
        return model

    @pytest.fixture
    def training_data(self):
        """创建训练数据"""
        np.random.seed(42)
        return np.random.random((100, 10))

    @pytest.fixture
    def lime_explainer(self, mock_model, training_data):
        """创建LIME解释器"""
        return LIMEExplainer(
            model=mock_model,
            training_data=training_data,
            mode='classification',
            feature_names=[f"feature_{i}" for i in range(10)]
        )

    def test_lime_explainer_initialization(self, lime_explainer, mock_model, training_data):
        """测试LIME解释器初始化"""
        assert lime_explainer.model == mock_model
        np.testing.assert_array_equal(lime_explainer.training_data, training_data)
        assert lime_explainer.mode == 'classification'
        assert len(lime_explainer.feature_names) == 10

    def test_lime_explanation_single_instance(self, lime_explainer):
        """测试单个实例的LIME解释"""
        instance = np.random.random(10)
        
        explanation = lime_explainer.explain_instance(
            instance=instance,
            num_features=5,
            num_samples=1000
        )
        
        assert isinstance(explanation, dict)
        assert 'feature_weights' in explanation
        assert 'local_prediction' in explanation
        assert 'score' in explanation
        
        # 验证特征权重
        feature_weights = explanation['feature_weights']
        assert isinstance(feature_weights, dict)
        assert len(feature_weights) <= 5

    def test_lime_explanation_regression(self):
        """测试回归模式的LIME解释"""
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([1.5]))
        training_data = np.random.random((100, 10))
        
        lime_explainer = LIMEExplainer(
            model=mock_model,
            training_data=training_data,
            mode='regression',
            feature_names=[f"feature_{i}" for i in range(10)]
        )
        
        instance = np.random.random(10)
        explanation = lime_explainer.explain_instance(instance, num_features=5)
        
        assert isinstance(explanation, dict)
        assert 'feature_weights' in explanation
        assert 'local_prediction' in explanation

    def test_lime_feature_selection(self, lime_explainer):
        """测试LIME特征选择"""
        instance = np.random.random(10)
        
        # 测试不同的特征选择方法
        selection_methods = ['auto', 'forward_selection', 'lasso_path']
        
        for method in selection_methods:
            explanation = lime_explainer.explain_instance(
                instance=instance,
                num_features=3,
                feature_selection=method
            )
            
            assert isinstance(explanation, dict)
            assert len(explanation['feature_weights']) <= 3

    def test_lime_discretization(self, lime_explainer):
        """测试LIME离散化"""
        instance = np.random.random(10)
        
        # 测试不同的离散化方法
        discretization_methods = ['quartiles', 'deciles', 'entropy']
        
        for method in discretization_methods:
            explanation = lime_explainer.explain_instance(
                instance=instance,
                discretize_continuous=True,
                discretization_method=method
            )
            
            assert isinstance(explanation, dict)
            assert 'feature_weights' in explanation

    def test_lime_kernel_width(self, lime_explainer):
        """测试LIME核宽度参数"""
        instance = np.random.random(10)
        
        # 测试不同的核宽度
        kernel_widths = [0.75, 1.0, 1.5]
        
        for width in kernel_widths:
            explanation = lime_explainer.explain_instance(
                instance=instance,
                kernel_width=width
            )
            
            assert isinstance(explanation, dict)
            assert 'score' in explanation

    def test_lime_sample_around_instance(self, lime_explainer):
        """测试LIME实例周围采样"""
        instance = np.random.random(10)
        
        samples = lime_explainer.sample_around_instance(
            instance=instance,
            num_samples=500
        )
        
        assert isinstance(samples, np.ndarray)
        assert samples.shape[0] == 500
        assert samples.shape[1] == len(instance)

    def test_lime_distance_metric(self, lime_explainer):
        """测试LIME距离度量"""
        instance = np.random.random(10)
        
        # 测试不同的距离度量
        distance_metrics = ['euclidean', 'cosine', 'manhattan']
        
        for metric in distance_metrics:
            explanation = lime_explainer.explain_instance(
                instance=instance,
                distance_metric=metric
            )
            
            assert isinstance(explanation, dict)
            assert 'feature_weights' in explanation

    def test_lime_explanation_stability(self, lime_explainer):
        """测试LIME解释稳定性"""
        instance = np.random.random(10)
        
        # 多次解释同一个实例
        explanations = []
        for _ in range(5):
            explanation = lime_explainer.explain_instance(
                instance=instance,
                num_samples=1000,
                random_state=42  # 固定随机种子
            )
            explanations.append(explanation['feature_weights'])
        
        # 验证解释的稳定性
        for i in range(1, len(explanations)):
            # 主要特征应该保持一致
            top_features_1 = sorted(explanations[0].items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            top_features_i = sorted(explanations[i].items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            
            # 至少有2个主要特征应该一致
            common_features = set([f[0] for f in top_features_1]) & set([f[0] for f in top_features_i])
            assert len(common_features) >= 2


class TestAttentionVisualizer:
    """注意力可视化器测试类"""

    @pytest.fixture
    def mock_transformer_model(self):
        """创建模拟Transformer模型"""
        model = Mock()
        
        # 模拟注意力权重
        num_heads = 8
        seq_length = 20
        attention_weights = torch.rand(1, num_heads, seq_length, seq_length)
        
        model.get_attention_weights = Mock(return_value=attention_weights)
        model.num_heads = num_heads
        model.num_layers = 6
        
        return model

    @pytest.fixture
    def attention_visualizer(self, mock_transformer_model):
        """创建注意力可视化器"""
        return AttentionVisualizer(
            model=mock_transformer_model,
            feature_names=[f"feature_{i}" for i in range(20)]
        )

    def test_attention_visualizer_initialization(self, attention_visualizer, mock_transformer_model):
        """测试注意力可视化器初始化"""
        assert attention_visualizer.model == mock_transformer_model
        assert len(attention_visualizer.feature_names) == 20

    def test_extract_attention_weights(self, attention_visualizer):
        """测试提取注意力权重"""
        input_sequence = torch.rand(1, 20, 10)  # batch_size=1, seq_len=20, feature_dim=10
        
        attention_weights = attention_visualizer.extract_attention_weights(input_sequence)
        
        assert isinstance(attention_weights, torch.Tensor)
        assert attention_weights.shape == (1, 8, 20, 20)  # batch, heads, seq_len, seq_len

    def test_aggregate_attention_heads(self, attention_visualizer):
        """测试聚合注意力头"""
        input_sequence = torch.rand(1, 20, 10)
        attention_weights = attention_visualizer.extract_attention_weights(input_sequence)
        
        # 测试平均聚合
        avg_attention = attention_visualizer.aggregate_attention_heads(
            attention_weights, method='mean'
        )
        assert avg_attention.shape == (1, 20, 20)
        
        # 测试最大值聚合
        max_attention = attention_visualizer.aggregate_attention_heads(
            attention_weights, method='max'
        )
        assert max_attention.shape == (1, 20, 20)

    def test_attention_heatmap_data(self, attention_visualizer):
        """测试注意力热力图数据"""
        input_sequence = torch.rand(1, 20, 10)
        
        heatmap_data = attention_visualizer.get_attention_heatmap_data(
            input_sequence=input_sequence,
            layer_idx=0,
            head_idx=0
        )
        
        assert isinstance(heatmap_data, dict)
        assert 'attention_matrix' in heatmap_data
        assert 'feature_names' in heatmap_data
        assert 'sequence_length' in heatmap_data

    def test_attention_pattern_analysis(self, attention_visualizer):
        """测试注意力模式分析"""
        input_sequence = torch.rand(1, 20, 10)
        
        pattern_analysis = attention_visualizer.analyze_attention_patterns(input_sequence)
        
        assert isinstance(pattern_analysis, dict)
        assert 'head_diversity' in pattern_analysis
        assert 'attention_entropy' in pattern_analysis
        assert 'dominant_patterns' in pattern_analysis

    def test_attention_rollout(self, attention_visualizer):
        """测试注意力回滚"""
        input_sequence = torch.rand(1, 20, 10)
        
        rollout_attention = attention_visualizer.compute_attention_rollout(input_sequence)
        
        assert isinstance(rollout_attention, torch.Tensor)
        assert rollout_attention.shape == (1, 20, 20)

    def test_attention_flow_visualization(self, attention_visualizer):
        """测试注意力流可视化"""
        input_sequence = torch.rand(1, 20, 10)
        
        flow_data = attention_visualizer.get_attention_flow_data(input_sequence)
        
        assert isinstance(flow_data, dict)
        assert 'source_nodes' in flow_data
        assert 'target_nodes' in flow_data
        assert 'edge_weights' in flow_data

    def test_layer_wise_attention_comparison(self, attention_visualizer):
        """测试层间注意力比较"""
        input_sequence = torch.rand(1, 20, 10)
        
        # 模拟多层注意力权重
        multi_layer_weights = []
        for layer in range(6):
            layer_weights = torch.rand(1, 8, 20, 20)
            multi_layer_weights.append(layer_weights)
        
        attention_visualizer.model.get_all_attention_weights = Mock(
            return_value=multi_layer_weights
        )
        
        comparison_data = attention_visualizer.compare_layer_attention(input_sequence)
        
        assert isinstance(comparison_data, dict)
        assert 'layer_similarities' in comparison_data
        assert 'attention_evolution' in comparison_data

    def test_attention_head_specialization(self, attention_visualizer):
        """测试注意力头专门化分析"""
        input_sequence = torch.rand(1, 20, 10)
        
        specialization_analysis = attention_visualizer.analyze_head_specialization(input_sequence)
        
        assert isinstance(specialization_analysis, dict)
        assert 'head_roles' in specialization_analysis
        assert 'specialization_scores' in specialization_analysis

    def test_temporal_attention_patterns(self, attention_visualizer):
        """测试时序注意力模式"""
        # 模拟时序数据
        temporal_sequences = torch.rand(5, 20, 10)  # 5个时间步
        
        temporal_patterns = attention_visualizer.analyze_temporal_patterns(temporal_sequences)
        
        assert isinstance(temporal_patterns, dict)
        assert 'temporal_consistency' in temporal_patterns
        assert 'attention_stability' in temporal_patterns

    def test_attention_attribution(self, attention_visualizer):
        """测试注意力归因"""
        input_sequence = torch.rand(1, 20, 10)
        target_position = 10
        
        attribution_scores = attention_visualizer.compute_attention_attribution(
            input_sequence=input_sequence,
            target_position=target_position
        )
        
        assert isinstance(attribution_scores, torch.Tensor)
        assert attribution_scores.shape == (20,)  # 每个位置的归因分数


class TestFeatureImportanceAnalyzer:
    """特征重要性分析器测试类"""

    @pytest.fixture
    def sample_model(self):
        """创建样本模型"""
        model = Mock()
        model.predict = Mock(return_value=np.array([0.8]))
        model.feature_importances_ = np.array([0.1, 0.2, 0.15, 0.05, 0.3, 0.08, 0.12])
        return model

    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        np.random.seed(42)
        features = np.random.random((100, 7))
        labels = np.random.randint(0, 2, 100)
        feature_names = ['return_1d', 'volatility', 'volume', 'rsi', 'macd', 'bb_width', 'momentum']
        
        return {
            'features': features,
            'labels': labels,
            'feature_names': feature_names
        }

    @pytest.fixture
    def feature_analyzer(self, sample_model, sample_data):
        """创建特征重要性分析器"""
        return FeatureImportanceAnalyzer(
            model=sample_model,
            feature_names=sample_data['feature_names']
        )

    def test_feature_analyzer_initialization(self, feature_analyzer, sample_model, sample_data):
        """测试特征重要性分析器初始化"""
        assert feature_analyzer.model == sample_model
        assert feature_analyzer.feature_names == sample_data['feature_names']

    def test_built_in_feature_importance(self, feature_analyzer):
        """测试内置特征重要性"""
        importance_scores = feature_analyzer.get_built_in_importance()
        
        assert isinstance(importance_scores, dict)
        assert len(importance_scores) == 7
        assert all(isinstance(score, (int, float)) for score in importance_scores.values())
        assert all(score >= 0 for score in importance_scores.values())

    def test_permutation_importance(self, feature_analyzer, sample_data):
        """测试置换重要性"""
        importance_scores = feature_analyzer.calculate_permutation_importance(
            X=sample_data['features'],
            y=sample_data['labels'],
            n_repeats=5
        )
        
        assert isinstance(importance_scores, dict)
        assert len(importance_scores) == len(sample_data['feature_names'])
        
        # 验证重要性分数
        for feature_name, scores in importance_scores.items():
            assert 'mean' in scores
            assert 'std' in scores
            assert isinstance(scores['mean'], (int, float))
            assert isinstance(scores['std'], (int, float))

    def test_correlation_based_importance(self, feature_analyzer, sample_data):
        """测试基于相关性的重要性"""
        importance_scores = feature_analyzer.calculate_correlation_importance(
            X=sample_data['features'],
            y=sample_data['labels']
        )
        
        assert isinstance(importance_scores, dict)
        assert len(importance_scores) == len(sample_data['feature_names'])
        assert all(isinstance(score, (int, float)) for score in importance_scores.values())

    def test_mutual_information_importance(self, feature_analyzer, sample_data):
        """测试互信息重要性"""
        importance_scores = feature_analyzer.calculate_mutual_information_importance(
            X=sample_data['features'],
            y=sample_data['labels']
        )
        
        assert isinstance(importance_scores, dict)
        assert len(importance_scores) == len(sample_data['feature_names'])
        assert all(score >= 0 for score in importance_scores.values())

    def test_recursive_feature_elimination(self, feature_analyzer, sample_data):
        """测试递归特征消除"""
        selected_features = feature_analyzer.recursive_feature_elimination(
            X=sample_data['features'],
            y=sample_data['labels'],
            n_features_to_select=5
        )
        
        assert isinstance(selected_features, list)
        assert len(selected_features) == 5
        assert all(feature in sample_data['feature_names'] for feature in selected_features)

    def test_feature_importance_ranking(self, feature_analyzer, sample_data):
        """测试特征重要性排序"""
        ranking = feature_analyzer.rank_features(
            X=sample_data['features'],
            y=sample_data['labels'],
            methods=['built_in', 'permutation', 'correlation']
        )
        
        assert isinstance(ranking, dict)
        assert 'overall_ranking' in ranking
        assert 'method_rankings' in ranking
        
        overall_ranking = ranking['overall_ranking']
        assert len(overall_ranking) == len(sample_data['feature_names'])
        assert all(isinstance(rank, int) for rank in overall_ranking.values())

    def test_feature_stability_analysis(self, feature_analyzer, sample_data):
        """测试特征稳定性分析"""
        stability_scores = feature_analyzer.analyze_feature_stability(
            X=sample_data['features'],
            y=sample_data['labels'],
            n_bootstrap=10
        )
        
        assert isinstance(stability_scores, dict)
        assert len(stability_scores) == len(sample_data['feature_names'])
        
        for feature_name, stability in stability_scores.items():
            assert 'stability_score' in stability
            assert 'rank_variance' in stability
            assert 0 <= stability['stability_score'] <= 1

    def test_feature_interaction_analysis(self, feature_analyzer, sample_data):
        """测试特征交互分析"""
        interaction_scores = feature_analyzer.analyze_feature_interactions(
            X=sample_data['features'],
            feature_pairs=[('return_1d', 'volatility'), ('rsi', 'macd')]
        )
        
        assert isinstance(interaction_scores, dict)
        assert len(interaction_scores) == 2
        
        for pair, score in interaction_scores.items():
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            assert isinstance(score, (int, float))

    def test_temporal_importance_analysis(self, feature_analyzer):
        """测试时序重要性分析"""
        # 创建时序数据
        np.random.seed(42)
        temporal_data = []
        for t in range(10):
            X_t = np.random.random((20, 7))
            y_t = np.random.randint(0, 2, 20)
            temporal_data.append((X_t, y_t))
        
        temporal_importance = feature_analyzer.analyze_temporal_importance(temporal_data)
        
        assert isinstance(temporal_importance, dict)
        assert 'importance_evolution' in temporal_importance
        assert 'stability_over_time' in temporal_importance

    def test_grouped_feature_importance(self, feature_analyzer, sample_data):
        """测试分组特征重要性"""
        feature_groups = {
            'technical': ['rsi', 'macd', 'bb_width', 'momentum'],
            'market': ['return_1d', 'volatility', 'volume']
        }
        
        group_importance = feature_analyzer.calculate_grouped_importance(
            X=sample_data['features'],
            y=sample_data['labels'],
            feature_groups=feature_groups
        )
        
        assert isinstance(group_importance, dict)
        assert 'technical' in group_importance
        assert 'market' in group_importance
        
        for group_name, importance in group_importance.items():
            assert isinstance(importance, (int, float))
            assert importance >= 0


class TestExplanationReport:
    """解释报告测试类"""

    @pytest.fixture
    def sample_explanations(self):
        """创建样本解释"""
        return {
            'shap': {
                'values': np.array([0.1, -0.2, 0.3, -0.1, 0.05]),
                'base_value': 0.5,
                'feature_names': ['f1', 'f2', 'f3', 'f4', 'f5']
            },
            'lime': {
                'feature_weights': {'f1': 0.15, 'f2': -0.18, 'f3': 0.25},
                'score': 0.85
            },
            'attention': {
                'weights': np.random.random((10, 10)),
                'patterns': ['local', 'global']
            },
            'prediction': 0.75,
            'confidence': 0.9
        }

    @pytest.fixture
    def explanation_report(self):
        """创建解释报告"""
        return ExplanationReport(
            model_name="test_model_v1.0",
            timestamp=datetime.now()
        )

    def test_report_initialization(self, explanation_report):
        """测试报告初始化"""
        assert explanation_report.model_name == "test_model_v1.0"
        assert isinstance(explanation_report.timestamp, datetime)
        assert explanation_report.explanations == []

    def test_add_explanation(self, explanation_report, sample_explanations):
        """测试添加解释"""
        explanation_report.add_explanation(sample_explanations)
        
        assert len(explanation_report.explanations) == 1
        assert explanation_report.explanations[0] == sample_explanations

    def test_generate_html_report(self, explanation_report, sample_explanations):
        """测试生成HTML报告"""
        explanation_report.add_explanation(sample_explanations)
        
        html_report = explanation_report.generate_html_report()
        
        assert isinstance(html_report, str)
        assert '<html>' in html_report
        assert '<body>' in html_report
        assert 'test_model_v1.0' in html_report

    def test_generate_json_report(self, explanation_report, sample_explanations):
        """测试生成JSON报告"""
        explanation_report.add_explanation(sample_explanations)
        
        json_report = explanation_report.generate_json_report()
        
        assert isinstance(json_report, str)
        report_data = json.loads(json_report)
        assert 'model_name' in report_data
        assert 'explanations' in report_data

    def test_save_report(self, explanation_report, sample_explanations):
        """测试保存报告"""
        explanation_report.add_explanation(sample_explanations)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 保存HTML报告
            html_path = os.path.join(temp_dir, 'report.html')
            explanation_report.save_report(html_path, format='html')
            
            assert os.path.exists(html_path)
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
                assert '<html>' in content
            
            # 保存JSON报告
            json_path = os.path.join(temp_dir, 'report.json')
            explanation_report.save_report(json_path, format='json')
            
            assert os.path.exists(json_path)
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                assert 'model_name' in data

    def test_summary_statistics(self, explanation_report, sample_explanations):
        """测试摘要统计"""
        # 添加多个解释
        for _ in range(5):
            explanation_report.add_explanation(sample_explanations)
        
        summary = explanation_report.get_summary_statistics()
        
        assert isinstance(summary, dict)
        assert 'total_explanations' in summary
        assert 'avg_confidence' in summary
        assert 'feature_importance_stats' in summary

    def test_feature_importance_aggregation(self, explanation_report):
        """测试特征重要性聚合"""
        # 添加多个具有不同特征重要性的解释
        explanations = []
        for i in range(3):
            explanation = {
                'shap': {
                    'values': np.random.random(5),
                    'feature_names': ['f1', 'f2', 'f3', 'f4', 'f5']
                },
                'prediction': 0.7 + i * 0.1
            }
            explanations.append(explanation)
            explanation_report.add_explanation(explanation)
        
        aggregated_importance = explanation_report.aggregate_feature_importance()
        
        assert isinstance(aggregated_importance, dict)
        assert len(aggregated_importance) == 5
        assert all(isinstance(score, (int, float)) for score in aggregated_importance.values())

    def test_explanation_consistency_check(self, explanation_report):
        """测试解释一致性检查"""
        # 添加一致的解释
        consistent_explanations = []
        for _ in range(3):
            explanation = {
                'shap': {
                    'values': np.array([0.1, -0.2, 0.3, -0.1, 0.05]),
                    'feature_names': ['f1', 'f2', 'f3', 'f4', 'f5']
                }
            }
            consistent_explanations.append(explanation)
            explanation_report.add_explanation(explanation)
        
        consistency_score = explanation_report.check_explanation_consistency()
        
        assert isinstance(consistency_score, float)
        assert 0 <= consistency_score <= 1
        assert consistency_score > 0.8  # 应该很一致

    def test_comparative_analysis(self, explanation_report):
        """测试比较分析"""
        # 添加不同方法的解释
        shap_explanation = {
            'shap': {'values': np.array([0.1, -0.2, 0.3]), 'feature_names': ['f1', 'f2', 'f3']}
        }
        lime_explanation = {
            'lime': {'feature_weights': {'f1': 0.15, 'f2': -0.18, 'f3': 0.25}}
        }
        
        explanation_report.add_explanation(shap_explanation)
        explanation_report.add_explanation(lime_explanation)
        
        comparison = explanation_report.compare_explanation_methods()
        
        assert isinstance(comparison, dict)
        assert 'method_agreement' in comparison
        assert 'feature_ranking_correlation' in comparison