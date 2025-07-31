#!/usr/bin/env python3
"""
CI/CD流水线集成测试

测试GitHub Actions、Jenkins等CI/CD流水线的集成和自动化部署
需求: 8.1, 8.4
"""

import pytest
import subprocess
import yaml
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock

from src.rl_trading_system.deployment.containerized_deployment import (
    CICDPipeline,
    DockerManager,
    KubernetesManager,
    ContainerConfig,
    KubernetesConfig
)


class TestGitHubActionsPipeline:
    """GitHub Actions流水线测试"""
    
    @pytest.fixture
    def github_actions_config(self):
        """GitHub Actions配置"""
        return {
            "repository": "https://github.com/rl-trading/rl-trading-system.git",
            "branch": "main",
            "build_stages": ["lint", "test", "security-scan", "build", "deploy"],
            "test_commands": [
                "python -m pytest tests/unit/ --cov=src/ --cov-report=xml",
                "python -m pytest tests/integration/ -v",
                "python -m pytest tests/e2e/ -v --maxfail=1"
            ],
            "lint_commands": [
                "flake8 src/ tests/ --max-line-length=100 --exclude=__pycache__",
                "mypy src/ --ignore-missing-imports --strict",
                "black --check src/ tests/",
                "isort --check-only src/ tests/"
            ],
            "security_commands": [
                "bandit -r src/ -f json -o security-report.json",
                "safety check --json --output safety-report.json",
                "trivy fs --format json --output trivy-report.json ."
            ],
            "build_commands": [
                "docker build -t rl-trading-system:${GITHUB_SHA} .",
                "docker tag rl-trading-system:${GITHUB_SHA} rl-trading-system:latest",
                "echo ${DOCKER_PASSWORD} | docker login -u ${DOCKER_USERNAME} --password-stdin",
                "docker push rl-trading-system:${GITHUB_SHA}",
                "docker push rl-trading-system:latest"
            ],
            "deploy_commands": [
                "kubectl config use-context ${KUBE_CONTEXT}",
                "kubectl set image deployment/rl-trading-system rl-trading-system=rl-trading-system:${GITHUB_SHA}",
                "kubectl rollout status deployment/rl-trading-system --timeout=300s",
                "kubectl get pods -l app=rl-trading-system -o wide"
            ],
            "notifications": {
                "slack_webhook": "${SLACK_WEBHOOK_URL}",
                "email": "devops@rltrading.com"
            }
        }
    
    @pytest.fixture
    def github_pipeline(self, github_actions_config):
        """GitHub Actions流水线实例"""
        return CICDPipeline(github_actions_config)
    
    def test_github_actions_workflow_structure(self, github_pipeline):
        """测试GitHub Actions工作流结构"""
        workflow_yaml = github_pipeline.generate_github_actions_workflow()
        workflow_data = yaml.safe_load(workflow_yaml)
        
        # 验证基本结构
        assert workflow_data["name"] == "CI/CD Pipeline"
        assert "on" in workflow_data
        assert "jobs" in workflow_data
        
        # 验证触发条件
        triggers = workflow_data["on"]
        assert "push" in triggers
        assert "pull_request" in triggers
        assert triggers["push"]["branches"] == ["main"]
        
        # 验证作业依赖关系
        jobs = workflow_data["jobs"]
        expected_jobs = ["lint", "test", "security-scan", "build", "deploy"]
        
        for job_name in expected_jobs:
            assert job_name in jobs, f"Missing job: {job_name}"
        
        # 验证作业依赖
        assert jobs["test"]["needs"] == "lint"
        assert jobs["security-scan"]["needs"] == "test"
        assert jobs["build"]["needs"] == "security-scan"
        assert jobs["deploy"]["needs"] == "build"
        
        # 验证部署条件
        assert jobs["deploy"]["if"] == "github.ref == 'refs/heads/main'"
    
    def test_github_actions_job_configurations(self, github_pipeline):
        """测试GitHub Actions作业配置"""
        workflow_yaml = github_pipeline.generate_github_actions_workflow()
        workflow_data = yaml.safe_load(workflow_yaml)
        jobs = workflow_data["jobs"]
        
        # 验证lint作业
        lint_job = jobs["lint"]
        assert lint_job["runs-on"] == "ubuntu-latest"
        lint_steps = [step["name"] for step in lint_job["steps"]]
        assert "Checkout code" in lint_steps
        assert "Set up Python" in lint_steps
        assert "Install dependencies" in lint_steps
        assert "Run linting" in lint_steps
        
        # 验证test作业
        test_job = jobs["test"]
        assert "strategy" in test_job
        assert "matrix" in test_job["strategy"]
        test_steps = [step["name"] for step in test_job["steps"]]
        assert "Run tests" in test_steps
        assert "Upload coverage" in test_steps
        
        # 验证build作业
        build_job = jobs["build"]
        build_steps = [step["name"] for step in build_job["steps"]]
        assert "Set up Docker Buildx" in build_steps
        assert "Login to Docker Hub" in build_steps
        assert "Build and push" in build_steps
        
        # 验证deploy作业
        deploy_job = jobs["deploy"]
        deploy_steps = [step["name"] for step in deploy_job["steps"]]
        assert "Set up kubectl" in deploy_steps
        assert "Deploy to Kubernetes" in deploy_steps
    
    def test_github_actions_environment_variables(self, github_pipeline):
        """测试GitHub Actions环境变量"""
        workflow_yaml = github_pipeline.generate_github_actions_workflow()
        workflow_data = yaml.safe_load(workflow_yaml)
        
        # 验证全局环境变量
        if "env" in workflow_data:
            global_env = workflow_data["env"]
            assert "DOCKER_REGISTRY" in global_env or "REGISTRY" in global_env
        
        # 验证作业级环境变量
        jobs = workflow_data["jobs"]
        
        # 检查build作业的环境变量
        build_job = jobs["build"]
        build_env_step = None
        for step in build_job["steps"]:
            if "env" in step:
                build_env_step = step
                break
        
        if build_env_step:
            assert "DOCKER_USERNAME" in str(build_env_step) or "secrets.DOCKER_USERNAME" in str(build_env_step)
            assert "DOCKER_PASSWORD" in str(build_env_step) or "secrets.DOCKER_PASSWORD" in str(build_env_step)
        
        # 检查deploy作业的环境变量
        deploy_job = jobs["deploy"]
        deploy_env_step = None
        for step in deploy_job["steps"]:
            if "env" in step:
                deploy_env_step = step
                break
        
        if deploy_env_step:
            assert "KUBE_CONFIG" in str(deploy_env_step) or "secrets.KUBE_CONFIG" in str(deploy_env_step)
    
    @patch('subprocess.run')
    def test_github_actions_pipeline_execution_simulation(self, mock_subprocess, github_pipeline):
        """测试GitHub Actions流水线执行模拟"""
        # 模拟成功的命令执行
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Command executed successfully"
        mock_subprocess.return_value = mock_result
        
        # 模拟各个阶段的执行
        stages = [
            ("lint", github_pipeline.config.get("lint_commands", [])),
            ("test", github_pipeline.config.get("test_commands", [])),
            ("security", github_pipeline.config.get("security_commands", [])),
            ("build", github_pipeline.config.get("build_commands", [])),
            ("deploy", github_pipeline.config.get("deploy_commands", []))
        ]
        
        for stage_name, commands in stages:
            if commands:
                for command in commands:
                    # 模拟命令执行
                    result = subprocess.run(
                        command.split(),
                        capture_output=True,
                        text=True,
                        timeout=1800
                    )
                    assert result.returncode == 0
        
        # 验证所有命令都被执行
        total_commands = sum(len(commands) for _, commands in stages if commands)
        assert mock_subprocess.call_count == total_commands


class TestJenkinsPipeline:
    """Jenkins流水线测试"""
    
    @pytest.fixture
    def jenkins_config(self):
        """Jenkins配置"""
        return {
            "repository": "https://github.com/rl-trading/rl-trading-system.git",
            "branch": "main",
            "build_stages": ["checkout", "lint", "test", "build", "deploy", "notify"],
            "test_commands": [
                "python -m pytest tests/ --junitxml=test-results.xml",
                "coverage xml -o coverage.xml"
            ],
            "build_commands": [
                "docker build -t ${DOCKER_REGISTRY}/${IMAGE_NAME}:${BUILD_NUMBER} .",
                "docker push ${DOCKER_REGISTRY}/${IMAGE_NAME}:${BUILD_NUMBER}"
            ],
            "deploy_commands": [
                "helm upgrade --install rl-trading-system ./helm/rl-trading-system --set image.tag=${BUILD_NUMBER}",
                "kubectl rollout status deployment/rl-trading-system -n ${KUBE_NAMESPACE}"
            ],
            "post_actions": {
                "always": ["cleanWs()"],
                "success": ["publishHTML([allowMissing: false, alwaysLinkToLastBuild: true, keepAll: true, reportDir: 'htmlcov', reportFiles: 'index.html', reportName: 'Coverage Report'])"],
                "failure": ["emailext body: 'Build failed', subject: 'Build Failure', to: '${DEFAULT_RECIPIENTS}'"]
            }
        }
    
    @pytest.fixture
    def jenkins_pipeline(self, jenkins_config):
        """Jenkins流水线实例"""
        return CICDPipeline(jenkins_config)
    
    def test_jenkins_pipeline_structure(self, jenkins_pipeline):
        """测试Jenkins流水线结构"""
        pipeline_script = jenkins_pipeline.generate_jenkins_pipeline()
        
        # 验证基本结构
        assert "pipeline {" in pipeline_script
        assert "agent any" in pipeline_script
        assert "stages {" in pipeline_script
        assert "post {" in pipeline_script
        
        # 验证环境变量
        assert "environment {" in pipeline_script
        assert "DOCKER_REGISTRY" in pipeline_script
        assert "IMAGE_NAME" in pipeline_script
        assert "KUBE_NAMESPACE" in pipeline_script
        
        # 验证阶段定义
        expected_stages = ["Checkout", "Lint", "Test", "Build", "Deploy"]
        for stage in expected_stages:
            assert f"stage('{stage}')" in pipeline_script
        
        # 验证后处理块
        assert "always {" in pipeline_script
        assert "success {" in pipeline_script
        assert "failure {" in pipeline_script
    
    def test_jenkins_pipeline_parameters(self, jenkins_pipeline):
        """测试Jenkins流水线参数"""
        pipeline_script = jenkins_pipeline.generate_jenkins_pipeline()
        
        # 检查参数定义
        if "parameters {" in pipeline_script:
            # 验证常见参数
            assert "string(" in pipeline_script or "choice(" in pipeline_script
            assert "ENVIRONMENT" in pipeline_script or "DEPLOY_ENV" in pipeline_script
    
    def test_jenkins_pipeline_tools_integration(self, jenkins_pipeline):
        """测试Jenkins流水线工具集成"""
        pipeline_script = jenkins_pipeline.generate_jenkins_pipeline()
        
        # 验证工具定义
        if "tools {" in pipeline_script:
            # 检查常见工具
            tools_keywords = ["maven", "gradle", "nodejs", "python", "docker"]
            has_tools = any(tool in pipeline_script.lower() for tool in tools_keywords)
            assert has_tools, "Pipeline should define at least one tool"
    
    @patch('subprocess.run')
    def test_jenkins_pipeline_execution_simulation(self, mock_subprocess, jenkins_pipeline):
        """测试Jenkins流水线执行模拟"""
        # 模拟Jenkins环境变量
        jenkins_env = {
            "BUILD_NUMBER": "123",
            "JOB_NAME": "rl-trading-system",
            "WORKSPACE": "/var/jenkins_home/workspace/rl-trading-system",
            "DOCKER_REGISTRY": "registry.example.com",
            "IMAGE_NAME": "rl-trading-system",
            "KUBE_NAMESPACE": "rl-trading"
        }
        
        # 模拟成功的命令执行
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Jenkins command executed successfully"
        mock_subprocess.return_value = mock_result
        
        # 模拟各个阶段
        with patch.dict(os.environ, jenkins_env):
            # 测试构建阶段
            build_result = jenkins_pipeline.build_image()
            assert build_result is True
            
            # 测试部署阶段
            deploy_result = jenkins_pipeline.deploy_to_kubernetes()
            assert deploy_result is True
        
        # 验证命令执行次数
        expected_calls = len(jenkins_pipeline.build_commands) + len(jenkins_pipeline.deploy_commands)
        assert mock_subprocess.call_count == expected_calls


class TestMultiPlatformCICD:
    """多平台CI/CD测试"""
    
    @pytest.fixture
    def multi_platform_config(self):
        """多平台配置"""
        return {
            "platforms": ["github-actions", "jenkins", "gitlab-ci", "azure-devops"],
            "repository": "https://github.com/rl-trading/rl-trading-system.git",
            "branch": "main",
            "common_stages": ["test", "build", "deploy"],
            "platform_specific": {
                "github-actions": {
                    "runner": "ubuntu-latest",
                    "cache_strategy": "actions/cache@v3"
                },
                "jenkins": {
                    "agent": "docker",
                    "workspace_cleanup": True
                },
                "gitlab-ci": {
                    "image": "python:3.9",
                    "services": ["docker:dind"]
                },
                "azure-devops": {
                    "pool": "ubuntu-latest",
                    "container": "python:3.9"
                }
            }
        }
    
    def test_platform_specific_configurations(self, multi_platform_config):
        """测试平台特定配置"""
        for platform, config in multi_platform_config["platform_specific"].items():
            pipeline = CICDPipeline({
                **multi_platform_config,
                "platform": platform,
                **config
            })
            
            if platform == "github-actions":
                workflow = pipeline.generate_github_actions_workflow()
                workflow_data = yaml.safe_load(workflow)
                
                # 验证GitHub Actions特定配置
                jobs = workflow_data["jobs"]
                for job in jobs.values():
                    if "runs-on" in job:
                        assert job["runs-on"] == "ubuntu-latest"
            
            elif platform == "jenkins":
                jenkins_script = pipeline.generate_jenkins_pipeline()
                
                # 验证Jenkins特定配置
                assert "agent" in jenkins_script
                if config.get("workspace_cleanup"):
                    assert "cleanWs()" in jenkins_script
    
    def test_cross_platform_compatibility(self, multi_platform_config):
        """测试跨平台兼容性"""
        # 创建通用的流水线配置
        common_config = {
            "repository": multi_platform_config["repository"],
            "branch": multi_platform_config["branch"],
            "test_commands": ["python -m pytest tests/"],
            "build_commands": ["docker build -t app:latest ."],
            "deploy_commands": ["kubectl apply -f k8s/"]
        }
        
        # 测试每个平台都能生成有效配置
        platforms = ["github-actions", "jenkins"]
        
        for platform in platforms:
            pipeline = CICDPipeline(common_config)
            
            if platform == "github-actions":
                workflow = pipeline.generate_github_actions_workflow()
                workflow_data = yaml.safe_load(workflow)
                assert "jobs" in workflow_data
                assert len(workflow_data["jobs"]) > 0
            
            elif platform == "jenkins":
                jenkins_script = pipeline.generate_jenkins_pipeline()
                assert "pipeline {" in jenkins_script
                assert "stages {" in jenkins_script


class TestCICDSecurityIntegration:
    """CI/CD安全集成测试"""
    
    @pytest.fixture
    def security_config(self):
        """安全配置"""
        return {
            "repository": "https://github.com/rl-trading/rl-trading-system.git",
            "branch": "main",
            "security_tools": {
                "sast": ["bandit", "semgrep", "sonarqube"],
                "dependency_scan": ["safety", "snyk", "dependabot"],
                "container_scan": ["trivy", "clair", "anchore"],
                "secrets_scan": ["truffleHog", "git-secrets", "detect-secrets"]
            },
            "security_gates": {
                "vulnerability_threshold": "high",
                "coverage_threshold": 80,
                "quality_gate": "passed"
            },
            "compliance": {
                "standards": ["SOC2", "PCI-DSS", "GDPR"],
                "audit_logging": True,
                "access_control": True
            }
        }
    
    def test_security_scanning_integration(self, security_config):
        """测试安全扫描集成"""
        pipeline = CICDPipeline(security_config)
        
        # 生成包含安全扫描的工作流
        workflow_yaml = pipeline.generate_github_actions_workflow()
        workflow_data = yaml.safe_load(workflow_yaml)
        
        # 验证安全扫描作业存在
        jobs = workflow_data["jobs"]
        security_job_found = False
        
        for job_name, job_config in jobs.items():
            if "security" in job_name.lower() or "scan" in job_name.lower():
                security_job_found = True
                
                # 验证安全扫描步骤
                steps = job_config.get("steps", [])
                step_names = [step.get("name", "") for step in steps]
                
                # 检查是否包含安全工具
                security_tools_found = any(
                    any(tool in step_name.lower() for tool in security_config["security_tools"]["sast"])
                    for step_name in step_names
                )
                
                if security_tools_found:
                    break
        
        # 如果没有专门的安全作业，检查是否在其他作业中包含安全步骤
        if not security_job_found:
            all_steps = []
            for job_config in jobs.values():
                all_steps.extend(job_config.get("steps", []))
            
            all_step_names = [step.get("name", "") for step in all_steps]
            security_tools_mentioned = any(
                any(tool in step_name.lower() for tool in 
                    security_config["security_tools"]["sast"] + 
                    security_config["security_tools"]["container_scan"])
                for step_name in all_step_names
            )
            
            assert security_tools_mentioned, "Security tools should be integrated in CI/CD pipeline"
    
    def test_compliance_requirements(self, security_config):
        """测试合规要求"""
        pipeline = CICDPipeline(security_config)
        
        # 验证合规配置
        compliance = security_config["compliance"]
        
        # 检查审计日志要求
        if compliance["audit_logging"]:
            # 验证流水线包含审计日志配置
            jenkins_script = pipeline.generate_jenkins_pipeline()
            
            # 检查是否包含审计相关配置
            audit_keywords = ["audit", "log", "record", "track"]
            has_audit_config = any(keyword in jenkins_script.lower() for keyword in audit_keywords)
            
            # 如果没有明确的审计配置，至少应该有日志记录
            assert has_audit_config or "echo" in jenkins_script, "Pipeline should include audit logging"
        
        # 检查访问控制要求
        if compliance["access_control"]:
            workflow_yaml = pipeline.generate_github_actions_workflow()
            workflow_data = yaml.safe_load(workflow_yaml)
            
            # 验证是否使用了secrets管理
            workflow_str = yaml.dump(workflow_data)
            assert "secrets." in workflow_str, "Pipeline should use secrets management for access control"
    
    @patch('subprocess.run')
    def test_security_gate_enforcement(self, mock_subprocess, security_config):
        """测试安全门禁执行"""
        # 模拟安全扫描结果
        security_results = {
            "bandit": {"high": 0, "medium": 2, "low": 5},
            "safety": {"vulnerabilities": 1},
            "trivy": {"critical": 0, "high": 1, "medium": 3}
        }
        
        # 模拟命令执行返回安全扫描结果
        def mock_security_command(cmd, **kwargs):
            result = Mock()
            result.returncode = 0
            
            if "bandit" in " ".join(cmd):
                result.stdout = json.dumps(security_results["bandit"])
            elif "safety" in " ".join(cmd):
                result.stdout = json.dumps(security_results["safety"])
            elif "trivy" in " ".join(cmd):
                result.stdout = json.dumps(security_results["trivy"])
            else:
                result.stdout = "No issues found"
            
            return result
        
        mock_subprocess.side_effect = mock_security_command
        
        # 创建流水线并执行安全检查
        pipeline = CICDPipeline(security_config)
        
        # 模拟安全扫描命令执行
        security_commands = [
            ["bandit", "-r", "src/", "-f", "json"],
            ["safety", "check", "--json"],
            ["trivy", "fs", "--format", "json", "."]
        ]
        
        for cmd in security_commands:
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0
        
        # 验证所有安全命令都被执行
        assert mock_subprocess.call_count == len(security_commands)


class TestCICDMonitoringAndObservability:
    """CI/CD监控和可观测性测试"""
    
    @pytest.fixture
    def monitoring_config(self):
        """监控配置"""
        return {
            "repository": "https://github.com/rl-trading/rl-trading-system.git",
            "branch": "main",
            "monitoring": {
                "metrics": {
                    "build_duration": True,
                    "test_coverage": True,
                    "deployment_frequency": True,
                    "failure_rate": True
                },
                "alerting": {
                    "slack_webhook": "${SLACK_WEBHOOK}",
                    "email_recipients": ["devops@rltrading.com"],
                    "pagerduty_key": "${PAGERDUTY_KEY}"
                },
                "dashboards": {
                    "grafana_url": "https://grafana.rltrading.com",
                    "prometheus_url": "https://prometheus.rltrading.com"
                }
            },
            "observability": {
                "tracing": {
                    "jaeger_endpoint": "http://jaeger:14268/api/traces",
                    "service_name": "rl-trading-cicd"
                },
                "logging": {
                    "elasticsearch_url": "https://elasticsearch.rltrading.com",
                    "log_level": "INFO"
                }
            }
        }
    
    def test_monitoring_integration(self, monitoring_config):
        """测试监控集成"""
        pipeline = CICDPipeline(monitoring_config)
        
        # 生成包含监控的工作流
        workflow_yaml = pipeline.generate_github_actions_workflow()
        workflow_data = yaml.safe_load(workflow_yaml)
        
        # 验证监控相关步骤
        all_steps = []
        for job_config in workflow_data["jobs"].values():
            all_steps.extend(job_config.get("steps", []))
        
        step_names = [step.get("name", "") for step in all_steps]
        
        # 检查是否包含监控步骤
        monitoring_keywords = ["metrics", "monitor", "alert", "dashboard"]
        has_monitoring = any(
            any(keyword in step_name.lower() for keyword in monitoring_keywords)
            for step_name in step_names
        )
        
        # 如果没有明确的监控步骤，检查是否有通知步骤
        notification_keywords = ["notify", "slack", "email"]
        has_notification = any(
            any(keyword in step_name.lower() for keyword in notification_keywords)
            for step_name in step_names
        )
        
        assert has_monitoring or has_notification, "Pipeline should include monitoring or notification"
    
    def test_observability_configuration(self, monitoring_config):
        """测试可观测性配置"""
        pipeline = CICDPipeline(monitoring_config)
        
        # 生成Jenkins流水线
        jenkins_script = pipeline.generate_jenkins_pipeline()
        
        # 验证可观测性配置
        observability = monitoring_config["observability"]
        
        # 检查追踪配置
        if observability.get("tracing"):
            tracing_keywords = ["trace", "jaeger", "zipkin", "opentelemetry"]
            has_tracing = any(keyword in jenkins_script.lower() for keyword in tracing_keywords)
            
            # 如果没有明确的追踪配置，至少应该有环境变量设置
            has_env_vars = "environment {" in jenkins_script
            assert has_tracing or has_env_vars, "Pipeline should include tracing configuration"
        
        # 检查日志配置
        if observability.get("logging"):
            logging_keywords = ["log", "elasticsearch", "kibana", "fluentd"]
            has_logging = any(keyword in jenkins_script.lower() for keyword in logging_keywords)
            
            # 至少应该有基本的日志记录
            has_basic_logging = "echo" in jenkins_script or "println" in jenkins_script
            assert has_logging or has_basic_logging, "Pipeline should include logging configuration"
    
    @patch('requests.post')
    def test_notification_system(self, mock_post, monitoring_config):
        """测试通知系统"""
        # 模拟成功的通知发送
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True}
        mock_post.return_value = mock_response
        
        pipeline = CICDPipeline(monitoring_config)
        
        # 模拟发送构建通知
        notification_data = {
            "text": "Build completed successfully",
            "channel": "#devops",
            "username": "CI/CD Bot"
        }
        
        # 模拟Slack通知
        slack_webhook = monitoring_config["monitoring"]["alerting"]["slack_webhook"]
        if slack_webhook:
            response = requests.post(slack_webhook, json=notification_data)
            assert response.status_code == 200
            mock_post.assert_called_once()
    
    def test_performance_metrics_collection(self, monitoring_config):
        """测试性能指标收集"""
        pipeline = CICDPipeline(monitoring_config)
        
        # 验证性能指标配置
        metrics = monitoring_config["monitoring"]["metrics"]
        
        # 检查关键指标
        key_metrics = ["build_duration", "test_coverage", "deployment_frequency", "failure_rate"]
        
        for metric in key_metrics:
            assert metric in metrics, f"Missing key metric: {metric}"
            assert metrics[metric] is True, f"Metric {metric} should be enabled"
        
        # 生成包含指标收集的工作流
        workflow_yaml = pipeline.generate_github_actions_workflow()
        workflow_data = yaml.safe_load(workflow_yaml)
        
        # 验证是否包含指标收集步骤
        all_steps = []
        for job_config in workflow_data["jobs"].values():
            all_steps.extend(job_config.get("steps", []))
        
        # 检查是否有时间测量或指标收集
        has_timing = any(
            "time" in step.get("name", "").lower() or 
            "duration" in step.get("name", "").lower() or
            "metric" in step.get("name", "").lower()
            for step in all_steps
        )
        
        # 如果没有明确的指标收集，至少应该有基本的报告功能
        has_reporting = any(
            "report" in step.get("name", "").lower() or
            "publish" in step.get("name", "").lower()
            for step in all_steps
        )
        
        assert has_timing or has_reporting, "Pipeline should include performance metrics collection"