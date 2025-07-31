#!/usr/bin/env python3
"""
容器化部署测试用例

测试Docker容器构建和运行、Kubernetes部署和服务发现、CI/CD流水线和自动化部署
需求: 8.1, 8.4
"""

import pytest
import subprocess
import tempfile
import yaml
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from src.rl_trading_system.deployment.containerized_deployment import (
    DockerManager,
    KubernetesManager,
    CICDPipeline,
    HealthChecker,
    ServiceDiscovery,
    ContainerConfig,
    KubernetesConfig,
    DeploymentStatus,
    ServiceStatus
)


class TestDockerManager:
    """Docker管理器测试"""
    
    @pytest.fixture
    def container_config(self):
        """创建容器配置"""
        return ContainerConfig(
            image_name="rl-trading-system",
            image_tag="v1.0.0",
            dockerfile_path="./Dockerfile",
            build_context=".",
            environment_vars={
                "PYTHONPATH": "/app/src",
                "LOG_LEVEL": "INFO"
            },
            ports={"8000": "8000", "8888": "8888"},
            volumes={
                "./data": "/app/data",
                "./logs": "/app/logs"
            },
            health_check_cmd="curl -f http://localhost:8000/health || exit 1",
            health_check_interval=30,
            health_check_timeout=30,
            health_check_retries=3
        )
    
    @pytest.fixture
    def docker_manager(self, container_config):
        """创建Docker管理器"""
        return DockerManager(container_config)
    
    def test_docker_manager_initialization(self, docker_manager, container_config):
        """测试Docker管理器初始化"""
        assert docker_manager.config == container_config
        assert docker_manager.client is not None
        assert docker_manager.image_name == "rl-trading-system:v1.0.0"
    
    @patch('docker.from_env')
    def test_build_image_success(self, mock_docker, docker_manager):
        """测试成功构建Docker镜像"""
        # 模拟Docker客户端
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        # 模拟构建成功
        mock_image = Mock()
        mock_image.id = "sha256:abc123"
        mock_client.images.build.return_value = (mock_image, [])
        
        # 执行构建
        result = docker_manager.build_image()
        
        # 验证结果
        assert result is True
        mock_client.images.build.assert_called_once_with(
            path=".",
            dockerfile="./Dockerfile",
            tag="rl-trading-system:v1.0.0",
            rm=True,
            forcerm=True
        )
    
    @patch('docker.from_env')
    def test_build_image_failure(self, mock_docker, docker_manager):
        """测试Docker镜像构建失败"""
        # 模拟Docker客户端
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        # 模拟构建失败
        mock_client.images.build.side_effect = Exception("Build failed")
        
        # 执行构建
        result = docker_manager.build_image()
        
        # 验证结果
        assert result is False
    
    @patch('docker.from_env')
    def test_run_container_success(self, mock_docker, docker_manager):
        """测试成功运行容器"""
        # 模拟Docker客户端
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        # 模拟容器运行
        mock_container = Mock()
        mock_container.id = "container123"
        mock_container.status = "running"
        mock_client.containers.run.return_value = mock_container
        
        # 执行运行
        container_id = docker_manager.run_container("test-container")
        
        # 验证结果
        assert container_id == "container123"
        mock_client.containers.run.assert_called_once_with(
            image="rl-trading-system:v1.0.0",
            name="test-container",
            ports={"8000": "8000", "8888": "8888"},
            volumes={
                "./data": {"bind": "/app/data", "mode": "rw"},
                "./logs": {"bind": "/app/logs", "mode": "rw"}
            },
            environment={
                "PYTHONPATH": "/app/src",
                "LOG_LEVEL": "INFO"
            },
            detach=True,
            restart_policy={"Name": "unless-stopped"}
        )
    
    @patch('docker.from_env')
    def test_stop_container(self, mock_docker, docker_manager):
        """测试停止容器"""
        # 模拟Docker客户端
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        # 模拟容器
        mock_container = Mock()
        mock_client.containers.get.return_value = mock_container
        
        # 执行停止
        result = docker_manager.stop_container("container123")
        
        # 验证结果
        assert result is True
        mock_container.stop.assert_called_once()
        mock_container.remove.assert_called_once()
    
    @patch('docker.from_env')
    def test_get_container_status(self, mock_docker, docker_manager):
        """测试获取容器状态"""
        # 模拟Docker客户端
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        # 模拟容器状态
        mock_container = Mock()
        mock_container.status = "running"
        mock_container.attrs = {
            "State": {
                "Health": {"Status": "healthy"}
            }
        }
        mock_client.containers.get.return_value = mock_container
        
        # 执行获取状态
        status = docker_manager.get_container_status("container123")
        
        # 验证结果
        assert status["status"] == "running"
        assert status["health"] == "healthy"
    
    def test_generate_dockerfile(self, docker_manager):
        """测试生成Dockerfile"""
        dockerfile_content = docker_manager.generate_dockerfile()
        
        # 验证Dockerfile内容
        assert "FROM python:3.9-slim" in dockerfile_content
        assert "WORKDIR /app" in dockerfile_content
        assert "COPY requirements.txt ." in dockerfile_content
        assert "RUN pip install --no-cache-dir -r requirements.txt" in dockerfile_content
        assert "EXPOSE 8000 8888" in dockerfile_content
        assert "HEALTHCHECK" in dockerfile_content
    
    def test_generate_docker_compose(self, docker_manager):
        """测试生成docker-compose.yml"""
        compose_content = docker_manager.generate_docker_compose()
        
        # 解析YAML内容
        compose_data = yaml.safe_load(compose_content)
        
        # 验证compose文件结构
        assert "version" in compose_data
        assert "services" in compose_data
        assert "rl-trading-system" in compose_data["services"]
        
        service = compose_data["services"]["rl-trading-system"]
        assert "ports" in service
        assert "volumes" in service
        assert "environment" in service


class TestKubernetesManager:
    """Kubernetes管理器测试"""
    
    @pytest.fixture
    def k8s_config(self):
        """创建Kubernetes配置"""
        return KubernetesConfig(
            namespace="rl-trading",
            deployment_name="rl-trading-system",
            service_name="rl-trading-service",
            image_name="rl-trading-system:v1.0.0",
            replicas=3,
            cpu_request="500m",
            cpu_limit="2000m",
            memory_request="1Gi",
            memory_limit="4Gi",
            ports=[8000, 8888],
            environment_vars={
                "PYTHONPATH": "/app/src",
                "LOG_LEVEL": "INFO"
            },
            config_maps=["trading-config"],
            secrets=["trading-secrets"],
            health_check_path="/health",
            readiness_check_path="/ready"
        )
    
    @pytest.fixture
    def k8s_manager(self, k8s_config):
        """创建Kubernetes管理器"""
        return KubernetesManager(k8s_config)
    
    def test_k8s_manager_initialization(self, k8s_manager, k8s_config):
        """测试Kubernetes管理器初始化"""
        assert k8s_manager.config == k8s_config
        assert k8s_manager.namespace == "rl-trading"
    
    @patch('kubernetes.config.load_incluster_config')
    @patch('kubernetes.client.AppsV1Api')
    def test_create_deployment_success(self, mock_apps_api, mock_load_config, k8s_manager):
        """测试成功创建Kubernetes部署"""
        # 模拟Kubernetes API
        mock_api = Mock()
        mock_apps_api.return_value = mock_api
        
        # 模拟部署创建成功
        mock_deployment = Mock()
        mock_deployment.metadata.name = "rl-trading-system"
        mock_api.create_namespaced_deployment.return_value = mock_deployment
        
        # 执行创建部署
        result = k8s_manager.create_deployment()
        
        # 验证结果
        assert result is True
        mock_api.create_namespaced_deployment.assert_called_once()
    
    @patch('kubernetes.config.load_incluster_config')
    @patch('kubernetes.client.CoreV1Api')
    def test_create_service_success(self, mock_core_api, mock_load_config, k8s_manager):
        """测试成功创建Kubernetes服务"""
        # 模拟Kubernetes API
        mock_api = Mock()
        mock_core_api.return_value = mock_api
        
        # 模拟服务创建成功
        mock_service = Mock()
        mock_service.metadata.name = "rl-trading-service"
        mock_api.create_namespaced_service.return_value = mock_service
        
        # 执行创建服务
        result = k8s_manager.create_service()
        
        # 验证结果
        assert result is True
        mock_api.create_namespaced_service.assert_called_once()
    
    @patch('kubernetes.config.load_incluster_config')
    @patch('kubernetes.client.AppsV1Api')
    def test_update_deployment(self, mock_apps_api, mock_load_config, k8s_manager):
        """测试更新Kubernetes部署"""
        # 模拟Kubernetes API
        mock_api = Mock()
        mock_apps_api.return_value = mock_api
        
        # 模拟部署更新成功
        mock_deployment = Mock()
        mock_api.patch_namespaced_deployment.return_value = mock_deployment
        
        # 执行更新部署
        result = k8s_manager.update_deployment("rl-trading-system:v1.1.0")
        
        # 验证结果
        assert result is True
        mock_api.patch_namespaced_deployment.assert_called_once()
    
    @patch('kubernetes.config.load_incluster_config')
    @patch('kubernetes.client.AppsV1Api')
    def test_scale_deployment(self, mock_apps_api, mock_load_config, k8s_manager):
        """测试扩缩容部署"""
        # 模拟Kubernetes API
        mock_api = Mock()
        mock_apps_api.return_value = mock_api
        
        # 模拟扩缩容成功
        mock_deployment = Mock()
        mock_api.patch_namespaced_deployment_scale.return_value = mock_deployment
        
        # 执行扩缩容
        result = k8s_manager.scale_deployment(5)
        
        # 验证结果
        assert result is True
        mock_api.patch_namespaced_deployment_scale.assert_called_once()
    
    @patch('kubernetes.config.load_incluster_config')
    @patch('kubernetes.client.AppsV1Api')
    def test_get_deployment_status(self, mock_apps_api, mock_load_config, k8s_manager):
        """测试获取部署状态"""
        # 模拟Kubernetes API
        mock_api = Mock()
        mock_apps_api.return_value = mock_api
        
        # 模拟部署状态
        mock_deployment = Mock()
        mock_deployment.status.replicas = 3
        mock_deployment.status.ready_replicas = 3
        mock_deployment.status.available_replicas = 3
        mock_api.read_namespaced_deployment.return_value = mock_deployment
        
        # 执行获取状态
        status = k8s_manager.get_deployment_status()
        
        # 验证结果
        assert status["replicas"] == 3
        assert status["ready_replicas"] == 3
        assert status["available_replicas"] == 3
        assert status["is_ready"] is True
    
    def test_generate_deployment_yaml(self, k8s_manager):
        """测试生成部署YAML"""
        deployment_yaml = k8s_manager.generate_deployment_yaml()
        
        # 解析YAML内容
        deployment_data = yaml.safe_load(deployment_yaml)
        
        # 验证部署配置
        assert deployment_data["kind"] == "Deployment"
        assert deployment_data["metadata"]["name"] == "rl-trading-system"
        assert deployment_data["spec"]["replicas"] == 3
        
        container = deployment_data["spec"]["template"]["spec"]["containers"][0]
        assert container["image"] == "rl-trading-system:v1.0.0"
        assert container["resources"]["requests"]["cpu"] == "500m"
        assert container["resources"]["limits"]["memory"] == "4Gi"
    
    def test_generate_service_yaml(self, k8s_manager):
        """测试生成服务YAML"""
        service_yaml = k8s_manager.generate_service_yaml()
        
        # 解析YAML内容
        service_data = yaml.safe_load(service_yaml)
        
        # 验证服务配置
        assert service_data["kind"] == "Service"
        assert service_data["metadata"]["name"] == "rl-trading-service"
        assert len(service_data["spec"]["ports"]) == 2
        assert service_data["spec"]["selector"]["app"] == "rl-trading-system"


class TestCICDPipeline:
    """CI/CD流水线测试"""
    
    @pytest.fixture
    def pipeline_config(self):
        """创建流水线配置"""
        return {
            "repository": "https://github.com/rl-trading/rl-trading-system.git",
            "branch": "main",
            "build_stages": ["test", "build", "deploy"],
            "test_commands": ["pytest tests/", "flake8 src/", "mypy src/"],
            "build_commands": ["docker build -t rl-trading-system:latest ."],
            "deploy_commands": ["kubectl apply -f k8s/"],
            "notifications": {
                "slack_webhook": "https://hooks.slack.com/services/xxx",
                "email": "team@rltrading.com"
            }
        }
    
    @pytest.fixture
    def cicd_pipeline(self, pipeline_config):
        """创建CI/CD流水线"""
        return CICDPipeline(pipeline_config)
    
    def test_pipeline_initialization(self, cicd_pipeline, pipeline_config):
        """测试流水线初始化"""
        assert cicd_pipeline.config == pipeline_config
        assert cicd_pipeline.repository == "https://github.com/rl-trading/rl-trading-system.git"
        assert cicd_pipeline.branch == "main"
    
    @patch('subprocess.run')
    def test_run_tests_success(self, mock_subprocess, cicd_pipeline):
        """测试成功运行测试"""
        # 模拟测试成功
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "All tests passed"
        mock_subprocess.return_value = mock_result
        
        # 执行测试
        result = cicd_pipeline.run_tests()
        
        # 验证结果
        assert result is True
        assert mock_subprocess.call_count == 3  # 三个测试命令
    
    @patch('subprocess.run')
    def test_run_tests_failure(self, mock_subprocess, cicd_pipeline):
        """测试测试失败"""
        # 模拟测试失败
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = "Test failed"
        mock_subprocess.return_value = mock_result
        
        # 执行测试
        result = cicd_pipeline.run_tests()
        
        # 验证结果
        assert result is False
    
    @patch('subprocess.run')
    def test_build_image_success(self, mock_subprocess, cicd_pipeline):
        """测试成功构建镜像"""
        # 模拟构建成功
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Successfully built"
        mock_subprocess.return_value = mock_result
        
        # 执行构建
        result = cicd_pipeline.build_image()
        
        # 验证结果
        assert result is True
        mock_subprocess.assert_called_with(
            ["docker", "build", "-t", "rl-trading-system:latest", "."],
            capture_output=True,
            text=True,
            timeout=1800
        )
    
    @patch('subprocess.run')
    def test_deploy_to_kubernetes(self, mock_subprocess, cicd_pipeline):
        """测试部署到Kubernetes"""
        # 模拟部署成功
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "deployment.apps/rl-trading-system created"
        mock_subprocess.return_value = mock_result
        
        # 执行部署
        result = cicd_pipeline.deploy_to_kubernetes()
        
        # 验证结果
        assert result is True
        mock_subprocess.assert_called_with(
            ["kubectl", "apply", "-f", "k8s/"],
            capture_output=True,
            text=True,
            timeout=600
        )
    
    def test_generate_github_actions_workflow(self, cicd_pipeline):
        """测试生成GitHub Actions工作流"""
        workflow_yaml = cicd_pipeline.generate_github_actions_workflow()
        
        # 解析YAML内容
        workflow_data = yaml.safe_load(workflow_yaml)
        
        # 验证工作流配置
        assert workflow_data["name"] == "CI/CD Pipeline"
        assert "push" in workflow_data["on"]
        assert "pull_request" in workflow_data["on"]
        
        jobs = workflow_data["jobs"]
        assert "test" in jobs
        assert "build" in jobs
        assert "deploy" in jobs
    
    def test_generate_jenkins_pipeline(self, cicd_pipeline):
        """测试生成Jenkins流水线"""
        pipeline_script = cicd_pipeline.generate_jenkins_pipeline()
        
        # 验证流水线脚本
        assert "pipeline {" in pipeline_script
        assert "agent any" in pipeline_script
        assert "stage('Test')" in pipeline_script
        assert "stage('Build')" in pipeline_script
        assert "stage('Deploy')" in pipeline_script


class TestHealthChecker:
    """健康检查器测试"""
    
    @pytest.fixture
    def health_checker(self):
        """创建健康检查器"""
        return HealthChecker(
            endpoints=[
                "http://localhost:8000/health",
                "http://localhost:8888/health"
            ],
            timeout=30,
            retry_count=3,
            retry_interval=5
        )
    
    @patch('requests.get')
    def test_check_endpoint_healthy(self, mock_get, health_checker):
        """测试端点健康检查成功"""
        # 模拟健康响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        mock_get.return_value = mock_response
        
        # 执行健康检查
        result = health_checker.check_endpoint("http://localhost:8000/health")
        
        # 验证结果
        assert result is True
        mock_get.assert_called_once_with(
            "http://localhost:8000/health",
            timeout=30
        )
    
    @patch('requests.get')
    def test_check_endpoint_unhealthy(self, mock_get, health_checker):
        """测试端点健康检查失败"""
        # 模拟不健康响应
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        # 执行健康检查
        result = health_checker.check_endpoint("http://localhost:8000/health")
        
        # 验证结果
        assert result is False
    
    @patch('requests.get')
    def test_check_all_endpoints(self, mock_get, health_checker):
        """测试检查所有端点"""
        # 模拟混合响应
        responses = [
            Mock(status_code=200, json=lambda: {"status": "healthy"}),
            Mock(status_code=500)
        ]
        mock_get.side_effect = responses
        
        # 执行检查
        results = health_checker.check_all_endpoints()
        
        # 验证结果
        assert len(results) == 2
        assert results["http://localhost:8000/health"] is True
        assert results["http://localhost:8888/health"] is False
    
    def test_wait_for_healthy_success(self, health_checker):
        """测试等待健康状态成功"""
        with patch.object(health_checker, 'check_endpoint', return_value=True):
            result = health_checker.wait_for_healthy("http://localhost:8000/health", max_wait=10)
            assert result is True
    
    def test_wait_for_healthy_timeout(self, health_checker):
        """测试等待健康状态超时"""
        with patch.object(health_checker, 'check_endpoint', return_value=False):
            result = health_checker.wait_for_healthy("http://localhost:8000/health", max_wait=1)
            assert result is False


class TestServiceDiscovery:
    """服务发现测试"""
    
    @pytest.fixture
    def service_discovery(self):
        """创建服务发现"""
        return ServiceDiscovery(
            consul_host="localhost",
            consul_port=8500,
            service_name="rl-trading-system",
            service_port=8000,
            health_check_url="http://localhost:8000/health"
        )
    
    @patch('consul.Consul')
    def test_register_service_success(self, mock_consul, service_discovery):
        """测试成功注册服务"""
        # 模拟Consul客户端
        mock_client = Mock()
        mock_consul.return_value = mock_client
        
        # 执行服务注册
        result = service_discovery.register_service()
        
        # 验证结果
        assert result is True
        mock_client.agent.service.register.assert_called_once()
    
    @patch('consul.Consul')
    def test_deregister_service(self, mock_consul, service_discovery):
        """测试注销服务"""
        # 模拟Consul客户端
        mock_client = Mock()
        mock_consul.return_value = mock_client
        
        # 执行服务注销
        result = service_discovery.deregister_service()
        
        # 验证结果
        assert result is True
        mock_client.agent.service.deregister.assert_called_once()
    
    @patch('consul.Consul')
    def test_discover_services(self, mock_consul, service_discovery):
        """测试服务发现"""
        # 模拟Consul客户端
        mock_client = Mock()
        mock_consul.return_value = mock_client
        
        # 模拟服务列表
        mock_services = {
            "rl-trading-system-1": {
                "Service": "rl-trading-system",
                "Address": "192.168.1.10",
                "Port": 8000
            },
            "rl-trading-system-2": {
                "Service": "rl-trading-system",
                "Address": "192.168.1.11",
                "Port": 8000
            }
        }
        mock_client.health.service.return_value = (None, mock_services.values())
        
        # 执行服务发现
        services = service_discovery.discover_services("rl-trading-system")
        
        # 验证结果
        assert len(services) == 2
        assert services[0]["address"] == "192.168.1.10"
        assert services[1]["address"] == "192.168.1.11"


class TestContainerizedDeploymentIntegration:
    """容器化部署集成测试"""
    
    @pytest.fixture
    def temp_project_dir(self):
        """创建临时项目目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)
            
            # 创建基本项目结构
            (project_dir / "src").mkdir()
            (project_dir / "tests").mkdir()
            (project_dir / "k8s").mkdir()
            (project_dir / "config").mkdir()
            (project_dir / "scripts").mkdir()
            
            # 创建基本文件
            (project_dir / "requirements.txt").write_text("torch>=1.12.0\nnumpy>=1.21.0")
            (project_dir / "setup.py").write_text("from setuptools import setup\nsetup(name='test')")
            
            # 创建配置文件
            config_content = """
database:
  host: postgres
  port: 5432
  name: trading_db
  
redis:
  host: redis
  port: 6379
  
influxdb:
  host: influxdb
  port: 8086
"""
            (project_dir / "config" / "system_config.yaml").write_text(config_content)
            
            yield project_dir
    
    def test_full_docker_deployment_flow(self, temp_project_dir):
        """测试完整的Docker部署流程"""
        # 创建容器配置
        config = ContainerConfig(
            image_name="test-app",
            image_tag="v1.0.0",
            dockerfile_path=str(temp_project_dir / "Dockerfile"),
            build_context=str(temp_project_dir)
        )
        
        # 创建Docker管理器
        docker_manager = DockerManager(config)
        
        # 生成Dockerfile
        dockerfile_content = docker_manager.generate_dockerfile()
        (temp_project_dir / "Dockerfile").write_text(dockerfile_content)
        
        # 验证Dockerfile存在
        assert (temp_project_dir / "Dockerfile").exists()
        
        # 生成docker-compose.yml
        compose_content = docker_manager.generate_docker_compose()
        (temp_project_dir / "docker-compose.yml").write_text(compose_content)
        
        # 验证compose文件存在
        assert (temp_project_dir / "docker-compose.yml").exists()
    
    def test_full_kubernetes_deployment_flow(self, temp_project_dir):
        """测试完整的Kubernetes部署流程"""
        # 创建Kubernetes配置
        config = KubernetesConfig(
            namespace="test-namespace",
            deployment_name="test-app",
            service_name="test-service",
            image_name="test-app:v1.0.0"
        )
        
        # 创建Kubernetes管理器
        k8s_manager = KubernetesManager(config)
        
        # 生成部署YAML
        deployment_yaml = k8s_manager.generate_deployment_yaml()
        (temp_project_dir / "k8s" / "deployment.yaml").write_text(deployment_yaml)
        
        # 生成服务YAML
        service_yaml = k8s_manager.generate_service_yaml()
        (temp_project_dir / "k8s" / "service.yaml").write_text(service_yaml)
        
        # 验证YAML文件存在
        assert (temp_project_dir / "k8s" / "deployment.yaml").exists()
        assert (temp_project_dir / "k8s" / "service.yaml").exists()
        
        # 验证YAML内容
        deployment_data = yaml.safe_load((temp_project_dir / "k8s" / "deployment.yaml").read_text())
        assert deployment_data["kind"] == "Deployment"
        assert deployment_data["metadata"]["name"] == "test-app"
    
    def test_cicd_pipeline_integration(self, temp_project_dir):
        """测试CI/CD流水线集成"""
        # 创建流水线配置
        config = {
            "repository": "https://github.com/test/test-app.git",
            "branch": "main",
            "build_stages": ["test", "build", "deploy"]
        }
        
        # 创建CI/CD流水线
        pipeline = CICDPipeline(config)
        
        # 生成GitHub Actions工作流
        workflow_yaml = pipeline.generate_github_actions_workflow()
        workflow_dir = temp_project_dir / ".github" / "workflows"
        workflow_dir.mkdir(parents=True)
        (workflow_dir / "ci-cd.yml").write_text(workflow_yaml)
        
        # 生成Jenkins流水线
        jenkins_script = pipeline.generate_jenkins_pipeline()
        (temp_project_dir / "Jenkinsfile").write_text(jenkins_script)
        
        # 验证文件存在
        assert (workflow_dir / "ci-cd.yml").exists()
        assert (temp_project_dir / "Jenkinsfile").exists()
        
        # 验证工作流内容
        workflow_data = yaml.safe_load((workflow_dir / "ci-cd.yml").read_text())
        assert workflow_data["name"] == "CI/CD Pipeline"
        assert "jobs" in workflow_data
    
    @patch('subprocess.run')
    def test_docker_compose_deployment(self, mock_subprocess, temp_project_dir):
        """测试Docker Compose部署"""
        # 创建docker-compose.yml
        compose_content = """
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app/src
"""
        (temp_project_dir / "docker-compose.yml").write_text(compose_content)
        
        # 模拟成功的docker-compose命令
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Creating network... done"
        mock_subprocess.return_value = mock_result
        
        # 测试docker-compose up
        result = subprocess.run(
            ["docker-compose", "up", "-d"],
            cwd=temp_project_dir,
            capture_output=True,
            text=True
        )
        
        # 验证命令执行
        mock_subprocess.assert_called_with(
            ["docker-compose", "up", "-d"],
            cwd=temp_project_dir,
            capture_output=True,
            text=True
        )
    
    def test_kubernetes_namespace_creation(self, temp_project_dir):
        """测试Kubernetes命名空间创建"""
        # 创建命名空间YAML
        namespace_yaml = """
apiVersion: v1
kind: Namespace
metadata:
  name: rl-trading
  labels:
    name: rl-trading
"""
        (temp_project_dir / "k8s" / "namespace.yaml").write_text(namespace_yaml)
        
        # 验证YAML文件
        namespace_data = yaml.safe_load((temp_project_dir / "k8s" / "namespace.yaml").read_text())
        assert namespace_data["kind"] == "Namespace"
        assert namespace_data["metadata"]["name"] == "rl-trading"
    
    def test_kubernetes_configmap_creation(self, temp_project_dir):
        """测试Kubernetes ConfigMap创建"""
        # 创建ConfigMap YAML
        configmap_yaml = """
apiVersion: v1
kind: ConfigMap
metadata:
  name: trading-config
  namespace: rl-trading
data:
  system_config.yaml: |
    database:
      host: postgres
      port: 5432
    redis:
      host: redis
      port: 6379
"""
        (temp_project_dir / "k8s" / "configmap.yaml").write_text(configmap_yaml)
        
        # 验证ConfigMap
        configmap_data = yaml.safe_load((temp_project_dir / "k8s" / "configmap.yaml").read_text())
        assert configmap_data["kind"] == "ConfigMap"
        assert configmap_data["metadata"]["name"] == "trading-config"
        assert "system_config.yaml" in configmap_data["data"]
    
    def test_kubernetes_secret_creation(self, temp_project_dir):
        """测试Kubernetes Secret创建"""
        # 创建Secret YAML
        secret_yaml = """
apiVersion: v1
kind: Secret
metadata:
  name: trading-secrets
  namespace: rl-trading
type: Opaque
data:
  db_password: dHJhZGluZ19wYXNz  # base64 encoded
  api_key: YWJjZGVmZ2hpams=      # base64 encoded
"""
        (temp_project_dir / "k8s" / "secret.yaml").write_text(secret_yaml)
        
        # 验证Secret
        secret_data = yaml.safe_load((temp_project_dir / "k8s" / "secret.yaml").read_text())
        assert secret_data["kind"] == "Secret"
        assert secret_data["metadata"]["name"] == "trading-secrets"
        assert "db_password" in secret_data["data"]
    
    @patch('subprocess.run')
    def test_helm_chart_deployment(self, mock_subprocess, temp_project_dir):
        """测试Helm Chart部署"""
        # 创建Helm Chart结构
        chart_dir = temp_project_dir / "helm" / "rl-trading-system"
        chart_dir.mkdir(parents=True)
        
        # 创建Chart.yaml
        chart_yaml = """
apiVersion: v2
name: rl-trading-system
description: A Helm chart for RL Trading System
type: application
version: 0.1.0
appVersion: "1.0.0"
"""
        (chart_dir / "Chart.yaml").write_text(chart_yaml)
        
        # 创建values.yaml
        values_yaml = """
replicaCount: 3
image:
  repository: rl-trading-system
  tag: "v1.0.0"
  pullPolicy: IfNotPresent
service:
  type: ClusterIP
  port: 8000
"""
        (chart_dir / "values.yaml").write_text(values_yaml)
        
        # 模拟Helm部署命令
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Release deployed successfully"
        mock_subprocess.return_value = mock_result
        
        # 测试Helm安装
        result = subprocess.run(
            ["helm", "install", "rl-trading", str(chart_dir)],
            capture_output=True,
            text=True
        )
        
        # 验证命令执行
        mock_subprocess.assert_called_with(
            ["helm", "install", "rl-trading", str(chart_dir)],
            capture_output=True,
            text=True
        )
    
    def test_monitoring_stack_deployment(self, temp_project_dir):
        """测试监控栈部署"""
        # 创建Prometheus配置
        prometheus_config = """
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: 'rl-trading-system'
    static_configs:
      - targets: ['rl-trading-system:8000']
"""
        monitoring_dir = temp_project_dir / "monitoring"
        monitoring_dir.mkdir()
        (monitoring_dir / "prometheus.yml").write_text(prometheus_config)
        
        # 创建Grafana仪表板配置
        dashboard_config = """
{
  "dashboard": {
    "title": "RL Trading System",
    "panels": [
      {
        "title": "Trading Performance",
        "type": "graph"
      }
    ]
  }
}
"""
        grafana_dir = monitoring_dir / "grafana" / "dashboards"
        grafana_dir.mkdir(parents=True)
        (grafana_dir / "trading-dashboard.json").write_text(dashboard_config)
        
        # 验证监控配置文件
        assert (monitoring_dir / "prometheus.yml").exists()
        assert (grafana_dir / "trading-dashboard.json").exists()
    
    @patch('subprocess.run')
    def test_container_security_scanning(self, mock_subprocess, temp_project_dir):
        """测试容器安全扫描"""
        # 模拟安全扫描命令
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "No vulnerabilities found"
        mock_subprocess.return_value = mock_result
        
        # 测试Trivy扫描
        result = subprocess.run(
            ["trivy", "image", "rl-trading-system:v1.0.0"],
            capture_output=True,
            text=True
        )
        
        # 验证扫描命令
        mock_subprocess.assert_called_with(
            ["trivy", "image", "rl-trading-system:v1.0.0"],
            capture_output=True,
            text=True
        )
    
    def test_multi_environment_deployment(self, temp_project_dir):
        """测试多环境部署配置"""
        environments = ["dev", "staging", "prod"]
        
        for env in environments:
            env_dir = temp_project_dir / "k8s" / env
            env_dir.mkdir(parents=True)
            
            # 创建环境特定的配置
            env_config = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: trading-config-{env}
  namespace: rl-trading-{env}
data:
  environment: {env}
  replicas: {"1" if env == "dev" else "3" if env == "staging" else "5"}
"""
            (env_dir / "configmap.yaml").write_text(env_config)
            
            # 验证环境配置
            config_data = yaml.safe_load((env_dir / "configmap.yaml").read_text())
            assert config_data["metadata"]["name"] == f"trading-config-{env}"
            assert config_data["data"]["environment"] == env
    
    @patch('subprocess.run')
    def test_blue_green_deployment(self, mock_subprocess, temp_project_dir):
        """测试蓝绿部署"""
        # 创建蓝绿部署脚本
        deployment_script = """#!/bin/bash
# Blue-Green Deployment Script

NAMESPACE="rl-trading"
APP_NAME="rl-trading-system"
NEW_VERSION=$1

# Deploy green version
kubectl apply -f k8s/deployment-green.yaml
kubectl wait --for=condition=available --timeout=300s deployment/${APP_NAME}-green -n ${NAMESPACE}

# Switch traffic to green
kubectl patch service ${APP_NAME} -n ${NAMESPACE} -p '{"spec":{"selector":{"version":"green"}}}'

# Remove blue version
kubectl delete deployment ${APP_NAME}-blue -n ${NAMESPACE}
"""
        (temp_project_dir / "scripts" / "blue-green-deploy.sh").write_text(deployment_script)
        
        # 模拟部署脚本执行
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Blue-green deployment completed"
        mock_subprocess.return_value = mock_result
        
        # 测试部署脚本
        result = subprocess.run(
            ["bash", str(temp_project_dir / "scripts" / "blue-green-deploy.sh"), "v1.1.0"],
            capture_output=True,
            text=True
        )
        
        # 验证脚本执行
        mock_subprocess.assert_called_with(
            ["bash", str(temp_project_dir / "scripts" / "blue-green-deploy.sh"), "v1.1.0"],
            capture_output=True,
            text=True
        )
    
    def test_resource_limits_and_requests(self, temp_project_dir):
        """测试资源限制和请求配置"""
        # 创建带资源限制的部署配置
        deployment_with_resources = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rl-trading-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rl-trading-system
  template:
    metadata:
      labels:
        app: rl-trading-system
    spec:
      containers:
      - name: rl-trading-system
        image: rl-trading-system:v1.0.0
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
"""
        (temp_project_dir / "k8s" / "deployment-with-resources.yaml").write_text(deployment_with_resources)
        
        # 验证资源配置
        deployment_data = yaml.safe_load((temp_project_dir / "k8s" / "deployment-with-resources.yaml").read_text())
        container = deployment_data["spec"]["template"]["spec"]["containers"][0]
        
        assert container["resources"]["requests"]["memory"] == "1Gi"
        assert container["resources"]["requests"]["cpu"] == "500m"
        assert container["resources"]["limits"]["memory"] == "4Gi"
        assert container["resources"]["limits"]["cpu"] == "2000m"
        assert container["livenessProbe"]["httpGet"]["path"] == "/health"
        assert container["readinessProbe"]["httpGet"]["path"] == "/ready"


class TestContainerOrchestration:
    """容器编排测试"""
    
    def test_docker_swarm_stack_deployment(self):
        """测试Docker Swarm栈部署"""
        stack_config = """
version: '3.8'
services:
  rl-trading-system:
    image: rl-trading-system:v1.0.0
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
    ports:
      - "8000:8000"
    networks:
      - trading-network
    
  redis:
    image: redis:7-alpine
    deploy:
      replicas: 1
    networks:
      - trading-network

networks:
  trading-network:
    driver: overlay
"""
        
        # 解析栈配置
        stack_data = yaml.safe_load(stack_config)
        
        # 验证栈配置
        assert "services" in stack_data
        assert "rl-trading-system" in stack_data["services"]
        
        service = stack_data["services"]["rl-trading-system"]
        assert service["deploy"]["replicas"] == 3
        assert service["deploy"]["update_config"]["parallelism"] == 1
    
    def test_kubernetes_horizontal_pod_autoscaler(self):
        """测试Kubernetes水平Pod自动扩缩容"""
        hpa_config = """
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rl-trading-system-hpa
  namespace: rl-trading
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rl-trading-system
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""
        
        # 解析HPA配置
        hpa_data = yaml.safe_load(hpa_config)
        
        # 验证HPA配置
        assert hpa_data["kind"] == "HorizontalPodAutoscaler"
        assert hpa_data["spec"]["minReplicas"] == 3
        assert hpa_data["spec"]["maxReplicas"] == 10
        
        metrics = hpa_data["spec"]["metrics"]
        cpu_metric = next(m for m in metrics if m["resource"]["name"] == "cpu")
        assert cpu_metric["resource"]["target"]["averageUtilization"] == 70
    
    def test_kubernetes_network_policies(self):
        """测试Kubernetes网络策略"""
        network_policy = """
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: rl-trading-system-netpol
  namespace: rl-trading
spec:
  podSelector:
    matchLabels:
      app: rl-trading-system
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: web-dashboard
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
"""
        
        # 解析网络策略
        netpol_data = yaml.safe_load(network_policy)
        
        # 验证网络策略
        assert netpol_data["kind"] == "NetworkPolicy"
        assert "Ingress" in netpol_data["spec"]["policyTypes"]
        assert "Egress" in netpol_data["spec"]["policyTypes"]
        
        ingress_rules = netpol_data["spec"]["ingress"]
        assert len(ingress_rules) == 1
        assert ingress_rules[0]["ports"][0]["port"] == 8000
    
    def test_kubernetes_persistent_volume_claims(self):
        """测试Kubernetes持久卷声明"""
        pvc_config = """
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: trading-data-pvc
  namespace: rl-trading
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: trading-logs-pvc
  namespace: rl-trading
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
  storageClassName: standard
"""
        
        # 解析PVC配置
        pvc_docs = list(yaml.safe_load_all(pvc_config))
        
        # 验证PVC配置
        assert len(pvc_docs) == 2
        
        data_pvc = pvc_docs[0]
        assert data_pvc["metadata"]["name"] == "trading-data-pvc"
        assert data_pvc["spec"]["resources"]["requests"]["storage"] == "100Gi"
        assert data_pvc["spec"]["storageClassName"] == "fast-ssd"
        
        logs_pvc = pvc_docs[1]
        assert logs_pvc["metadata"]["name"] == "trading-logs-pvc"
        assert "ReadWriteMany" in logs_pvc["spec"]["accessModes"]