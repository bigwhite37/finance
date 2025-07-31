#!/usr/bin/env python3
"""
容器化部署端到端测试

测试完整的容器化部署流程，包括Docker构建、Kubernetes部署、CI/CD流水线等
需求: 8.1, 8.4
"""

import pytest
import subprocess
import time
import requests
import yaml
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch

from src.rl_trading_system.deployment.containerized_deployment import (
    DockerManager,
    KubernetesManager,
    CICDPipeline,
    HealthChecker,
    ContainerConfig,
    KubernetesConfig
)


class TestDockerContainerE2E:
    """Docker容器端到端测试"""
    
    @pytest.fixture(scope="class")
    def docker_test_config(self):
        """Docker测试配置"""
        return ContainerConfig(
            image_name="rl-trading-test",
            image_tag="e2e-test",
            dockerfile_path="./Dockerfile",
            build_context=".",
            environment_vars={
                "PYTHONPATH": "/app/src",
                "LOG_LEVEL": "DEBUG",
                "TESTING": "true"
            },
            ports={"8000": "8001", "8888": "8889"},
            volumes={
                "./test_data": "/app/data",
                "./test_logs": "/app/logs"
            },
            health_check_cmd="curl -f http://localhost:8000/health || exit 1"
        )
    
    @pytest.fixture(scope="class")
    def docker_manager(self, docker_test_config):
        """Docker管理器实例"""
        return DockerManager(docker_test_config)
    
    def test_docker_image_build_and_run(self, docker_manager):
        """测试Docker镜像构建和运行完整流程"""
        # 跳过实际的Docker操作，使用模拟
        with patch('docker.from_env') as mock_docker:
            # 模拟Docker客户端
            mock_client = Mock()
            mock_docker.return_value = mock_client
            
            # 模拟镜像构建
            mock_image = Mock()
            mock_image.id = "sha256:test123"
            mock_client.images.build.return_value = (mock_image, [])
            
            # 测试构建
            build_result = docker_manager.build_image()
            assert build_result is True
            
            # 模拟容器运行
            mock_container = Mock()
            mock_container.id = "container_test_123"
            mock_container.status = "running"
            mock_client.containers.run.return_value = mock_container
            
            # 测试运行
            container_id = docker_manager.run_container("test-container")
            assert container_id == "container_test_123"
            
            # 模拟容器状态检查
            mock_container.attrs = {
                "Created": "2024-01-01T00:00:00Z",
                "State": {
                    "StartedAt": "2024-01-01T00:00:01Z",
                    "Health": {"Status": "healthy"}
                }
            }
            mock_client.containers.get.return_value = mock_container
            
            # 测试状态获取
            status = docker_manager.get_container_status(container_id)
            assert status["status"] == "running"
            assert status["health"] == "healthy"
            
            # 测试停止容器
            stop_result = docker_manager.stop_container(container_id)
            assert stop_result is True
    
    def test_docker_compose_stack_deployment(self, docker_manager):
        """测试Docker Compose栈部署"""
        # 生成docker-compose.yml
        compose_content = docker_manager.generate_docker_compose()
        
        # 验证compose内容
        compose_data = yaml.safe_load(compose_content)
        assert "services" in compose_data
        assert "rl-trading-test" in compose_data["services"]
        
        # 验证服务配置
        service = compose_data["services"]["rl-trading-test"]
        assert "ports" in service
        assert "volumes" in service
        assert "environment" in service
        assert "depends_on" in service
        
        # 验证依赖服务
        assert "redis" in compose_data["services"]
        assert "influxdb" in compose_data["services"]
    
    @patch('subprocess.run')
    def test_docker_health_check_integration(self, mock_subprocess, docker_manager):
        """测试Docker健康检查集成"""
        # 模拟健康检查命令执行
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "healthy"
        mock_subprocess.return_value = mock_result
        
        # 创建健康检查器
        health_checker = HealthChecker(
            endpoints=["http://localhost:8001/health"],
            timeout=10,
            retry_count=3
        )
        
        # 模拟健康检查
        with patch.object(health_checker, 'check_endpoint', return_value=True):
            result = health_checker.wait_for_healthy("http://localhost:8001/health", max_wait=30)
            assert result is True


class TestKubernetesDeploymentE2E:
    """Kubernetes部署端到端测试"""
    
    @pytest.fixture(scope="class")
    def k8s_test_config(self):
        """Kubernetes测试配置"""
        return KubernetesConfig(
            namespace="rl-trading-test",
            deployment_name="rl-trading-system-test",
            service_name="rl-trading-service-test",
            image_name="rl-trading-test:e2e-test",
            replicas=2,
            cpu_request="100m",
            cpu_limit="500m",
            memory_request="256Mi",
            memory_limit="1Gi",
            ports=[8000, 8888],
            environment_vars={
                "PYTHONPATH": "/app/src",
                "LOG_LEVEL": "DEBUG",
                "TESTING": "true"
            },
            health_check_path="/health",
            readiness_check_path="/ready"
        )
    
    @pytest.fixture(scope="class")
    def k8s_manager(self, k8s_test_config):
        """Kubernetes管理器实例"""
        return KubernetesManager(k8s_test_config)
    
    def test_kubernetes_deployment_yaml_generation(self, k8s_manager):
        """测试Kubernetes部署YAML生成"""
        deployment_yaml = k8s_manager.generate_deployment_yaml()
        deployment_data = yaml.safe_load(deployment_yaml)
        
        # 验证部署配置
        assert deployment_data["kind"] == "Deployment"
        assert deployment_data["metadata"]["name"] == "rl-trading-system-test"
        assert deployment_data["metadata"]["namespace"] == "rl-trading-test"
        
        # 验证副本数
        assert deployment_data["spec"]["replicas"] == 2
        
        # 验证容器配置
        container = deployment_data["spec"]["template"]["spec"]["containers"][0]
        assert container["name"] == "rl-trading-system-test"
        assert container["image"] == "rl-trading-test:e2e-test"
        
        # 验证资源限制
        resources = container["resources"]
        assert resources["requests"]["cpu"] == "100m"
        assert resources["requests"]["memory"] == "256Mi"
        assert resources["limits"]["cpu"] == "500m"
        assert resources["limits"]["memory"] == "1Gi"
        
        # 验证健康检查
        assert container["livenessProbe"]["httpGet"]["path"] == "/health"
        assert container["readinessProbe"]["httpGet"]["path"] == "/ready"
    
    def test_kubernetes_service_yaml_generation(self, k8s_manager):
        """测试Kubernetes服务YAML生成"""
        service_yaml = k8s_manager.generate_service_yaml()
        service_data = yaml.safe_load(service_yaml)
        
        # 验证服务配置
        assert service_data["kind"] == "Service"
        assert service_data["metadata"]["name"] == "rl-trading-service-test"
        assert service_data["metadata"]["namespace"] == "rl-trading-test"
        
        # 验证端口配置
        ports = service_data["spec"]["ports"]
        assert len(ports) == 2
        assert ports[0]["port"] == 8000
        assert ports[1]["port"] == 8888
        
        # 验证选择器
        assert service_data["spec"]["selector"]["app"] == "rl-trading-system-test"
    
    @patch('kubernetes.config.load_incluster_config')
    @patch('kubernetes.client.AppsV1Api')
    @patch('kubernetes.client.CoreV1Api')
    def test_kubernetes_full_deployment_flow(self, mock_core_api, mock_apps_api, mock_load_config, k8s_manager):
        """测试Kubernetes完整部署流程"""
        # 模拟API客户端
        mock_apps_client = Mock()
        mock_core_client = Mock()
        mock_apps_api.return_value = mock_apps_client
        mock_core_api.return_value = mock_core_client
        
        # 模拟部署创建
        mock_deployment = Mock()
        mock_deployment.metadata.name = "rl-trading-system-test"
        mock_apps_client.create_namespaced_deployment.return_value = mock_deployment
        
        # 测试创建部署
        deployment_result = k8s_manager.create_deployment()
        assert deployment_result is True
        mock_apps_client.create_namespaced_deployment.assert_called_once()
        
        # 模拟服务创建
        mock_service = Mock()
        mock_service.metadata.name = "rl-trading-service-test"
        mock_core_client.create_namespaced_service.return_value = mock_service
        
        # 测试创建服务
        service_result = k8s_manager.create_service()
        assert service_result is True
        mock_core_client.create_namespaced_service.assert_called_once()
        
        # 模拟部署状态检查
        mock_deployment_status = Mock()
        mock_deployment_status.status.replicas = 2
        mock_deployment_status.status.ready_replicas = 2
        mock_deployment_status.status.available_replicas = 2
        mock_apps_client.read_namespaced_deployment.return_value = mock_deployment_status
        
        # 测试状态获取
        status = k8s_manager.get_deployment_status()
        assert status["replicas"] == 2
        assert status["ready_replicas"] == 2
        assert status["is_ready"] is True
        
        # 测试扩缩容
        scale_result = k8s_manager.scale_deployment(3)
        assert scale_result is True
        mock_apps_client.patch_namespaced_deployment_scale.assert_called_once()
        
        # 测试更新部署
        update_result = k8s_manager.update_deployment("rl-trading-test:v2.0.0")
        assert update_result is True
        mock_apps_client.patch_namespaced_deployment.assert_called_once()


class TestCICDPipelineE2E:
    """CI/CD流水线端到端测试"""
    
    @pytest.fixture(scope="class")
    def pipeline_config(self):
        """CI/CD流水线配置"""
        return {
            "repository": "https://github.com/test/rl-trading-system.git",
            "branch": "main",
            "build_stages": ["test", "build", "deploy"],
            "test_commands": [
                "python -m pytest tests/unit/ -v",
                "python -m pytest tests/integration/ -v",
                "flake8 src/ --max-line-length=100",
                "mypy src/ --ignore-missing-imports"
            ],
            "build_commands": [
                "docker build -t rl-trading-system:${BUILD_NUMBER} .",
                "docker tag rl-trading-system:${BUILD_NUMBER} rl-trading-system:latest",
                "docker push rl-trading-system:${BUILD_NUMBER}",
                "docker push rl-trading-system:latest"
            ],
            "deploy_commands": [
                "kubectl set image deployment/rl-trading-system rl-trading-system=rl-trading-system:${BUILD_NUMBER}",
                "kubectl rollout status deployment/rl-trading-system",
                "kubectl get pods -l app=rl-trading-system"
            ]
        }
    
    @pytest.fixture(scope="class")
    def cicd_pipeline(self, pipeline_config):
        """CI/CD流水线实例"""
        return CICDPipeline(pipeline_config)
    
    @patch('subprocess.run')
    def test_full_cicd_pipeline_execution(self, mock_subprocess, cicd_pipeline):
        """测试完整CI/CD流水线执行"""
        # 模拟成功的命令执行
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Command executed successfully"
        mock_subprocess.return_value = mock_result
        
        # 测试运行测试阶段
        test_result = cicd_pipeline.run_tests()
        assert test_result is True
        assert mock_subprocess.call_count == 4  # 四个测试命令
        
        # 重置mock
        mock_subprocess.reset_mock()
        
        # 测试构建阶段
        build_result = cicd_pipeline.build_image()
        assert build_result is True
        assert mock_subprocess.call_count == 4  # 四个构建命令
        
        # 重置mock
        mock_subprocess.reset_mock()
        
        # 测试部署阶段
        deploy_result = cicd_pipeline.deploy_to_kubernetes()
        assert deploy_result is True
        assert mock_subprocess.call_count == 3  # 三个部署命令
    
    def test_github_actions_workflow_generation(self, cicd_pipeline):
        """测试GitHub Actions工作流生成"""
        workflow_yaml = cicd_pipeline.generate_github_actions_workflow()
        workflow_data = yaml.safe_load(workflow_yaml)
        
        # 验证工作流基本结构
        assert workflow_data["name"] == "CI/CD Pipeline"
        assert "on" in workflow_data
        assert "push" in workflow_data["on"]
        assert "pull_request" in workflow_data["on"]
        
        # 验证作业配置
        jobs = workflow_data["jobs"]
        assert "test" in jobs
        assert "build" in jobs
        assert "deploy" in jobs
        
        # 验证测试作业
        test_job = jobs["test"]
        assert test_job["runs-on"] == "ubuntu-latest"
        assert len(test_job["steps"]) >= 4  # 至少包含checkout, setup-python, install, test步骤
        
        # 验证构建作业
        build_job = jobs["build"]
        assert build_job["needs"] == "test"
        assert any("docker" in step.get("name", "").lower() for step in build_job["steps"])
        
        # 验证部署作业
        deploy_job = jobs["deploy"]
        assert deploy_job["needs"] == "build"
        assert deploy_job["if"] == "github.ref == 'refs/heads/main'"
    
    def test_jenkins_pipeline_generation(self, cicd_pipeline):
        """测试Jenkins流水线生成"""
        pipeline_script = cicd_pipeline.generate_jenkins_pipeline()
        
        # 验证流水线脚本结构
        assert "pipeline {" in pipeline_script
        assert "agent any" in pipeline_script
        
        # 验证阶段定义
        assert "stage('Checkout')" in pipeline_script
        assert "stage('Test')" in pipeline_script
        assert "stage('Build')" in pipeline_script
        assert "stage('Deploy')" in pipeline_script
        
        # 验证环境变量
        assert "DOCKER_REGISTRY" in pipeline_script
        assert "IMAGE_NAME" in pipeline_script
        assert "KUBE_NAMESPACE" in pipeline_script
        
        # 验证后处理
        assert "post {" in pipeline_script
        assert "always {" in pipeline_script
        assert "success {" in pipeline_script
        assert "failure {" in pipeline_script


class TestContainerizedDeploymentIntegration:
    """容器化部署集成测试"""
    
    @pytest.fixture(scope="class")
    def temp_project_structure(self):
        """创建临时项目结构"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)
            
            # 创建项目目录结构
            directories = [
                "src/rl_trading_system",
                "tests/unit",
                "tests/integration",
                "tests/e2e",
                "config",
                "scripts",
                "k8s/base",
                "k8s/overlays/dev",
                "k8s/overlays/staging",
                "k8s/overlays/prod",
                ".github/workflows",
                "helm/rl-trading-system/templates",
                "monitoring/prometheus",
                "monitoring/grafana/dashboards"
            ]
            
            for directory in directories:
                (project_dir / directory).mkdir(parents=True)
            
            # 创建基本文件
            files_content = {
                "requirements.txt": "torch>=1.12.0\nnumpy>=1.21.0\nkubernetes>=24.0.0\ndocker>=6.0.0",
                "setup.py": "from setuptools import setup, find_packages\nsetup(name='rl-trading-system', packages=find_packages())",
                "Dockerfile": """FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "-m", "rl_trading_system.main"]""",
                "docker-compose.yml": """version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app""",
                "config/system_config.yaml": """database:
  host: postgres
  port: 5432
redis:
  host: redis
  port: 6379""",
                ".github/workflows/ci-cd.yml": """name: CI/CD
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Test
        run: echo "Testing" """,
                "k8s/base/kustomization.yaml": """apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - deployment.yaml
  - service.yaml""",
                "helm/rl-trading-system/Chart.yaml": """apiVersion: v2
name: rl-trading-system
version: 0.1.0
appVersion: 1.0.0""",
                "monitoring/prometheus/prometheus.yml": """global:
  scrape_interval: 15s
scrape_configs:
  - job_name: 'rl-trading-system'
    static_configs:
      - targets: ['localhost:8000']"""
            }
            
            for file_path, content in files_content.items():
                (project_dir / file_path).write_text(content)
            
            yield project_dir
    
    def test_complete_project_structure_validation(self, temp_project_structure):
        """测试完整项目结构验证"""
        project_dir = temp_project_structure
        
        # 验证核心文件存在
        core_files = [
            "Dockerfile",
            "docker-compose.yml",
            "requirements.txt",
            "setup.py",
            "config/system_config.yaml"
        ]
        
        for file_path in core_files:
            assert (project_dir / file_path).exists(), f"Missing core file: {file_path}"
        
        # 验证Kubernetes配置目录
        k8s_dirs = [
            "k8s/base",
            "k8s/overlays/dev",
            "k8s/overlays/staging",
            "k8s/overlays/prod"
        ]
        
        for k8s_dir in k8s_dirs:
            assert (project_dir / k8s_dir).exists(), f"Missing k8s directory: {k8s_dir}"
        
        # 验证CI/CD配置
        assert (project_dir / ".github/workflows/ci-cd.yml").exists()
        
        # 验证Helm Chart
        assert (project_dir / "helm/rl-trading-system/Chart.yaml").exists()
        
        # 验证监控配置
        assert (project_dir / "monitoring/prometheus/prometheus.yml").exists()
    
    def test_dockerfile_best_practices_validation(self, temp_project_structure):
        """测试Dockerfile最佳实践验证"""
        dockerfile_path = temp_project_structure / "Dockerfile"
        dockerfile_content = dockerfile_path.read_text()
        
        # 验证Dockerfile最佳实践
        best_practices = [
            "FROM python:3.9-slim",  # 使用官方基础镜像
            "WORKDIR /app",          # 设置工作目录
            "COPY requirements.txt", # 先复制依赖文件
            "RUN pip install",       # 安装依赖
            "COPY . .",             # 复制应用代码
            "EXPOSE",               # 暴露端口
            "CMD"                   # 设置默认命令
        ]
        
        for practice in best_practices:
            assert practice in dockerfile_content, f"Dockerfile missing best practice: {practice}"
    
    @patch('subprocess.run')
    def test_multi_stage_deployment_simulation(self, mock_subprocess, temp_project_structure):
        """测试多阶段部署模拟"""
        project_dir = temp_project_structure
        
        # 模拟成功的命令执行
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Success"
        mock_subprocess.return_value = mock_result
        
        # 模拟部署流程
        deployment_stages = [
            # 构建阶段
            ["docker", "build", "-t", "rl-trading-system:test", "."],
            # 测试阶段
            ["docker", "run", "--rm", "rl-trading-system:test", "python", "-m", "pytest"],
            # 推送阶段
            ["docker", "push", "rl-trading-system:test"],
            # 部署阶段
            ["kubectl", "apply", "-f", "k8s/base/"],
            # 验证阶段
            ["kubectl", "rollout", "status", "deployment/rl-trading-system"]
        ]
        
        # 执行每个阶段
        for stage_cmd in deployment_stages:
            result = subprocess.run(stage_cmd, cwd=project_dir, capture_output=True, text=True)
            # 验证命令被调用
            mock_subprocess.assert_called_with(stage_cmd, cwd=project_dir, capture_output=True, text=True)
        
        # 验证所有阶段都被执行
        assert mock_subprocess.call_count == len(deployment_stages)
    
    def test_environment_specific_configurations(self, temp_project_structure):
        """测试环境特定配置"""
        project_dir = temp_project_structure
        environments = ["dev", "staging", "prod"]
        
        for env in environments:
            env_dir = project_dir / "k8s" / "overlays" / env
            
            # 创建环境特定的kustomization.yaml
            kustomization_content = f"""apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: rl-trading-{env}
resources:
  - ../../base
patchesStrategicMerge:
  - deployment-patch.yaml
images:
  - name: rl-trading-system
    newTag: {env}-latest
replicas:
  - name: rl-trading-system
    count: {1 if env == 'dev' else 3 if env == 'staging' else 5}
"""
            (env_dir / "kustomization.yaml").write_text(kustomization_content)
            
            # 创建环境特定的部署补丁
            deployment_patch = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: rl-trading-system
spec:
  template:
    spec:
      containers:
      - name: rl-trading-system
        env:
        - name: ENVIRONMENT
          value: {env}
        - name: LOG_LEVEL
          value: {"DEBUG" if env == "dev" else "INFO"}
        resources:
          requests:
            cpu: {"100m" if env == "dev" else "500m" if env == "staging" else "1000m"}
            memory: {"256Mi" if env == "dev" else "1Gi" if env == "staging" else "2Gi"}
          limits:
            cpu: {"500m" if env == "dev" else "2000m" if env == "staging" else "4000m"}
            memory: {"1Gi" if env == "dev" else "4Gi" if env == "staging" else "8Gi"}
"""
            (env_dir / "deployment-patch.yaml").write_text(deployment_patch)
            
            # 验证配置文件创建成功
            assert (env_dir / "kustomization.yaml").exists()
            assert (env_dir / "deployment-patch.yaml").exists()
            
            # 验证配置内容
            kustomization_data = yaml.safe_load((env_dir / "kustomization.yaml").read_text())
            assert kustomization_data["namespace"] == f"rl-trading-{env}"
            
            deployment_data = yaml.safe_load((env_dir / "deployment-patch.yaml").read_text())
            env_vars = deployment_data["spec"]["template"]["spec"]["containers"][0]["env"]
            env_var_dict = {var["name"]: var["value"] for var in env_vars}
            assert env_var_dict["ENVIRONMENT"] == env
    
    @patch('requests.get')
    def test_health_check_endpoints_validation(self, mock_get, temp_project_structure):
        """测试健康检查端点验证"""
        # 模拟健康检查响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",
            "version": "1.0.0",
            "components": {
                "database": "healthy",
                "redis": "healthy",
                "model": "healthy"
            }
        }
        mock_get.return_value = mock_response
        
        # 创建健康检查器
        health_checker = HealthChecker(
            endpoints=[
                "http://localhost:8000/health",
                "http://localhost:8000/ready",
                "http://localhost:8000/metrics"
            ],
            timeout=10,
            retry_count=3
        )
        
        # 测试所有端点
        results = health_checker.check_all_endpoints()
        
        # 验证结果
        assert len(results) == 3
        for endpoint, result in results.items():
            assert result is True, f"Health check failed for {endpoint}"
        
        # 验证请求被正确调用
        assert mock_get.call_count == 3
    
    def test_security_configurations_validation(self, temp_project_structure):
        """测试安全配置验证"""
        project_dir = temp_project_structure
        
        # 创建安全相关的Kubernetes配置
        security_configs = {
            "network-policy.yaml": """apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: rl-trading-system-netpol
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
      port: 8000""",
            
            "pod-security-policy.yaml": """apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: rl-trading-system-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'""",
            
            "rbac.yaml": """apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: rl-trading-system-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: rl-trading-system-rolebinding
subjects:
- kind: ServiceAccount
  name: rl-trading-system
roleRef:
  kind: Role
  name: rl-trading-system-role
  apiGroup: rbac.authorization.k8s.io"""
        }
        
        # 创建安全配置文件
        security_dir = project_dir / "k8s" / "security"
        security_dir.mkdir()
        
        for filename, content in security_configs.items():
            (security_dir / filename).write_text(content)
            
            # 验证YAML格式正确
            config_data = yaml.safe_load(content)
            assert "apiVersion" in config_data
            assert "kind" in config_data
            assert "metadata" in config_data
        
        # 验证所有安全配置文件存在
        for filename in security_configs.keys():
            assert (security_dir / filename).exists(), f"Missing security config: {filename}"