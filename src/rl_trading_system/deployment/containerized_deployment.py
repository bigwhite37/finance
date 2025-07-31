#!/usr/bin/env python3
"""
容器化部署实现

实现Docker容器构建和运行、Kubernetes部署和服务发现、CI/CD流水线和自动化部署
需求: 8.1, 8.4
"""

import os
import json
import yaml
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import docker
import kubernetes
import requests
import consul
from loguru import logger


class DeploymentStatus(Enum):
    """部署状态枚举"""
    PENDING = "pending"
    BUILDING = "building"
    DEPLOYING = "deploying"
    RUNNING = "running"
    FAILED = "failed"
    STOPPED = "stopped"


class ServiceStatus(Enum):
    """服务状态枚举"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ContainerConfig:
    """容器配置"""
    image_name: str
    image_tag: str
    dockerfile_path: str = "./Dockerfile"
    build_context: str = "."
    environment_vars: Dict[str, str] = field(default_factory=dict)
    ports: Dict[str, str] = field(default_factory=dict)
    volumes: Dict[str, str] = field(default_factory=dict)
    health_check_cmd: str = "curl -f http://localhost:8000/health || exit 1"
    health_check_interval: int = 30
    health_check_timeout: int = 30
    health_check_retries: int = 3
    restart_policy: str = "unless-stopped"


@dataclass
class KubernetesConfig:
    """Kubernetes配置"""
    namespace: str
    deployment_name: str
    service_name: str
    image_name: str
    replicas: int = 3
    cpu_request: str = "500m"
    cpu_limit: str = "2000m"
    memory_request: str = "1Gi"
    memory_limit: str = "4Gi"
    ports: List[int] = field(default_factory=lambda: [8000])
    environment_vars: Dict[str, str] = field(default_factory=dict)
    config_maps: List[str] = field(default_factory=list)
    secrets: List[str] = field(default_factory=list)
    health_check_path: str = "/health"
    readiness_check_path: str = "/ready"
    service_type: str = "ClusterIP"


class DockerManager:
    """Docker管理器"""
    
    def __init__(self, config: ContainerConfig):
        """初始化Docker管理器"""
        self.config = config
        self.client = docker.from_env()
        self.image_name = f"{config.image_name}:{config.image_tag}"
        
        logger.info(f"初始化Docker管理器: {self.image_name}")
    
    def build_image(self) -> bool:
        """构建Docker镜像"""
        try:
            logger.info(f"开始构建镜像: {self.image_name}")
            
            # 构建镜像
            image, build_logs = self.client.images.build(
                path=self.config.build_context,
                dockerfile=self.config.dockerfile_path,
                tag=self.image_name,
                rm=True,
                forcerm=True
            )
            
            logger.info(f"镜像构建成功: {image.id}")
            return True
            
        except Exception as e:
            logger.error(f"镜像构建失败: {e}")
            return False
    
    def run_container(self, container_name: str) -> Optional[str]:
        """运行容器"""
        try:
            logger.info(f"启动容器: {container_name}")
            
            # 准备端口映射
            ports = {}
            for container_port, host_port in self.config.ports.items():
                ports[container_port] = host_port
            
            # 准备卷映射
            volumes = {}
            for host_path, container_path in self.config.volumes.items():
                volumes[host_path] = {
                    'bind': container_path,
                    'mode': 'rw'
                }
            
            # 运行容器
            container = self.client.containers.run(
                image=self.image_name,
                name=container_name,
                ports=ports,
                volumes=volumes,
                environment=self.config.environment_vars,
                detach=True,
                restart_policy={"Name": self.config.restart_policy}
            )
            
            logger.info(f"容器启动成功: {container.id}")
            return container.id
            
        except Exception as e:
            logger.error(f"容器启动失败: {e}")
            return None
    
    def stop_container(self, container_id: str) -> bool:
        """停止容器"""
        try:
            logger.info(f"停止容器: {container_id}")
            
            container = self.client.containers.get(container_id)
            container.stop()
            container.remove()
            
            logger.info(f"容器停止成功: {container_id}")
            return True
            
        except Exception as e:
            logger.error(f"容器停止失败: {e}")
            return False
    
    def get_container_status(self, container_id: str) -> Dict[str, Any]:
        """获取容器状态"""
        try:
            container = self.client.containers.get(container_id)
            
            status = {
                "id": container.id,
                "name": container.name,
                "status": container.status,
                "created": container.attrs["Created"],
                "started": container.attrs["State"]["StartedAt"]
            }
            
            # 获取健康检查状态
            if "Health" in container.attrs["State"]:
                status["health"] = container.attrs["State"]["Health"]["Status"]
            else:
                status["health"] = "unknown"
            
            return status
            
        except Exception as e:
            logger.error(f"获取容器状态失败: {e}")
            return {"status": "unknown", "error": str(e)}
    
    def generate_dockerfile(self) -> str:
        """生成Dockerfile"""
        dockerfile_content = f"""# 强化学习量化交易系统 Docker 镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY src/ src/
COPY config/ config/
COPY scripts/ scripts/
COPY setup.py .
COPY README.md .

# 安装项目
RUN pip install -e .

# 创建必要的目录
RUN mkdir -p /app/logs /app/data /app/checkpoints /app/outputs

# 设置权限
RUN chmod +x scripts/*.py

# 暴露端口
EXPOSE {' '.join(self.config.ports.keys())}

# 健康检查
HEALTHCHECK --interval={self.config.health_check_interval}s \\
    --timeout={self.config.health_check_timeout}s \\
    --start-period=5s \\
    --retries={self.config.health_check_retries} \\
    CMD {self.config.health_check_cmd}

# 默认命令
CMD ["python", "scripts/monitor.py"]
"""
        return dockerfile_content
    
    def generate_docker_compose(self) -> str:
        """生成docker-compose.yml"""
        compose_data = {
            "version": "3.8",
            "services": {
                "rl-trading-system": {
                    "build": ".",
                    "container_name": f"{self.config.image_name}",
                    "ports": [f"{host}:{container}" for container, host in self.config.ports.items()],
                    "volumes": [f"{host}:{container}" for host, container in self.config.volumes.items()],
                    "environment": self.config.environment_vars,
                    "restart": self.config.restart_policy,
                    "depends_on": ["redis", "influxdb"],
                    "networks": ["trading-network"]
                },
                "redis": {
                    "image": "redis:7-alpine",
                    "container_name": "rl-trading-redis",
                    "ports": ["6379:6379"],
                    "volumes": ["redis_data:/data"],
                    "command": "redis-server --appendonly yes",
                    "networks": ["trading-network"],
                    "restart": "unless-stopped"
                },
                "influxdb": {
                    "image": "influxdb:2.7-alpine",
                    "container_name": "rl-trading-influxdb",
                    "ports": ["8086:8086"],
                    "volumes": ["influxdb_data:/var/lib/influxdb2"],
                    "environment": {
                        "DOCKER_INFLUXDB_INIT_MODE": "setup",
                        "DOCKER_INFLUXDB_INIT_USERNAME": "admin",
                        "DOCKER_INFLUXDB_INIT_PASSWORD": "password123",
                        "DOCKER_INFLUXDB_INIT_ORG": "rl-trading",
                        "DOCKER_INFLUXDB_INIT_BUCKET": "trading-data"
                    },
                    "networks": ["trading-network"],
                    "restart": "unless-stopped"
                }
            },
            "volumes": {
                "redis_data": None,
                "influxdb_data": None
            },
            "networks": {
                "trading-network": {
                    "driver": "bridge"
                }
            }
        }
        
        return yaml.dump(compose_data, default_flow_style=False)


class KubernetesManager:
    """Kubernetes管理器"""
    
    def __init__(self, config: KubernetesConfig):
        """初始化Kubernetes管理器"""
        self.config = config
        self.namespace = config.namespace
        
        # 加载Kubernetes配置
        try:
            kubernetes.config.load_incluster_config()
        except kubernetes.config.ConfigException:
            kubernetes.config.load_kube_config()
        
        # 初始化API客户端
        self.apps_v1 = kubernetes.client.AppsV1Api()
        self.core_v1 = kubernetes.client.CoreV1Api()
        
        logger.info(f"初始化Kubernetes管理器: {config.deployment_name}")
    
    def create_deployment(self) -> bool:
        """创建Kubernetes部署"""
        try:
            logger.info(f"创建部署: {self.config.deployment_name}")
            
            # 构建部署对象
            deployment = self._build_deployment_object()
            
            # 创建部署
            self.apps_v1.create_namespaced_deployment(
                namespace=self.namespace,
                body=deployment
            )
            
            logger.info(f"部署创建成功: {self.config.deployment_name}")
            return True
            
        except Exception as e:
            logger.error(f"部署创建失败: {e}")
            return False
    
    def create_service(self) -> bool:
        """创建Kubernetes服务"""
        try:
            logger.info(f"创建服务: {self.config.service_name}")
            
            # 构建服务对象
            service = self._build_service_object()
            
            # 创建服务
            self.core_v1.create_namespaced_service(
                namespace=self.namespace,
                body=service
            )
            
            logger.info(f"服务创建成功: {self.config.service_name}")
            return True
            
        except Exception as e:
            logger.error(f"服务创建失败: {e}")
            return False
    
    def update_deployment(self, new_image: str) -> bool:
        """更新部署"""
        try:
            logger.info(f"更新部署镜像: {new_image}")
            
            # 构建更新补丁
            patch = {
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [{
                                "name": self.config.deployment_name,
                                "image": new_image
                            }]
                        }
                    }
                }
            }
            
            # 应用更新
            self.apps_v1.patch_namespaced_deployment(
                name=self.config.deployment_name,
                namespace=self.namespace,
                body=patch
            )
            
            logger.info(f"部署更新成功: {self.config.deployment_name}")
            return True
            
        except Exception as e:
            logger.error(f"部署更新失败: {e}")
            return False
    
    def scale_deployment(self, replicas: int) -> bool:
        """扩缩容部署"""
        try:
            logger.info(f"扩缩容部署: {replicas} 副本")
            
            # 构建扩缩容补丁
            patch = {"spec": {"replicas": replicas}}
            
            # 应用扩缩容
            self.apps_v1.patch_namespaced_deployment_scale(
                name=self.config.deployment_name,
                namespace=self.namespace,
                body=patch
            )
            
            logger.info(f"扩缩容成功: {replicas} 副本")
            return True
            
        except Exception as e:
            logger.error(f"扩缩容失败: {e}")
            return False
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """获取部署状态"""
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=self.config.deployment_name,
                namespace=self.namespace
            )
            
            status = {
                "name": deployment.metadata.name,
                "namespace": deployment.metadata.namespace,
                "replicas": deployment.status.replicas or 0,
                "ready_replicas": deployment.status.ready_replicas or 0,
                "available_replicas": deployment.status.available_replicas or 0,
                "updated_replicas": deployment.status.updated_replicas or 0,
                "is_ready": False
            }
            
            # 检查是否就绪
            if (status["ready_replicas"] == status["replicas"] and 
                status["available_replicas"] == status["replicas"]):
                status["is_ready"] = True
            
            return status
            
        except Exception as e:
            logger.error(f"获取部署状态失败: {e}")
            return {"is_ready": False, "error": str(e)}
    
    def _build_deployment_object(self) -> kubernetes.client.V1Deployment:
        """构建部署对象"""
        # 容器配置
        container = kubernetes.client.V1Container(
            name=self.config.deployment_name,
            image=self.config.image_name,
            ports=[kubernetes.client.V1ContainerPort(container_port=port) for port in self.config.ports],
            env=[kubernetes.client.V1EnvVar(name=k, value=v) for k, v in self.config.environment_vars.items()],
            resources=kubernetes.client.V1ResourceRequirements(
                requests={
                    "cpu": self.config.cpu_request,
                    "memory": self.config.memory_request
                },
                limits={
                    "cpu": self.config.cpu_limit,
                    "memory": self.config.memory_limit
                }
            ),
            liveness_probe=kubernetes.client.V1Probe(
                http_get=kubernetes.client.V1HTTPGetAction(
                    path=self.config.health_check_path,
                    port=self.config.ports[0]
                ),
                initial_delay_seconds=30,
                period_seconds=10
            ),
            readiness_probe=kubernetes.client.V1Probe(
                http_get=kubernetes.client.V1HTTPGetAction(
                    path=self.config.readiness_check_path,
                    port=self.config.ports[0]
                ),
                initial_delay_seconds=5,
                period_seconds=5
            )
        )
        
        # Pod模板
        template = kubernetes.client.V1PodTemplateSpec(
            metadata=kubernetes.client.V1ObjectMeta(
                labels={"app": self.config.deployment_name}
            ),
            spec=kubernetes.client.V1PodSpec(containers=[container])
        )
        
        # 部署规格
        spec = kubernetes.client.V1DeploymentSpec(
            replicas=self.config.replicas,
            selector=kubernetes.client.V1LabelSelector(
                match_labels={"app": self.config.deployment_name}
            ),
            template=template
        )
        
        # 部署对象
        deployment = kubernetes.client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=kubernetes.client.V1ObjectMeta(
                name=self.config.deployment_name,
                namespace=self.namespace
            ),
            spec=spec
        )
        
        return deployment
    
    def _build_service_object(self) -> kubernetes.client.V1Service:
        """构建服务对象"""
        # 服务端口
        ports = [
            kubernetes.client.V1ServicePort(
                port=port,
                target_port=port,
                protocol="TCP"
            ) for port in self.config.ports
        ]
        
        # 服务规格
        spec = kubernetes.client.V1ServiceSpec(
            selector={"app": self.config.deployment_name},
            ports=ports,
            type=self.config.service_type
        )
        
        # 服务对象
        service = kubernetes.client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=kubernetes.client.V1ObjectMeta(
                name=self.config.service_name,
                namespace=self.namespace
            ),
            spec=spec
        )
        
        return service
    
    def generate_deployment_yaml(self) -> str:
        """生成部署YAML"""
        deployment_data = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": self.config.deployment_name,
                "namespace": self.config.namespace
            },
            "spec": {
                "replicas": self.config.replicas,
                "selector": {
                    "matchLabels": {
                        "app": self.config.deployment_name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": self.config.deployment_name
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": self.config.deployment_name,
                            "image": self.config.image_name,
                            "ports": [{"containerPort": port} for port in self.config.ports],
                            "env": [{"name": k, "value": v} for k, v in self.config.environment_vars.items()],
                            "resources": {
                                "requests": {
                                    "cpu": self.config.cpu_request,
                                    "memory": self.config.memory_request
                                },
                                "limits": {
                                    "cpu": self.config.cpu_limit,
                                    "memory": self.config.memory_limit
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": self.config.health_check_path,
                                    "port": self.config.ports[0]
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": self.config.readiness_check_path,
                                    "port": self.config.ports[0]
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }]
                    }
                }
            }
        }
        
        return yaml.dump(deployment_data, default_flow_style=False)
    
    def generate_service_yaml(self) -> str:
        """生成服务YAML"""
        service_data = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": self.config.service_name,
                "namespace": self.config.namespace
            },
            "spec": {
                "selector": {
                    "app": self.config.deployment_name
                },
                "ports": [
                    {
                        "port": port,
                        "targetPort": port,
                        "protocol": "TCP"
                    } for port in self.config.ports
                ],
                "type": self.config.service_type
            }
        }
        
        return yaml.dump(service_data, default_flow_style=False)


class CICDPipeline:
    """CI/CD流水线"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化CI/CD流水线"""
        self.config = config
        self.repository = config.get("repository", "")
        self.branch = config.get("branch", "main")
        self.build_stages = config.get("build_stages", ["test", "build", "deploy"])
        self.test_commands = config.get("test_commands", [])
        self.build_commands = config.get("build_commands", [])
        self.deploy_commands = config.get("deploy_commands", [])
        
        logger.info(f"初始化CI/CD流水线: {self.repository}")
    
    def run_tests(self) -> bool:
        """运行测试"""
        try:
            logger.info("开始运行测试")
            
            for command in self.test_commands:
                logger.info(f"执行测试命令: {command}")
                
                result = subprocess.run(
                    command.split(),
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30分钟超时
                )
                
                if result.returncode != 0:
                    logger.error(f"测试失败: {result.stderr}")
                    return False
                
                logger.info(f"测试通过: {result.stdout}")
            
            logger.info("所有测试通过")
            return True
            
        except Exception as e:
            logger.error(f"测试执行失败: {e}")
            return False
    
    def build_image(self) -> bool:
        """构建镜像"""
        try:
            logger.info("开始构建镜像")
            
            for command in self.build_commands:
                logger.info(f"执行构建命令: {command}")
                
                result = subprocess.run(
                    command.split(),
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30分钟超时
                )
                
                if result.returncode != 0:
                    logger.error(f"构建失败: {result.stderr}")
                    return False
                
                logger.info(f"构建成功: {result.stdout}")
            
            logger.info("镜像构建完成")
            return True
            
        except Exception as e:
            logger.error(f"镜像构建失败: {e}")
            return False
    
    def deploy_to_kubernetes(self) -> bool:
        """部署到Kubernetes"""
        try:
            logger.info("开始部署到Kubernetes")
            
            for command in self.deploy_commands:
                logger.info(f"执行部署命令: {command}")
                
                result = subprocess.run(
                    command.split(),
                    capture_output=True,
                    text=True,
                    timeout=600  # 10分钟超时
                )
                
                if result.returncode != 0:
                    logger.error(f"部署失败: {result.stderr}")
                    return False
                
                logger.info(f"部署成功: {result.stdout}")
            
            logger.info("Kubernetes部署完成")
            return True
            
        except Exception as e:
            logger.error(f"Kubernetes部署失败: {e}")
            return False
    
    def generate_github_actions_workflow(self) -> str:
        """生成GitHub Actions工作流"""
        workflow_data = {
            "name": "CI/CD Pipeline",
            "on": {
                "push": {
                    "branches": [self.branch]
                },
                "pull_request": {
                    "branches": [self.branch]
                }
            },
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v3"
                        },
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {
                                "python-version": "3.9"
                            }
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install -r requirements.txt"
                        },
                        {
                            "name": "Run tests",
                            "run": " && ".join(self.test_commands)
                        }
                    ]
                },
                "build": {
                    "needs": "test",
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v3"
                        },
                        {
                            "name": "Set up Docker Buildx",
                            "uses": "docker/setup-buildx-action@v2"
                        },
                        {
                            "name": "Login to Docker Hub",
                            "uses": "docker/login-action@v2",
                            "with": {
                                "username": "${{ secrets.DOCKER_USERNAME }}",
                                "password": "${{ secrets.DOCKER_PASSWORD }}"
                            }
                        },
                        {
                            "name": "Build and push",
                            "run": " && ".join(self.build_commands)
                        }
                    ]
                },
                "deploy": {
                    "needs": "build",
                    "runs-on": "ubuntu-latest",
                    "if": "github.ref == 'refs/heads/main'",
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v3"
                        },
                        {
                            "name": "Set up kubectl",
                            "uses": "azure/setup-kubectl@v3"
                        },
                        {
                            "name": "Deploy to Kubernetes",
                            "run": " && ".join(self.deploy_commands),
                            "env": {
                                "KUBE_CONFIG": "${{ secrets.KUBE_CONFIG }}"
                            }
                        }
                    ]
                }
            }
        }
        
        return yaml.dump(workflow_data, default_flow_style=False)
    
    def generate_jenkins_pipeline(self) -> str:
        """生成Jenkins流水线"""
        pipeline_script = f"""pipeline {{
    agent any
    
    environment {{
        DOCKER_REGISTRY = 'your-registry.com'
        IMAGE_NAME = 'rl-trading-system'
        KUBE_NAMESPACE = 'rl-trading'
    }}
    
    stages {{
        stage('Checkout') {{
            steps {{
                git branch: '{self.branch}', url: '{self.repository}'
            }}
        }}
        
        stage('Test') {{
            steps {{
                script {{
                    sh '''
                        {' && '.join(self.test_commands)}
                    '''
                }}
            }}
        }}
        
        stage('Build') {{
            steps {{
                script {{
                    sh '''
                        {' && '.join(self.build_commands)}
                    '''
                }}
            }}
        }}
        
        stage('Deploy') {{
            when {{
                branch '{self.branch}'
            }}
            steps {{
                script {{
                    sh '''
                        {' && '.join(self.deploy_commands)}
                    '''
                }}
            }}
        }}
    }}
    
    post {{
        always {{
            cleanWs()
        }}
        success {{
            echo 'Pipeline succeeded!'
        }}
        failure {{
            echo 'Pipeline failed!'
        }}
    }}
}}"""
        return pipeline_script


class HealthChecker:
    """健康检查器"""
    
    def __init__(self, endpoints: List[str], timeout: int = 30, 
                 retry_count: int = 3, retry_interval: int = 5):
        """初始化健康检查器"""
        self.endpoints = endpoints
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_interval = retry_interval
        
        logger.info(f"初始化健康检查器: {len(endpoints)} 个端点")
    
    def check_endpoint(self, endpoint: str) -> bool:
        """检查单个端点"""
        try:
            response = requests.get(endpoint, timeout=self.timeout)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"端点检查失败 {endpoint}: {e}")
            return False
    
    def check_all_endpoints(self) -> Dict[str, bool]:
        """检查所有端点"""
        results = {}
        
        for endpoint in self.endpoints:
            logger.info(f"检查端点: {endpoint}")
            results[endpoint] = self.check_endpoint(endpoint)
        
        return results
    
    def wait_for_healthy(self, endpoint: str, max_wait: int = 300) -> bool:
        """等待端点健康"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            if self.check_endpoint(endpoint):
                logger.info(f"端点健康: {endpoint}")
                return True
            
            logger.info(f"等待端点健康: {endpoint}")
            time.sleep(self.retry_interval)
        
        logger.error(f"端点健康检查超时: {endpoint}")
        return False


class ServiceDiscovery:
    """服务发现"""
    
    def __init__(self, consul_host: str = "localhost", consul_port: int = 8500,
                 service_name: str = "rl-trading-system", service_port: int = 8000,
                 health_check_url: str = "http://localhost:8000/health"):
        """初始化服务发现"""
        self.consul_host = consul_host
        self.consul_port = consul_port
        self.service_name = service_name
        self.service_port = service_port
        self.health_check_url = health_check_url
        
        # 初始化Consul客户端
        self.consul = consul.Consul(host=consul_host, port=consul_port)
        
        logger.info(f"初始化服务发现: {service_name}")
    
    def register_service(self) -> bool:
        """注册服务"""
        try:
            logger.info(f"注册服务: {self.service_name}")
            
            # 注册服务
            self.consul.agent.service.register(
                name=self.service_name,
                service_id=f"{self.service_name}-{os.getpid()}",
                port=self.service_port,
                check=consul.Check.http(self.health_check_url, interval="10s")
            )
            
            logger.info(f"服务注册成功: {self.service_name}")
            return True
            
        except Exception as e:
            logger.error(f"服务注册失败: {e}")
            return False
    
    def deregister_service(self) -> bool:
        """注销服务"""
        try:
            logger.info(f"注销服务: {self.service_name}")
            
            service_id = f"{self.service_name}-{os.getpid()}"
            self.consul.agent.service.deregister(service_id)
            
            logger.info(f"服务注销成功: {self.service_name}")
            return True
            
        except Exception as e:
            logger.error(f"服务注销失败: {e}")
            return False
    
    def discover_services(self, service_name: str) -> List[Dict[str, Any]]:
        """发现服务"""
        try:
            logger.info(f"发现服务: {service_name}")
            
            # 获取健康的服务实例
            _, services = self.consul.health.service(service_name, passing=True)
            
            service_list = []
            for service in services:
                service_info = {
                    "service_id": service["Service"]["ID"],
                    "service_name": service["Service"]["Service"],
                    "address": service["Service"]["Address"],
                    "port": service["Service"]["Port"],
                    "tags": service["Service"]["Tags"]
                }
                service_list.append(service_info)
            
            logger.info(f"发现 {len(service_list)} 个服务实例")
            return service_list
            
        except Exception as e:
            logger.error(f"服务发现失败: {e}")
            return []