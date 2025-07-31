#!/bin/bash

# 容器化部署脚本
# 用于部署RL Trading System到Kubernetes集群

set -euo pipefail

# 默认配置
DEFAULT_NAMESPACE="rl-trading"
DEFAULT_ENVIRONMENT="dev"
DEFAULT_IMAGE_TAG="latest"
DEFAULT_REGISTRY="ghcr.io/rl-trading"
DEFAULT_IMAGE_NAME="rl-trading-system"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
容器化部署脚本

用法: $0 [选项]

选项:
    -e, --environment ENV       部署环境 (dev|staging|prod) [默认: ${DEFAULT_ENVIRONMENT}]
    -n, --namespace NAMESPACE   Kubernetes命名空间 [默认: ${DEFAULT_NAMESPACE}]
    -t, --tag TAG              Docker镜像标签 [默认: ${DEFAULT_IMAGE_TAG}]
    -r, --registry REGISTRY    Docker镜像仓库 [默认: ${DEFAULT_REGISTRY}]
    -i, --image IMAGE          Docker镜像名称 [默认: ${DEFAULT_IMAGE_NAME}]
    --dry-run                  只显示将要执行的命令，不实际执行
    --skip-build               跳过Docker镜像构建
    --skip-push                跳过Docker镜像推送
    --skip-deploy              跳过Kubernetes部署
    --force                    强制部署，忽略健康检查失败
    --rollback                 回滚到上一个版本
    --cleanup                  清理旧的资源
    -h, --help                 显示此帮助信息

示例:
    $0 -e dev -t v1.0.0                    # 部署v1.0.0到开发环境
    $0 -e prod -t v2.0.0 --force          # 强制部署v2.0.0到生产环境
    $0 --rollback -e staging               # 回滚staging环境
    $0 --cleanup -e dev                    # 清理开发环境资源
    $0 --dry-run -e prod -t latest         # 预览生产环境部署命令

EOF
}

# 解析命令行参数
parse_args() {
    ENVIRONMENT="${DEFAULT_ENVIRONMENT}"
    NAMESPACE="${DEFAULT_NAMESPACE}"
    IMAGE_TAG="${DEFAULT_IMAGE_TAG}"
    REGISTRY="${DEFAULT_REGISTRY}"
    IMAGE_NAME="${DEFAULT_IMAGE_NAME}"
    DRY_RUN=false
    SKIP_BUILD=false
    SKIP_PUSH=false
    SKIP_DEPLOY=false
    FORCE=false
    ROLLBACK=false
    CLEANUP=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -t|--tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            -r|--registry)
                REGISTRY="$2"
                shift 2
                ;;
            -i|--image)
                IMAGE_NAME="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --skip-push)
                SKIP_PUSH=true
                shift
                ;;
            --skip-deploy)
                SKIP_DEPLOY=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --rollback)
                ROLLBACK=true
                shift
                ;;
            --cleanup)
                CLEANUP=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # 根据环境设置命名空间
    case $ENVIRONMENT in
        dev)
            NAMESPACE="rl-trading-dev"
            ;;
        staging)
            NAMESPACE="rl-trading-staging"
            ;;
        prod)
            NAMESPACE="rl-trading"
            ;;
        *)
            log_error "不支持的环境: $ENVIRONMENT"
            exit 1
            ;;
    esac

    FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
}

# 执行命令（支持dry-run模式）
execute_command() {
    local cmd="$1"
    local description="$2"
    
    log_info "$description"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] $cmd"
    else
        echo "  执行: $cmd"
        if ! eval "$cmd"; then
            log_error "$description 失败"
            return 1
        fi
    fi
}

# 检查必要的工具
check_prerequisites() {
    log_info "检查必要的工具..."
    
    local tools=("docker" "kubectl" "helm")
    local missing_tools=()
    
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "缺少必要的工具: ${missing_tools[*]}"
        log_error "请安装这些工具后重试"
        exit 1
    fi
    
    # 检查kubectl连接
    if ! kubectl cluster-info &> /dev/null; then
        log_error "无法连接到Kubernetes集群"
        log_error "请检查kubectl配置"
        exit 1
    fi
    
    log_success "所有必要工具已就绪"
}

# 构建Docker镜像
build_image() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        log_info "跳过Docker镜像构建"
        return 0
    fi
    
    log_info "构建Docker镜像: $FULL_IMAGE_NAME"
    
    local build_args=(
        "--build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
        "--build-arg VCS_REF=$(git rev-parse --short HEAD)"
        "--build-arg VERSION=${IMAGE_TAG}"
        "--build-arg ENVIRONMENT=${ENVIRONMENT}"
    )
    
    local cmd="docker build ${build_args[*]} -t $FULL_IMAGE_NAME ."
    execute_command "$cmd" "构建Docker镜像"
}

# 推送Docker镜像
push_image() {
    if [[ "$SKIP_PUSH" == "true" ]]; then
        log_info "跳过Docker镜像推送"
        return 0
    fi
    
    log_info "推送Docker镜像: $FULL_IMAGE_NAME"
    
    execute_command "docker push $FULL_IMAGE_NAME" "推送Docker镜像"
}

# 创建命名空间
create_namespace() {
    log_info "创建命名空间: $NAMESPACE"
    
    execute_command "kubectl apply -f k8s/namespace.yaml" "创建命名空间"
}

# 部署配置
deploy_configs() {
    log_info "部署配置文件..."
    
    local config_files=(
        "k8s/configmap.yaml"
        "k8s/secret.yaml"
        "k8s/rbac.yaml"
        "k8s/pvc.yaml"
    )
    
    for file in "${config_files[@]}"; do
        if [[ -f "$file" ]]; then
            execute_command "kubectl apply -f $file" "部署 $file"
        else
            log_warning "配置文件不存在: $file"
        fi
    done
}

# 部署应用
deploy_application() {
    log_info "部署应用到 $NAMESPACE 命名空间..."
    
    # 部署主要资源
    local resource_files=(
        "k8s/deployment.yaml"
        "k8s/service.yaml"
        "k8s/ingress.yaml"
        "k8s/hpa.yaml"
        "k8s/network-policy.yaml"
    )
    
    for file in "${resource_files[@]}"; do
        if [[ -f "$file" ]]; then
            execute_command "kubectl apply -f $file" "部署 $file"
        else
            log_warning "资源文件不存在: $file"
        fi
    done
    
    # 更新镜像
    local deployment_name="rl-trading-system"
    if [[ "$ENVIRONMENT" == "dev" ]]; then
        deployment_name="rl-trading-system-dev"
    fi
    
    execute_command "kubectl set image deployment/$deployment_name rl-trading-system=$FULL_IMAGE_NAME -n $NAMESPACE" "更新部署镜像"
}

# 等待部署完成
wait_for_deployment() {
    local deployment_name="rl-trading-system"
    if [[ "$ENVIRONMENT" == "dev" ]]; then
        deployment_name="rl-trading-system-dev"
    fi
    
    log_info "等待部署完成..."
    
    execute_command "kubectl rollout status deployment/$deployment_name -n $NAMESPACE --timeout=600s" "等待部署完成"
}

# 健康检查
health_check() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] 跳过健康检查"
        return 0
    fi
    
    log_info "执行健康检查..."
    
    # 等待Pod就绪
    sleep 30
    
    # 检查Pod状态
    local pods=$(kubectl get pods -l app=rl-trading-system -n $NAMESPACE -o jsonpath='{.items[*].metadata.name}')
    
    if [[ -z "$pods" ]]; then
        log_error "没有找到运行中的Pod"
        return 1
    fi
    
    for pod in $pods; do
        local status=$(kubectl get pod $pod -n $NAMESPACE -o jsonpath='{.status.phase}')
        if [[ "$status" != "Running" ]]; then
            log_error "Pod $pod 状态异常: $status"
            if [[ "$FORCE" != "true" ]]; then
                return 1
            fi
        else
            log_success "Pod $pod 运行正常"
        fi
    done
    
    # HTTP健康检查
    local service_url=""
    case $ENVIRONMENT in
        dev)
            service_url="https://trading-dev.rlsystem.com"
            ;;
        staging)
            service_url="https://trading-staging.rlsystem.com"
            ;;
        prod)
            service_url="https://trading.rlsystem.com"
            ;;
    esac
    
    if [[ -n "$service_url" ]]; then
        log_info "检查服务健康状态: $service_url"
        
        local max_retries=10
        local retry_count=0
        
        while [[ $retry_count -lt $max_retries ]]; do
            if curl -f -s "$service_url/api/health" > /dev/null; then
                log_success "服务健康检查通过"
                break
            else
                retry_count=$((retry_count + 1))
                log_warning "健康检查失败，重试 $retry_count/$max_retries"
                sleep 10
            fi
        done
        
        if [[ $retry_count -eq $max_retries ]]; then
            log_error "服务健康检查失败"
            if [[ "$FORCE" != "true" ]]; then
                return 1
            fi
        fi
    fi
}

# 回滚部署
rollback_deployment() {
    local deployment_name="rl-trading-system"
    if [[ "$ENVIRONMENT" == "dev" ]]; then
        deployment_name="rl-trading-system-dev"
    fi
    
    log_info "回滚部署: $deployment_name"
    
    execute_command "kubectl rollout undo deployment/$deployment_name -n $NAMESPACE" "回滚部署"
    execute_command "kubectl rollout status deployment/$deployment_name -n $NAMESPACE --timeout=300s" "等待回滚完成"
    
    log_success "回滚完成"
}

# 清理资源
cleanup_resources() {
    log_info "清理旧资源..."
    
    # 清理旧的ReplicaSets
    execute_command "kubectl delete replicaset -l app=rl-trading-system -n $NAMESPACE --cascade=orphan" "清理旧的ReplicaSets"
    
    # 清理未使用的ConfigMaps和Secrets（谨慎操作）
    if [[ "$ENVIRONMENT" == "dev" ]]; then
        log_info "清理开发环境的临时资源..."
        execute_command "kubectl delete configmap -l environment=development,temporary=true -n $NAMESPACE" "清理临时ConfigMaps"
    fi
    
    # 清理Docker镜像
    if command -v docker &> /dev/null; then
        execute_command "docker image prune -f" "清理未使用的Docker镜像"
    fi
    
    log_success "资源清理完成"
}

# 显示部署状态
show_status() {
    log_info "部署状态:"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] 跳过状态显示"
        return 0
    fi
    
    echo "  环境: $ENVIRONMENT"
    echo "  命名空间: $NAMESPACE"
    echo "  镜像: $FULL_IMAGE_NAME"
    echo ""
    
    echo "  Pods:"
    kubectl get pods -l app=rl-trading-system -n $NAMESPACE -o wide || true
    echo ""
    
    echo "  Services:"
    kubectl get services -l app=rl-trading-system -n $NAMESPACE || true
    echo ""
    
    echo "  Ingress:"
    kubectl get ingress -l app=rl-trading-system -n $NAMESPACE || true
    echo ""
    
    echo "  HPA:"
    kubectl get hpa -l app=rl-trading-system -n $NAMESPACE || true
}

# 主函数
main() {
    log_info "开始容器化部署流程..."
    
    parse_args "$@"
    
    # 显示配置信息
    echo "部署配置:"
    echo "  环境: $ENVIRONMENT"
    echo "  命名空间: $NAMESPACE"
    echo "  镜像: $FULL_IMAGE_NAME"
    echo "  Dry Run: $DRY_RUN"
    echo ""
    
    # 检查前置条件
    check_prerequisites
    
    # 执行相应操作
    if [[ "$CLEANUP" == "true" ]]; then
        cleanup_resources
        exit 0
    fi
    
    if [[ "$ROLLBACK" == "true" ]]; then
        rollback_deployment
        show_status
        exit 0
    fi
    
    # 正常部署流程
    build_image
    push_image
    
    if [[ "$SKIP_DEPLOY" != "true" ]]; then
        create_namespace
        deploy_configs
        deploy_application
        wait_for_deployment
        health_check
    fi
    
    show_status
    
    log_success "部署完成!"
    
    # 显示访问信息
    case $ENVIRONMENT in
        dev)
            echo ""
            echo "访问地址:"
            echo "  Web界面: https://trading-dev.rlsystem.com"
            echo "  API: https://api-dev.trading.rlsystem.com"
            echo "  监控: https://metrics-dev.trading.rlsystem.com"
            echo "  TensorBoard: https://tensorboard-dev.trading.rlsystem.com"
            ;;
        staging)
            echo ""
            echo "访问地址:"
            echo "  Web界面: https://trading-staging.rlsystem.com"
            echo "  API: https://api-staging.trading.rlsystem.com"
            echo "  监控: https://metrics-staging.trading.rlsystem.com"
            echo "  TensorBoard: https://tensorboard-staging.trading.rlsystem.com"
            ;;
        prod)
            echo ""
            echo "访问地址:"
            echo "  Web界面: https://trading.rlsystem.com"
            echo "  API: https://api.trading.rlsystem.com"
            echo "  监控: https://metrics.trading.rlsystem.com"
            echo "  TensorBoard: https://tensorboard.trading.rlsystem.com"
            ;;
    esac
}

# 错误处理
trap 'log_error "部署过程中发生错误，退出码: $?"' ERR

# 执行主函数
main "$@"