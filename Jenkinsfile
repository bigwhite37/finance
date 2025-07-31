pipeline {
    agent any
    
    environment {
        DOCKER_REGISTRY = 'registry.rlsystem.com'
        IMAGE_NAME = 'rl-trading-system'
        KUBE_NAMESPACE = 'rl-trading'
        PYTHON_VERSION = '3.9'
        
        // Credentials
        DOCKER_CREDENTIALS = credentials('docker-registry-credentials')
        KUBE_CONFIG = credentials('kubernetes-config')
        SLACK_WEBHOOK = credentials('slack-webhook-url')
        EMAIL_CREDENTIALS = credentials('email-credentials')
        SNYK_TOKEN = credentials('snyk-token')
    }
    
    parameters {
        choice(
            name: 'ENVIRONMENT',
            choices: ['dev', 'staging', 'prod'],
            description: 'Target deployment environment'
        )
        booleanParam(
            name: 'SKIP_TESTS',
            defaultValue: false,
            description: 'Skip test execution'
        )
        booleanParam(
            name: 'FORCE_DEPLOY',
            defaultValue: false,
            description: 'Force deployment even if tests fail'
        )
        string(
            name: 'IMAGE_TAG',
            defaultValue: '',
            description: 'Custom image tag (leave empty for auto-generated)'
        )
    }
    
    options {
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timeout(time: 60, unit: 'MINUTES')
        timestamps()
        ansiColor('xterm')
        skipDefaultCheckout(false)
    }
    
    triggers {
        // Poll SCM every 5 minutes
        pollSCM('H/5 * * * *')
        
        // Trigger on upstream job completion
        upstream(upstreamProjects: 'data-pipeline', threshold: hudson.model.Result.SUCCESS)
    }
    
    stages {
        stage('Checkout') {
            steps {
                script {
                    // Clean workspace
                    cleanWs()
                    
                    // Checkout code
                    checkout scm
                    
                    // Set build information
                    env.BUILD_TIMESTAMP = sh(
                        script: 'date +"%Y%m%d-%H%M%S"',
                        returnStdout: true
                    ).trim()
                    
                    env.GIT_COMMIT_SHORT = sh(
                        script: 'git rev-parse --short HEAD',
                        returnStdout: true
                    ).trim()
                    
                    env.IMAGE_TAG_FINAL = params.IMAGE_TAG ?: "${env.BRANCH_NAME}-${env.BUILD_TIMESTAMP}-${env.GIT_COMMIT_SHORT}"
                    
                    echo "Build Information:"
                    echo "  Branch: ${env.BRANCH_NAME}"
                    echo "  Commit: ${env.GIT_COMMIT_SHORT}"
                    echo "  Image Tag: ${env.IMAGE_TAG_FINAL}"
                    echo "  Environment: ${params.ENVIRONMENT}"
                }
            }
        }
        
        stage('Setup Environment') {
            steps {
                script {
                    // Setup Python virtual environment
                    sh '''
                        python${PYTHON_VERSION} -m venv venv
                        . venv/bin/activate
                        pip install --upgrade pip
                        pip install -r requirements.txt
                        pip install pytest pytest-cov flake8 mypy black isort bandit safety
                    '''
                }
            }
        }
        
        stage('Code Quality') {
            parallel {
                stage('Linting') {
                    steps {
                        script {
                            sh '''
                                . venv/bin/activate
                                
                                echo "Running Black formatter check..."
                                black --check --diff src/ tests/ || exit 1
                                
                                echo "Running isort import sorting check..."
                                isort --check-only --diff src/ tests/ || exit 1
                                
                                echo "Running flake8 linting..."
                                flake8 src/ tests/ --max-line-length=100 --exclude=__pycache__ --output-file=flake8-report.txt || exit 1
                                
                                echo "Running mypy type checking..."
                                mypy src/ --ignore-missing-imports --strict --junit-xml=mypy-report.xml || exit 1
                            '''
                        }
                    }
                    post {
                        always {
                            archiveArtifacts artifacts: 'flake8-report.txt,mypy-report.xml', allowEmptyArchive: true
                        }
                    }
                }
                
                stage('Security Scan') {
                    steps {
                        script {
                            sh '''
                                . venv/bin/activate
                                
                                echo "Running Bandit security scan..."
                                bandit -r src/ -f json -o bandit-report.json || true
                                
                                echo "Running Safety dependency check..."
                                safety check --json --output safety-report.json || true
                            '''
                            
                            // Run Snyk scan if token is available
                            if (env.SNYK_TOKEN) {
                                sh '''
                                    echo "Running Snyk vulnerability scan..."
                                    snyk test --severity-threshold=high --json > snyk-report.json || true
                                '''
                            }
                        }
                    }
                    post {
                        always {
                            archiveArtifacts artifacts: '*-report.json', allowEmptyArchive: true
                        }
                    }
                }
            }
        }
        
        stage('Tests') {
            when {
                not { params.SKIP_TESTS }
            }
            parallel {
                stage('Unit Tests') {
                    steps {
                        script {
                            sh '''
                                . venv/bin/activate
                                
                                echo "Running unit tests..."
                                python -m pytest tests/unit/ \
                                    --cov=src/ \
                                    --cov-report=xml:coverage-unit.xml \
                                    --cov-report=html:htmlcov-unit \
                                    --cov-report=term-missing \
                                    --junitxml=junit-unit.xml \
                                    -v
                            '''
                        }
                    }
                    post {
                        always {
                            publishTestResults testResultsPattern: 'junit-unit.xml'
                            publishCoverage adapters: [
                                coberturaAdapter('coverage-unit.xml')
                            ], sourceFileResolver: sourceFiles('STORE_LAST_BUILD')
                            archiveArtifacts artifacts: 'htmlcov-unit/**', allowEmptyArchive: true
                        }
                    }
                }
                
                stage('Integration Tests') {
                    steps {
                        script {
                            // Start test services
                            sh '''
                                docker-compose -f docker-compose.test.yml up -d postgres redis influxdb
                                sleep 30
                            '''
                            
                            sh '''
                                . venv/bin/activate
                                
                                export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/test_db"
                                export REDIS_URL="redis://localhost:6379/0"
                                export INFLUXDB_URL="http://localhost:8086"
                                
                                echo "Running integration tests..."
                                python -m pytest tests/integration/ \
                                    --junitxml=junit-integration.xml \
                                    -v
                            '''
                        }
                    }
                    post {
                        always {
                            sh 'docker-compose -f docker-compose.test.yml down || true'
                            publishTestResults testResultsPattern: 'junit-integration.xml'
                        }
                    }
                }
                
                stage('E2E Tests') {
                    steps {
                        script {
                            sh '''
                                . venv/bin/activate
                                
                                export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/test_db"
                                export REDIS_URL="redis://localhost:6379/0"
                                export INFLUXDB_URL="http://localhost:8086"
                                
                                echo "Running E2E tests..."
                                python -m pytest tests/e2e/ \
                                    --junitxml=junit-e2e.xml \
                                    --maxfail=1 \
                                    -v
                            '''
                        }
                    }
                    post {
                        always {
                            publishTestResults testResultsPattern: 'junit-e2e.xml'
                        }
                    }
                }
            }
        }
        
        stage('Build Docker Image') {
            steps {
                script {
                    echo "Building Docker image: ${env.DOCKER_REGISTRY}/${env.IMAGE_NAME}:${env.IMAGE_TAG_FINAL}"
                    
                    // Build image
                    sh '''
                        docker build \
                            -t ${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG_FINAL} \
                            -t ${DOCKER_REGISTRY}/${IMAGE_NAME}:latest \
                            --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
                            --build-arg VCS_REF=${GIT_COMMIT_SHORT} \
                            --build-arg VERSION=${IMAGE_TAG_FINAL} \
                            .
                    '''
                    
                    // Security scan with Trivy
                    sh '''
                        echo "Running Trivy security scan..."
                        trivy image --format json --output trivy-report.json ${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG_FINAL} || true
                    '''
                }
            }
            post {
                always {
                    archiveArtifacts artifacts: 'trivy-report.json', allowEmptyArchive: true
                }
            }
        }
        
        stage('Push Docker Image') {
            when {
                anyOf {
                    branch 'main'
                    branch 'develop'
                    expression { params.FORCE_DEPLOY }
                }
            }
            steps {
                script {
                    // Login to Docker registry
                    sh '''
                        echo ${DOCKER_CREDENTIALS_PSW} | docker login ${DOCKER_REGISTRY} -u ${DOCKER_CREDENTIALS_USR} --password-stdin
                    '''
                    
                    // Push images
                    sh '''
                        docker push ${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG_FINAL}
                        docker push ${DOCKER_REGISTRY}/${IMAGE_NAME}:latest
                    '''
                    
                    // Generate SBOM
                    sh '''
                        echo "Generating SBOM..."
                        syft ${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG_FINAL} -o spdx-json=sbom.spdx.json || true
                    '''
                }
            }
            post {
                always {
                    sh 'docker logout ${DOCKER_REGISTRY} || true'
                    archiveArtifacts artifacts: 'sbom.spdx.json', allowEmptyArchive: true
                }
            }
        }
        
        stage('Deploy') {
            when {
                anyOf {
                    branch 'main'
                    branch 'develop'
                    expression { params.FORCE_DEPLOY }
                }
            }
            steps {
                script {
                    // Setup kubectl
                    sh '''
                        mkdir -p ~/.kube
                        echo "${KUBE_CONFIG}" | base64 -d > ~/.kube/config
                        chmod 600 ~/.kube/config
                    '''
                    
                    def namespace = "rl-trading"
                    if (params.ENVIRONMENT == 'dev') {
                        namespace = "rl-trading-dev"
                    } else if (params.ENVIRONMENT == 'staging') {
                        namespace = "rl-trading-staging"
                    }
                    
                    echo "Deploying to ${params.ENVIRONMENT} environment (namespace: ${namespace})"
                    
                    // Apply Kubernetes manifests
                    sh """
                        kubectl apply -f k8s/namespace.yaml
                        kubectl apply -f k8s/configmap.yaml
                        kubectl apply -f k8s/secret.yaml
                        kubectl apply -f k8s/rbac.yaml
                        kubectl apply -f k8s/pvc.yaml
                        kubectl apply -f k8s/deployment.yaml
                        kubectl apply -f k8s/service.yaml
                        kubectl apply -f k8s/ingress.yaml
                        kubectl apply -f k8s/hpa.yaml
                        kubectl apply -f k8s/network-policy.yaml
                    """
                    
                    // Update deployment image
                    def deploymentName = params.ENVIRONMENT == 'dev' ? 'rl-trading-system-dev' : 'rl-trading-system'
                    sh """
                        kubectl set image deployment/${deploymentName} \
                            rl-trading-system=${env.DOCKER_REGISTRY}/${env.IMAGE_NAME}:${env.IMAGE_TAG_FINAL} \
                            -n ${namespace}
                    """
                    
                    // Wait for rollout
                    sh """
                        kubectl rollout status deployment/${deploymentName} \
                            -n ${namespace} \
                            --timeout=600s
                    """
                    
                    // Verify deployment
                    sh """
                        kubectl get pods -l app=rl-trading-system -n ${namespace}
                        kubectl get services -n ${namespace}
                        kubectl get ingress -n ${namespace}
                    """
                }
            }
        }
        
        stage('Post-Deployment Tests') {
            when {
                anyOf {
                    branch 'main'
                    branch 'develop'
                    expression { params.FORCE_DEPLOY }
                }
            }
            steps {
                script {
                    def baseUrl = "https://trading.rlsystem.com"
                    if (params.ENVIRONMENT == 'dev') {
                        baseUrl = "https://trading-dev.rlsystem.com"
                    } else if (params.ENVIRONMENT == 'staging') {
                        baseUrl = "https://trading-staging.rlsystem.com"
                    }
                    
                    echo "Running post-deployment health checks..."
                    
                    // Wait for service to be ready
                    sleep(time: 60, unit: 'SECONDS')
                    
                    // Health checks
                    sh """
                        curl -f ${baseUrl}/api/health || exit 1
                        curl -f ${baseUrl}/api/ready || exit 1
                        curl -f ${baseUrl}/metrics || exit 1
                    """
                    
                    echo "All health checks passed!"
                }
            }
        }
    }
    
    post {
        always {
            script {
                // Clean up Docker images
                sh '''
                    docker image prune -f || true
                    docker system prune -f || true
                '''
                
                // Archive logs
                archiveArtifacts artifacts: '**/*.log', allowEmptyArchive: true
                
                // Clean workspace
                cleanWs()
            }
        }
        
        success {
            script {
                def message = """
✅ *Deployment Successful*

*Project:* ${env.JOB_NAME}
*Build:* #${env.BUILD_NUMBER}
*Branch:* ${env.BRANCH_NAME}
*Commit:* ${env.GIT_COMMIT_SHORT}
*Environment:* ${params.ENVIRONMENT}
*Image:* ${env.DOCKER_REGISTRY}/${env.IMAGE_NAME}:${env.IMAGE_TAG_FINAL}
*Duration:* ${currentBuild.durationString}

*Build URL:* ${env.BUILD_URL}
"""
                
                // Send Slack notification
                sh """
                    curl -X POST -H 'Content-type: application/json' \
                        --data '{"text":"${message}"}' \
                        ${env.SLACK_WEBHOOK}
                """
                
                // Send email notification for production deployments
                if (params.ENVIRONMENT == 'prod') {
                    emailext (
                        subject: "✅ Production Deployment Successful: ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                        body: message,
                        to: "devops@rltrading.com",
                        from: "jenkins@rltrading.com"
                    )
                }
            }
        }
        
        failure {
            script {
                def message = """
❌ *Deployment Failed*

*Project:* ${env.JOB_NAME}
*Build:* #${env.BUILD_NUMBER}
*Branch:* ${env.BRANCH_NAME}
*Commit:* ${env.GIT_COMMIT_SHORT}
*Environment:* ${params.ENVIRONMENT}
*Duration:* ${currentBuild.durationString}

*Build URL:* ${env.BUILD_URL}
*Console:* ${env.BUILD_URL}console

Please check the build logs for more details.
"""
                
                // Send Slack notification
                sh """
                    curl -X POST -H 'Content-type: application/json' \
                        --data '{"text":"${message}"}' \
                        ${env.SLACK_WEBHOOK}
                """
                
                // Send email notification
                emailext (
                    subject: "❌ Deployment Failed: ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                    body: message,
                    to: "devops@rltrading.com",
                    from: "jenkins@rltrading.com"
                )
            }
        }
        
        unstable {
            script {
                def message = """
⚠️ *Deployment Unstable*

*Project:* ${env.JOB_NAME}
*Build:* #${env.BUILD_NUMBER}
*Branch:* ${env.BRANCH_NAME}
*Commit:* ${env.GIT_COMMIT_SHORT}
*Environment:* ${params.ENVIRONMENT}
*Duration:* ${currentBuild.durationString}

*Build URL:* ${env.BUILD_URL}

Some tests may have failed, but deployment proceeded.
"""
                
                // Send Slack notification
                sh """
                    curl -X POST -H 'Content-type: application/json' \
                        --data '{"text":"${message}"}' \
                        ${env.SLACK_WEBHOOK}
                """
            }
        }
    }
}