pipeline {
    agent any

    environment {
        DOCKER_HUB_REPO = 'your-dockerhub-username/medibot'
        KUBE_CONFIG_ID = 'kube-config' // ID of your Kubeconfig credentials in Jenkins
        DOCKER_REGISTRY_CREDENTIALS_ID = 'docker-hub-credentials' // ID of your DockerHub credentials in Jenkins
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    dockerImage = docker.build("${DOCKER_HUB_REPO}:${env.BUILD_NUMBER}")
                }
            }
        }

        stage('Push Docker Image') {
            steps {
                script {
                    docker.withRegistry('', DOCKER_REGISTRY_CREDENTIALS_ID) {
                        dockerImage.push()
                        dockerImage.push('latest')
                    }
                }
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                withKubeConfig([credentialsId: KUBE_CONFIG_ID]) {
                    sh 'kubectl apply -f k8s/deployment.yaml'
                    sh 'kubectl apply -f k8s/service.yaml'
                    
                    // Force rollout to pick up new image tag if using latest, 
                    // or ideally update the deployment yaml to use specific build tag.
                    // For this example, we simply rollout restart.
                    sh 'kubectl rollout restart deployment/medibot-deployment'
                }
            }
        }
    }
}
