pipeline {
    agent any

    environment {
        DOCKERHUB_CREDS=credentials('mle-template')
        LC_ALL = "en_US.UTF-8"
        LANG    = "en_US.UTF-8"
        LANGUAGE = "en_US.UTF-8"
    }

options {
        timestamps()
        skipDefaultCheckout(true)
	}
    stages {

        stage('Login'){
            steps{
                //withCredentials([usernamePassword(credentialsId: 'mle-template', passwordVariable: 'DOCKER_REGISTRY_PWD', usernameVariable: 'DOCKER_REGISTRY_USER')]){
                //bat 'chcp 65001 && echo %DOCKER_REGISTRY_PWD% | docker login -u %DOCKER_REGISTRY_USER% --password-stdin'}
                //bat 'chcp 65001 && echo %DOCKERHUB_CREDS_PSW% | docker login -u %DOCKERHUB_CREDS_USR% --password-stdin'
                bat 'docker login -u %DOCKERHUB_CREDS_USR% -p %DOCKERHUB_CREDS_PSW%'
                }
            }

        stage('Pull image'){
            steps{
                bat '''
                        docker pull altairzero/mle-template:latest
                '''
            }
        }

        stage('Run container'){
            steps{
                bat '''
                        docker run --name mle-template -p 80:5556 -d altairzero/mle-template:latest
                '''
            }
        }
	}

    post {
        always {
            bat 'docker stop mle-template && docker logout'
        }
    }
}
