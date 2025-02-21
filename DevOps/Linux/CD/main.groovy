pipeline {
    agent any

    environment {
        DOCKERHUB_CREDS=credentials('mle-template')
    }

options {
        timestamps()
        skipDefaultCheckout(true)
	}
    stages {
        stage('Login'){
            steps{
                    sh "docker login -u ${DOCKERHUB_CREDS_USR} -p ${DOCKERHUB_CREDS_PSW}"
                }
            }

        stage('Pull image'){
            steps{
                sh '''
                        docker pull altairzero/mle-template:latest
                '''
            }
        }

        stage('Run container'){
            steps{
                sh '''
                        docker run --name mle-template-test -p 80:5556 -d altairzero/mle-template:latest
                '''
            }
        }
	}

    post {
        always {
            sh 'docker stop mle-template-test && docker logout'
            cleanWs()
        }
    }
}
