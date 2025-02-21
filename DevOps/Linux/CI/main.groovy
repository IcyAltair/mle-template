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
        stage('Checkout repo dir') {
            steps {
                    sh 'git clone -b feature/api-call https://github.com/IcyAltair/mle-template.git'
                    sh 'cd mle-template && ls -lash'
                    sh 'whoami'
				}
			}

        stage('Login'){
            steps{
                    sh "docker login -u ${DOCKERHUB_CREDS_USR} -p ${DOCKERHUB_CREDS_PSW}"
                }
            }

        stage('Create and run docker container') {
            steps {
                script {
                    try {
                            sh 'cd mle-template && docker compose build'
                        }

                    finally {
                            sh 'cd mle-template && docker compose up -d'
                        }
				    }
                }
            }

        // use for multi containers:
        //docker exec %containerId% cd app && coverage run src/unit_tests/test_preprocess.py && coverage run -a src/unit_tests/test_training.py && coverage report -m
        //use for single container (with auth):
        //docker run --name mle-template_web_1 cd app && coverage run src/unit_tests/test_preprocess.py && coverage run -a src/unit_tests/test_training.py && coverage report -m

        stage('Checkout container logs') {
            steps {
                dir("mle-template") {
                        sh '''
                            docker compose up -d
                            containerId=$(docker ps -qf "name=^mle-template-web-1")
                            if [[ -z "$containerId" ]]; then
                                echo "No container running"
                            else
                                
                                docker logs --tail 1000 -f "$containerId"
                            fi
                        '''
                    }
            }
        }

        stage('Checkout coverage report'){
            steps{
                dir("mle-template"){
                        sh '''
                        docker compose logs -t --tail 10
                        '''
                }
            }
        }

        stage('Push'){
            steps{
                    sh 'docker push altairzero/mle-template:latest'
            }
        }
	}

    post {
        always {
                sh 'docker logout'
                cleanWs()
        }
    }
}