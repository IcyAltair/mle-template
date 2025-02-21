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
                sh 'ls -lash'
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
                        sh 'docker-compose build'
                        }

                    finally {
                        sh'docker-compose up -d'
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
                        bat '''
                            docker-compose up -d
                            for /f %%i in ('docker ps -qf "name=^mle-template-web-1"') do set containerId=%%i
                            echo %containerId%
                            IF "%containerId%" == "" (
                                echo "No container running"
                            )
                            ELSE (
                                docker logs --tail 1000 -f %containerId%
                                )
                        '''
                    }
            }
        }

        stage('Checkout coverage report'){
            steps{
                dir("mle-template"){
                    bat '''
                    docker-compose logs -t --tail 10
                    '''
                }
            }
        }

        stage('Push'){
            steps{
                bat 'chcp 65001 && docker push altairzero/mle-template:latest'
            }
        }
	}

    post {
        always {
            bat 'chcp 65001 && docker logout'
        }
    }
}