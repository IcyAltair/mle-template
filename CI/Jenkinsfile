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
        stage('Clone github repository') {
            steps {
                cleanWs()
                bat 'chcp 65001 && git clone -b main https://github.com/IcyAltair/mle-template.git'
				}
			}

        stage('Checkout repo dir') {
            steps {
                bat 'chcp 65001 && cd mle-template && dir'
				}
			}

        stage('Login'){
            steps{
                //withCredentials([usernamePassword(credentialsId: 'mle-template', passwordVariable: 'DOCKER_REGISTRY_PWD', usernameVariable: 'DOCKER_REGISTRY_USER')]){
                //bat 'chcp 65001 && echo %DOCKER_REGISTRY_PWD% | docker login -u %DOCKER_REGISTRY_USER% --password-stdin'}
                //bat 'chcp 65001 && echo %DOCKERHUB_CREDS_PSW% | docker login -u %DOCKERHUB_CREDS_USR% --password-stdin'
                bat 'chcp 65001 && docker login -u %DOCKERHUB_CREDS_USR% -p %DOCKERHUB_CREDS_PSW%'
                }
            }

        stage('Create and run docker container') {
            steps {
                script {
                    try {
                        bat 'chcp 65001 && cd mle-template && docker-compose build'
                        }

                    finally {
                    bat '''
                        chcp 65001
                        cd mle-template
                        docker-compose up -d
                        '''
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