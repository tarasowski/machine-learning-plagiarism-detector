CFN_ARTIFACTS_BUCKET := 'plagiarism-detector-sagemaker-tarasowski'
AWS_REGION := 'eu-central-1'

create_bucket:
	@aws s3api create-bucket --bucket $(CFN_ARTIFACTS_BUCKET) --region $(AWS_REGION) --create-bucket-configuration LocationConstraint=$(AWS_REGION)

download:
	@curl https://s3.amazonaws.com/video.udacity-data.com/topher/2019/January/5c4147f9_data/data.zip -o ./input/data.zip
	@unzip ./input/data.zip -d ./input/
	@rm ./input/data.zip

preprocess:
	DATA_DIR=./input/data/test_info.csv SAVE_DIR=./models/ python3 ./src/preprocess.py

train_cloud:
	MODE=cloud python3 ./jobsubmit.py

train_local:
	MODE=local python3 ./jobsubmit.py

deploy_model:
ifdef JOB_NAME
ifdef STACK_NAME
	@aws cloudformation package --template-file ./infra/app/main.template.yaml --output-template-file ./infra/app/output.yaml --s3-bucket $(CFN_ARTIFACTS_BUCKET) --region eu-central-1
	TRAINING_JOB_NAME=$(JOB_NAME) STACK_NAME=$(STACK_NAME) python3 ./infra/app/deploy.py
else
	$(error "Please provide following arguments: job_name=string, stack_name=string")
endif
endif

predict:
	python3 ./get_predictions.py
