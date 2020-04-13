import boto3
import os
import json

TRAINING_JOB_NAME = os.environ.get('TRAINING_JOB_NAME') or 'sagemaker-scikit-learn-2020-04-13-11-36-57-229'
STACK_NAME = os.environ.get('STACK_NAME')

dir_path = os.path.dirname(os.path.realpath(__file__))

smc = boto3.client('sagemaker')

def get_job(training_job_name):
    job = smc.describe_training_job(TrainingJobName=training_job_name)
    role_arn = job['RoleArn']
    model_data_url = job['ModelArtifacts']['S3ModelArtifacts']
    training_image = job['AlgorithmSpecification']['TrainingImage']
    inference_code = job['HyperParameters']['sagemaker_submit_directory'].replace('"', '')
    return (job, role_arn, model_data_url, training_image, inference_code)



cfn = boto3.client('cloudformation')

with open(dir_path + '/output.yaml', 'r') as f:
    _, role_arn, model_data_url, training_image, inference_code = get_job(TRAINING_JOB_NAME)
    stack = cfn.create_stack(StackName=STACK_NAME,
            TemplateBody = f.read(),
            Capabilities = ['CAPABILITY_NAMED_IAM', 'CAPABILITY_AUTO_EXPAND'],
            Parameters=[
                    {'ParameterKey':'ModelName', 'ParameterValue': TRAINING_JOB_NAME},
                    {'ParameterKey':'TrainingImage', 'ParameterValue': training_image},
                    {'ParameterKey':'ModelDataUrl', 'ParameterValue': model_data_url},
                    {'ParameterKey':'RoleArn', 'ParameterValue': role_arn},
                    {'ParameterKey':'InferenceCode', 'ParameterValue': inference_code}])

