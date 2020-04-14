# Detect Plagiarism with 100% Accuary

The goal of the project was to build a machine learning system that can detect plagiarism. The whole project is built on the AWS Sagemaker service. It can be deployed and managed via Cloudformation. The inference happens in real-time through a Sagemaker Endpoint.

## Pre-requisites

* Python v3.8
* Boto3 v1.11.14
* Numpy v1.18.1
* Pandas v1.0.1
* Sagemaker v1.55.3
* Scikit-learn v0.22.1
* Run pip install 'sagemaker[local]' --upgrade to use local mode
* See the full list under `requirements.txt`

## Getting Started

The machine learning pipeline can run locally as well in the cloud. Due to a limitation of Cloudformation a training job needs to be deployed from a local machine and cannot be packed into infrastructure as code. The current repository includes all the code needed to start the training job, to create a model and to
make the prediction. Following commands can be used to kick-off the pipeline:

* Download the training data from the servers
  * Basic usage: `make download`

* Preprocess the data, generate features and save the file
  * Basic usage: `make preprocess`

* Start a training job on the local machine and test the results
  * Basic usage: `make train_local`

* Start a training job on AWS Sagemaker. RandomForest no need for GPU.
  * Basic usage: `make train_cloud`

* Deploy the model and an endpoint on AWS Sagemaker (Cloudformation)
  * Basic usage: `make create_bucket`
  * Basic usage: `make deploy_model JOB_NAME=<String> STACK_NAME=<String>`
  * Description: The job name is shown in the Sagemaker console

* Make predictions through Sagemaker Endpoint.
  * Basic usage: `make prediction ENDPOINT_NAME=<String>`
  * Description: The endpoint name can be is under outputs in Cloudformation

**Important**: The AWS Sagemaker endpoints are billed per second. The endpoints are not serverless. Therefore the endpoint should be deleted if not in use. The deletion of the CloudFormation stack also deletes the AWS Sagemaker endpoint.

## Support
Patches are encouraged and may be submitted by forking this project and submitting a pull request through GitHub.

## Credits
The project was developed during the ML program of
[Udacity.com](https://www.udacity.com/)

## Licence
Released under the [MIT License](./License.md)

