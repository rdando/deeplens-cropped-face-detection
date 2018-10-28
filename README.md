# AWS DeepLens Cropped Face Detection
Source code for getting started with AWS DeepLens face detection and IoT Rules.

### Deployment
Made and deployed using [AWS Serverless Application Model (AWS SAM)](https://github.com/awslabs/serverless-application-model) extension of CloudFormation.

```bash
aws cloudformation package \
--template-file template.yaml \
--output-template-file template-out.yaml \
--s3-bucket <your-s3-bucket-name>

aws cloudformation deploy \=
--template-file <path-to-file>/template-out.yaml \
--parameter-overrides DeepLensTopic=<DEEPLENS_TOPIC> \
--stack-name <STACK_NAME>  \
--capabilities CAPABILITY_IAM
```