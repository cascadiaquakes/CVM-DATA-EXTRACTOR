import aws_cdk as core
import aws_cdk.assertions as assertions

from infra.CvmExtractLambdas.cvm_extract_lambdas.cvm_extract_lambda_stack import CvmExtractLambdaStack

# example tests. To run these tests, uncomment this file along with the example
# resource in cvm_extract_lambdas/cvm_extract_lambdas_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = CvmExtractLambdaStack(app, "cvm-extract-lambdas")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
