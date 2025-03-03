# Use the AWS Lambda Python base image for Python 3.12
FROM public.ecr.aws/lambda/python:3.12

# Set the working directory inside the container
WORKDIR /var/task

# Create the /tmp directory.
RUN mkdir -p  /tmp

# Copy the requirements file to the working directory
COPY requirements.txt ./

# Install dependencies to the working directory
RUN pip install --no-cache-dir -r requirements.txt -t .

# Copy the function code to the working directory
COPY utils.py ./
COPY extract-volume-data.py ./

# Command for the Lambda runtime to run the handler function
CMD ["extract-volume-data.lambda_handler"]

