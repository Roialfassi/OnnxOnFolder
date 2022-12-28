# Start from a base image with TensorFlow and TensorRT installed
# FROM nvcr.io/nvidia/tensorrt:19.05-py3

FROM python:3.8
RUN pip install tensorflow onnxruntime numpy Pillow

# Set the working directory
WORKDIR /app

# Copy the MNIST model file to the container
COPY test/ /app/images/

# Copy the script for running the inference
COPY checkResults.py .

COPY checkResults.py /app/checkResults.py
COPY  mnist.onnx /app/mnist.onnx

# Set the entrypoint to the script
ENTRYPOINT ["python", "checkResults.py"]

# Set the command line arguments to the path to the PNG file and the log file
CMD ["/app/mnist.onnx"]
