import onnxruntime
import numpy as np
from PIL import Image
import sys
import glob
import os

def list_of_files(path, starts_with='', ends_with='.png'):
    return [fileName for fileName in os.listdir(path) if
            fileName.endswith(ends_with) and fileName.startswith(starts_with)]

def run_inference(model):
    # Load the ONNX model
    session = onnxruntime.InferenceSession(model)

    # Get the input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    for file in list_of_files("/app/images/"):

        # Load the input image and preprocess it
        image = Image.open("/app/images/"+ file)
        image = image.convert('L')  # convert to grayscale
        image = image.resize((28, 28))  # resize to 28x28 pixels
        image = np.array(image).astype(np.float32) / 255.0  # normalize and convert to float
        image = image.reshape(1, 28, 28)  # add batch dimension

        # Run inference
        output_data = session.run([output_name], {input_name: image})[0]

        # The output is a probability distribution over the 10 classes
        print(output_data)

        # Get the predicted class
        predicted_class = int(output_data.argmax())
        # Write the result to the log file
        with open(model+"-output.log", "a") as f:
            f.write("Prediction: " + str(predicted_class))
        print(file)
        print('Prediction:', predicted_class)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input', type=str, help='Path to input image',
    #                     default='image.png')
    # parser.add_argument('--output', type=str, help='Output Prediction',
    #                     default='output.log')
    # parser.add_argument('--model', type=str, help='Model File',
    #                     default='mnist.onnx')
    #
    # args = parser.parse_args()
    run_inference(sys.argv[1])
