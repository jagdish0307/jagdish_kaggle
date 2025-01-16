## Digit Recognizer Using CNN

- **Importing All Necessary Libraries**
- **Loading the Dataset from Kaggle**
- **Checking for Null and Duplicate Values**

- **Data Visualization:**

    1. Using a pie chart to visualize the percentage distribution of handwritten digits
    2. Using a bar graph to display the count distribution of handwritten digits
    3. Visualizing a specific digit using its pixel features
    4. Analyzing the pixel intensity of a particular digit with a bar graph
- **Splitting Data into X and y:**

    **After splitting, the data is divided into training and validation sets, with 20% reserved for validation.**

- **CNN Model Building:**

    1. The model uses 3 convolution layers with input size 28x28. Each layer has 32, 64, and 128 neurons, respectively.
    2. The activation function applied is tanh to detect non-linear patterns in the data.
    3.MaxPooling2D with a pool size of 2x2 is applied.
    4. Dropout is applied after each layer with rates of 0.25, 0.25, and 0.5, respectively.
    5. After the convolutional layers, a Flatten layer is used to convert the multidimensional output into a one-dimensional vector.
    6. A fully connected dense layer with 128 nodes and the tanh activation function is added.
    7. The final output layer has 10 nodes for classification, as there are 10 classes. The activation function used is softmax for multi-class classification.

- **Model Compilation:**

    1. The model is compiled using the Adam optimizer, categorical_crossentropy loss function, and accuracy as the evaluation metric.

- **Training the Model:**

    1. The model was trained for 15 epochs with a batch size of 128.
    2. Training Dataset Results:
    3. Accuracy: 0.9860
    4. Categorical Crossentropy Loss: 0.0411
    5. Validation Dataset Results:
    6. Accuracy: 0.9877
    7. Categorical Crossentropy Loss: 0.0411

- **Changing Activation Function to ReLU Instead of Tanh:**

    1. The model was retrained for 15 epochs with a batch size of 128.
    2. Training Dataset Results:
    3. Accuracy: 0.9908
    4. Categorical Crossentropy Loss: 0.0334
    5. Validation Dataset Results:
    6. Accuracy: 0.9910
    7. Categorical Crossentropy Loss: 0.029
       
 ## Final Results:
    1. Final validation accuracy: 0.9910
    2. Final categorical crossentropy loss: 0.029
    3. On unseen Kaggle data, the model achieved an accuracy of 0.98964.

