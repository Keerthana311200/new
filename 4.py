import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])

outputs = np.array([[0],
                    [1],
                    [1],
                    [0]])
input_layer_neurons = inputs.shape[1] 
hidden_layer_neurons = 2  
output_neurons = 1 
wh = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bh = np.random.uniform(size=(1, hidden_layer_neurons))
wo = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
bo = np.random.uniform(size=(1, output_neurons))
learning_rate = 0.1
epochs = 10000
for epoch in range(epochs):
    hidden_layer_input = np.dot(inputs, wh) + bh
    hidden_layer_activation = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_activation, wo) + bo
    predicted_output = sigmoid(output_layer_input)
    error = outputs - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(wo.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_activation)
    wo += hidden_layer_activation.T.dot(d_predicted_output) * learning_rate
    bo += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    wh += inputs.T.dot(d_hidden_layer) * learning_rate
    bh += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}")
print("Final predicted outputs:")
print(predicted_output)
new_sample = np.array([[1, 0]])
hidden_layer_input = np.dot(new_sample, wh) + bh
hidden_layer_activation = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_activation, wo) + bo
new_sample_output = sigmoid(output_layer_input)

print("Prediction for the new sample [1, 0]:", new_sample_output)
