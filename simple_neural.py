import numpy as np

vodka = 1.0
rain = 1.0
friend = 1.0

def activation_function(x):
	return 1 if x >= 0.5 else 0

def predict(vodka, rain, friend):
	inputs = np.array([vodka, rain, friend])
	weights_input_to_hidden_1 = [0.25, 0.25, 0]
	weights_input_to_hidden_2 = [0.5, -0.4, 0.9]
	weights_input_to_hidden = np.array([weights_input_to_hidden_1, weights_input_to_hidden_2])

	weights_hidden_to_output = np.array([-1, 1])

	hidden_input = np.dot(weights_input_to_hidden, inputs)
	print(f"hidden input: {hidden_input}")

	hidden_output = np.array(list(map(activation_function, hidden_input)))
	print(f"hidden output: {hidden_output}")

	output = np.dot(weights_hidden_to_output, hidden_output)
	print(f"output: {output}")

	return activation_function(output) == 1

print(f"result: {predict(vodka, rain, friend)}")