def mcculloch_pitts_neuron(inputs, weights, threshold):
    if len(inputs) != len(weights):
        raise ValueError("Number of inputs must be equal to the number of weights")
    weighted_sum = sum(x * w for x, w in zip(inputs, weights))

    if weighted_sum >= threshold:
        return 1
    else:
        return 0

inputs = [1, 1, 2]
weights = [1, 0.5, 0.5]
threshold = 1.5

output = mcculloch_pitts_neuron(inputs, weights, threshold)
print("Output:", output)
