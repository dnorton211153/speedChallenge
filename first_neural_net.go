//go:build ignore
// +build ignore

package main

import "fmt"

var (
	inputs = []float64{
		1, 2, 2.5, 3,
	}

	layer_outputs = []float64{}

	// weights are randomized between -1 and 1
	weights = [][]float64{
		{0.1, 0.1, -0.3, 0.1}, // each represent a neuron
		{0.1, 0.2, 0.0, -0.1},
		{0.0, -0.3, 0.1, 0.2},
	}

	// biases are randomized between 0 and 3
	biases = []float64{
		2, 3, 0.3,
	}
)

func main() {

	for i, neuronWeights := range weights {

		// neuronWeights is the weights for a single neuron
		neuron_output := 0.0

		// grab the bias for this neuron
		neuron_bias := biases[i]

		for j, weight := range neuronWeights {

			// grab the input for this neuron
			// (the inputs are the same for each neuron)
			input := inputs[j]

			// increment the neuron output by the input * weight
			neuron_output += input * weight
		}

		// add the bias to the neuron output
		neuron_output += neuron_bias

		// add the neuron output to the layer outputs
		layer_outputs = append(layer_outputs, neuron_output)
	}

	fmt.Println(layer_outputs)
}
