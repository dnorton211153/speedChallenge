//go:build ignore
// +build ignore

package main

import (
	"fmt"

	t "gorgonia.org/tensor"
)

func main() {

	inputs := []float32{1.0, 2.0, 2.5, 3.0}

	// layer_outputs := []float32{}

	// weights are randomized between -1 and 1
	weights := [][]float32{
		{0.1, 0.1, -0.3, 0.1},
		{0.1, 0.2, 0.0, -0.1},
		{0.0, -0.3, 0.1, 0.2},
	}

	// biases are randomized between 0 and 3
	biases := []float32{2.0, 3.0, 0.3}

	fmt.Print("Hello World")

	inputTensor := t.New(t.WithBacking(inputs))
	weightTensor := t.New(t.WithShape(3, 4), t.WithBacking(weights))
	// weightTensor := t.New(t.WithBacking(weights))

	// log
	fmt.Println(inputTensor)
	fmt.Println(weightTensor)

	// create a tensor with the bias of the same type (float32)
	biasTensor := t.New(t.WithBacking(biases))

	fmt.Println(biasTensor)

	outputTensor, err := t.Dot(weightTensor, inputTensor)
	if err != nil {
		panic(err)
	}

	outputTensor, err = t.Add(outputTensor, biasTensor)
	if err != nil {
		panic(err)
	}

	fmt.Println(outputTensor)

}
