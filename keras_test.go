// +build ignore

package main

# Incomplete GoLang implementation
import (
	"github.com/leeyikjiun/go-keras"
)

func main() {

	layers := []keras.Layer{
		keras.NewDense(32, keras.WithInputShape(784)),
		keras.NewActivation("relu"),
		keras.NewDense(10),
		keras.NewActivation("softmax"),
	}

	model := keras.Sequential()
	model.Add(keras.Dense(32, keras.Input(784)))
	model.Add(keras.Activation("relu"))
	model.Add(keras.Dense(10))
	model.Add(keras.Activation("softmax"))
	model.Compile(keras.Loss("categorical_crossentropy"), keras.Optimizer("rmsprop"), keras.Metrics("accuracy"))
}
