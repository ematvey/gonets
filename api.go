package gonets

import "errors"

type Layer interface {
	AttachInputs(Layer) error
	GetOutput([]float64) ([]float64, error)
	GetSize() int
}

type Network interface {
	GetOutput([]float64) ([]float64, error)
	GetSize() int
}

// API function to create Input layer
func InputLayer(size int) Layer {
	return Layer(inputLayer{size})
}

// API function to create layer with Rectifier units
//
// Rectifier is a neuron that outputs linear if activation is above 0 and 0 otherwise
func RectifyLayer(input Layer, biases []float64, weights [][]float64) (Layer, error) {
	return makeLayer(input, rectify, biases, weights)
}

// API function to create layer with Rectifier units
//
// Tanh is activation function that outputs close to 1 when activation is high and
// close to 0 if activation is low
func TanhLayer(input Layer, biases []float64, weights [][]float64) (Layer, error) {
	return makeLayer(input, tanh, biases, weights)
}

// API function to create Softmax layer
//
// Softmax outputs activation that sums to 1 after exponentiation
func SoftmaxLayer(input Layer, biases []float64, weights [][]float64) (Layer, error) {
	return makeLayer(input, softmax, biases, weights)
}

// API function to create Logit layer
//
// Logit layer currently supports only 1d output
func LogitLayer(input Layer, biases []float64, weights [][]float64) (Layer, error) {
	if len(biases) != 1 {
		return nil, errors.New("Logit layer currently supports only 1d output")
	}
	return makeLayer(input, softmax, biases, weights)
}
