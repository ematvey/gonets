package gonets

type ActivationFunc func([]float64) []float64

type Layer interface {
	AttachInputs(Layer) error
	GetOutput([]float64) ([]float64, error)
	GetSize() int
}

func InputLayer(size int) Layer {
	return Layer(inputLayer{size})
}

func RectifyLayer(input Layer, biases []float64, weights [][]float64) (Layer, error) {
	return MakeLayer(input, Rectify, biases, weights)
}

func SoftmaxLayer(input Layer, biases []float64, weights [][]float64) (Layer, error) {
	return MakeLayer(input, Softmax, biases, weights)
}
