package gonets

import (
	"errors"
	"fmt"
	"math"
)

type actFunc func([]float64) []float64

type unit struct {
	b float64
	w []float64
}

type hiddenLayer struct {
	i Layer
	a actFunc
	u []unit
}

func (l *hiddenLayer) AttachInputs(inputs Layer) error {
	for i := range l.u {
		if inputs.GetSize() != len(l.u[i].w) {
			return errors.New(fmt.Sprintf(
				"cannot attach layers, input layer dim: %+v, receiving layer dim: %+v", inputs.GetSize(), len(l.u[i].w),
			))
		}
	}
	l.i = inputs
	return nil
}

func (l *hiddenLayer) GetSize() int {
	return len(l.u)
}

func (l *hiddenLayer) GetOutput(inputs []float64) ([]float64, error) {
	inc, err := l.i.GetOutput(inputs)
	if err != nil {
		return nil, err
	}

	var linear_outputs = make([]float64, len(l.u))
	for i := range l.u {
		if len(l.u[i].w) != len(inc) {
			return nil, errors.New("incorrect dimensionality")
		}
		var linear_output float64
		for j := range l.u[i].w {
			linear_output += l.u[i].w[j] * inc[j]
		}
		linear_output += l.u[i].b
		linear_outputs[i] = linear_output
	}

	return l.a(linear_outputs), nil
}

type inputLayer struct {
	s int
}

func (l inputLayer) AttachInputs(inputs Layer) error {
	return errors.New("cannot attach inputs to input layer")
}
func (l inputLayer) GetSize() int {
	return l.s
}
func (l inputLayer) GetOutput(inputs []float64) ([]float64, error) {
	if len(inputs) != l.s {
		return nil, errors.New("incorrect dimensionality in InputLayer")
	}
	return inputs, nil
}

func rectify(in []float64) (out []float64) {
	out = make([]float64, len(in))
	for i, v := range in {
		if v > 0 {
			out[i] = v
		} else {
			out[i] = 0
		}
	}
	return
}

func tanh(in []float64) (out []float64) {
	out = make([]float64, len(in))
	for i, v := range in {
		out[i] = math.Tanh(v)
	}
	return
}

func softmax(in []float64) (out []float64) {
	out = make([]float64, len(in))
	var max float64
	for i, v := range in {
		out[i] = v
		if v > max {
			max = v
		}
	}
	var sum float64
	for i := range out {
		out[i] -= max
		out[i] = math.Exp(out[i])
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return
}

func logit(in []float64) (out []float64) {
	out = make([]float64, len(in))
	for i, v := range in {
		out[i] = 1 / (1 + math.Exp(-v))
	}
	return
}

func makeLayer(input Layer, activation actFunc, biases []float64, weights [][]float64) (Layer, error) {
	if len(biases) != len(weights) {
		return nil, errors.New("layer misspecification")
	}
	l := hiddenLayer{u: make([]unit, len(biases)), a: activation}
	for i := range biases {
		l.u[i] = unit{b: biases[i], w: weights[i]}
	}
	err := l.AttachInputs(input)
	if err != nil {
		return nil, err
	}
	return &l, nil
}
