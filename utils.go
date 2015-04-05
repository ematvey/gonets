package gonets

import "errors"

type Normalizer struct {
	mean []float64
	std  []float64
}

func NormalizerNew(mean, std []float64) (*Normalizer, error) {
	if len(mean) != len(std) || len(mean) == 0 {
		return nil, errors.New("incorrect lenghts")
	}
	return &Normalizer{mean, std}, nil
}

func (n *Normalizer) Normalize(in []float64) (out []float64) {
	out = make([]float64, len(in))
	for i := range in {
		out[i] = in[i]*n.std[i] + n.mean[i]
	}
	return
}
