package gonets

import "testing"

func TestBasicNetDirect(t *testing.T) {
	inp_l := Layer(inputLayer{3})
	sfm := Layer(&hiddenLayer{
		u: []unit{
			unit{0.0, []float64{0.3, 0.4, -0.1}},
			unit{0.1, []float64{-0.1, 0.1, 0.0}},
			unit{-0.3, []float64{-0.3, 0.0, 0.0}},
		},
		a: Softmax,
	})
	sfm.AttachInputs(inp_l)
	o, err := sfm.GetOutput([]float64{3, 2, 1})
	if err != nil {
		t.Fail()
	}
	tgt := []float64{0.7919496226186542, 0.1598918712338877, 0.048158506147458105}
	for i := range tgt {
		if o[i] != tgt[i] {
			t.Fail()
		}
	}
}

func TestBasicNetConstructors(t *testing.T) {
	biases := []float64{0.0, 0.1, -0.3}
	weights := [][]float64{
		{0.3, 0.4, -0.1},
		{-0.1, 0.1, 0.0},
		{-0.3, 0.0, 0.0},
	}
	sfm, err := SoftmaxLayer(InputLayer(3), biases, weights)
	if err != nil {
		t.Fatal(err)
	}
	o, err := sfm.GetOutput([]float64{3, 2, 1})
	if err != nil {
		t.Fatal(err)
	}
	tgt := []float64{0.7919496226186542, 0.1598918712338877, 0.048158506147458105}
	for i := range tgt {
		if o[i] != tgt[i] {
			t.Fatalf("%+v != %+v", o[i], tgt[i])
		}
	}
}
