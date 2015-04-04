package gonets

import (
	"math"
	"testing"
)

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

func TestLasagneCompatibility(t *testing.T) {
	w1 := [][]float64{
		[]float64{-0.109595783122, 0.209483981573, 0.122117914097, 0.0716140910105, 0.112168624432, 0.0159258536371, -0.0151195257045, 0.0962302240911, 0.0458310692353, -0.0257266391724, -0.422622765543, -0.0165887479738, -0.218176051145, -0.317418433196, -0.179569247066, -0.0914008534842, -0.191939092518, -0.000108985369519, -0.174754967039, 0.160224194794, -0.0639363286407, -0.129051442081, 0.386961666847, 0.0150403736405, -0.70029036513, 0.460468224536, 0.121088748999, 0.117098491034},
		[]float64{0.211631211691, 0.134569394477, 0.115913647831, 0.0783742465631, 0.0413851115954, 0.0479523717229, 0.0475213332174, 0.122535496511, 0.0647260567506, 0.0886193293738, -0.0252346973127, -6.73335956438e-05, -0.0331124776276, -0.0985513085794, -0.0312408551554, -0.220824313021, -0.0558351858373, -0.0212985755574, 0.0195726728305, 0.053837776281, 0.0262370827495, 0.019485934755, 0.215210739953, 0.0243230441865, 0.0852560049886, 0.139319097071, -1.81127046853, 0.0507853151942},
		[]float64{-0.0835754939928, -0.375922688896, 0.0487980670745, -0.0112866143314, 0.0967032912317, 0.0316108126655, 0.0475973181855, 0.0341371087019, 0.00719269855435, -0.0686237820401, -0.181060805933, -0.0489050970871, -0.0851999611865, -0.0714608505252, -0.193938304972, 0.135735304938, -0.215817260336, -0.0372146245005, 0.102994743277, 0.0554789320264, -0.0611136747927, -0.0526758103667, 0.307219039928, 0.11109501764, -0.478557124179, 0.0555677729173, -1.39206529973, -0.145246469005},
		[]float64{0.729282052301, -1.07478893256, 0.438879343319, 0.0813187884668, 0.145962614996, 0.132920880587, 0.285901713238, 0.167462902015, 0.200951338505, 0.218269255499, -0.429428320235, -0.00972881578782, 0.0930611149157, 0.157340375191, 0.407451641692, 0.0276353807711, -0.0996143652946, -0.0851605921999, -0.0574387955867, 0.229387382637, 0.00450069390171, -0.0250284174177, 0.369870997728, -0.0533345239895, -0.95769297787, 0.603201435829, 0.015390587776, 0.31935992233},
		[]float64{0.0764285076257, -0.0972343313161, 0.0613440747912, 0.140477699738, 0.151804130187, 0.110119209635, 0.177345192999, 0.237278875926, 0.0927115482558, 0.057776602444, 0.0443569202995, -0.12813757285, -0.112509774897, -0.246624945246, -0.537158679623, 0.0431268143683, -0.277826764246, -0.248726000094, 0.0981974202112, -0.0377599799245, -0.00381517759309, 0.0689059526618, 0.7291501788, 0.0321694999756, 0.379698120135, -0.189547451852, 0.0753656864124, -0.469598580609},
	}
	b1 := []float64{1.06529159066, -0.271159499769, -0.360616089225, 0.758574607807, 0.18340392505}
	w2 := [][]float64{
		[]float64{0.617280342307, -0.929100661125, -0.647544732776, 0.621824740939, 1.05812547924},
		[]float64{-0.210913081076, 0.930927754667, 0.895306960904, -0.056629859462, 0.194744500807},
	}
	b2 := []float64{1.03980357614, -1.03980357614}
	l0 := InputLayer(28)
	l1, err := RectifyLayer(l0, b1, w1)
	if err != nil {
		t.Fatal(err)
	}
	l2, err := SoftmaxLayer(l1, b2, w2)
	if err != nil {
		t.Fatal(err)
	}

	inp := []float64{-0.311443752632, -0.33276644248, -0.279749856683, -0.23357894362, -0.388451400757, -0.290115813651, -0.288566611959, -0.486674811383, -0.336321895253, 2.87178876171, -0.0264714918414, -0.470125525019, -0.376509260503, 1.92171443031, -0.327773601422, -0.134484865699, -0.517244623798, -0.670748122659, 1.30312227345, -0.293501652974, -1.11866423067, -0.646212635966, 2.69934924792, 1.3389814651, -1.05340000685, 0.827605976572, 1.79816100342, 0.739421624031}
	target := []float64{9.99812890e-01, 1.87110350e-04}

	out, err := l2.GetOutput(inp)
	if err != nil {
		t.Fatal(err)
	}
	approx := func(a, b float64) bool {
		if math.Abs(a-b) <= 0.01 {
			return true
		}
		return false
	}
	for i := range out {
		if !approx(target[i], out[i]) {
			t.Fatalf("%+v != %+v", out[i], target[i])
		}
	}
}