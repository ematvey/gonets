# GoNets
Running Neural Networks in Go

GoNets allows you to specify and run feed-forward neural networks. It does not support training, and I have no plans to implement that in immediate future. The goal of this package is to integrate nets trained elsewhere into golang projects.

Specifying networks is as simple as:
```golang
biases := []float64{0.0, 0.1, -0.3}
weights := [][]float64{
    {0.3, 0.4, -0.1},
    {-0.1, 0.1, 0.0},
    {-0.3, 0.0, 0.0},
}
net, err := SoftmaxLayer(InputLayer(3), biases, weights)
predictions, err := net.GetOutput([]float64{3, 2, 1})
```