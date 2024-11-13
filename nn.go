package main

import (
	"math"
	"math/rand"
)

type Neuron struct {
	in          int
	weights     []float64
	bias        float64
	activations float64
}

func newNeuron(num int) Neuron {
	n := Neuron{
		in:      num,
		weights: make([]float64, num),
		bias:    2*rand.Float64() - 1,
	}

	for i := 0; i < num; i++ {
		n.weights[i] = 2*rand.Float64() - 1
	}
	return n
}

func (n Neuron) call(input []float64) float64 {

	//Calculates the weighted sum of that neuron from a given input
	var z float64
	for i := 0; i < n.in; i++ {
		z += input[i] * n.weights[i]
	}

	z += n.bias

	a := math.Tanh(z)

	n.activations = a
	return a

}

type Layer struct {
	in          int
	out         int //How many neurons in a layer
	Neurons     []Neuron
	activations []float64
}

func newLayer(inSize int, outSize int) Layer {
	l := Layer{
		in:      inSize,
		out:     outSize,
		Neurons: make([]Neuron, outSize),
	}
	for i := 0; i < outSize; i++ {
		l.Neurons[i] = newNeuron(inSize)
	}

	return l
}

func (l Layer) call(input []float64) []float64 {

	outs := make([]float64, l.out)

	for i := range l.Neurons {
		outs[i] = l.Neurons[i].call(input)
	}

	return outs
}

type MLP struct {
	in     int
	outs   []int
	Layers []Layer
}

func NewMLP(inSize int, outSizes []int) MLP {
	layers := MLP{
		in:     inSize,
		outs:   outSizes,
		Layers: make([]Layer, len(outSizes)),
	}
	prevSize := inSize
	for i := 0; i < len(outSizes); i++ {
		layers.Layers[i] = newLayer(prevSize, outSizes[i])
		prevSize = outSizes[i]
	}

	return layers
}

func (mlp MLP) Forward(input []float64) []float64 {
	output := input
	for _, layer := range mlp.Layers {
		output = layer.call(output)
	}
	return output
}

func (mlp *MLP) Backward(preds, input []float64, learningRate float64) {

	outputErrors := make([]float64, len(input))

	for i := range preds {
		outputErrors[i] = preds[i] - input[i]
	}

	for l := len(mlp.Layers) - 1; l >= 0; l-- {
		layer := &mlp.Layers[l]
		nextLayerError := make([]float64, layer.in)

		for j, n := range layer.Neurons {
			a := n.activations
			dadzVal := dadz(a)
			dCdz := outputErrors[j] * dadzVal

			for k := 0; k < n.in; k++ {
				inputActivation := n.activations
				weightGradient := dCdz * inputActivation
				n.weights[k] -= learningRate * weightGradient
				nextLayerError[k] += n.weights[k] * dCdz
			}

			n.bias -= learningRate * dCdz

		}
		outputErrors = nextLayerError
	}

}

func Cost(preds, actuals []float64) float64 { // MSE cost function
	var cost float64
	for i := range preds {
		cost += math.Pow(preds[i]-actuals[i], 2)
	}
	return cost / float64(len(preds))
}

func dadz(x float64) float64 { // da/dz
	return (1 - math.Pow(math.Tanh(x), 2))
}
