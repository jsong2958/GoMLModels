package models

import (
	"fmt"
	"math"
	"math/rand"
)

type Neuron struct {
	in      int
	weights []float64
	bias    float64
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

	var out float64 = 1 / (1 + math.Exp(-z))
	return out
}

type Layer struct {
	in      int
	out     int //How many neurons in a layer
	Neurons []Neuron
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

	for i := 0; i < l.out; i++ {
		outs[i] = l.Neurons[i].call(input)
	}

	return outs
}

type MLP struct {
	in     int
	outs   []int
	Layers []Layer
}

func newMLP(inSize int, outSizes []int) MLP {
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

func (mlp MLP) call(input []float64) []float64 {
	output := input
	for _, layer := range mlp.Layers {
		output = layer.call(output)
	}
	return output
}

func main() {

	var x []float64 = []float64{1.2, 2.0, 3.0, 1.4}

	var mlp MLP = newMLP(4, []int{3, 3, 1})

	final := mlp.call(x)

	fmt.Printf("Output: %.2f\n", final)

}
