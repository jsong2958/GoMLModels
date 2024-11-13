package main

import "fmt"

func main() {

	input := [][]float64{
		{1.2, 3.6, 9.1},
		{2.3, 6.1, -2.5},
		{1.1, -5.5, -4.3},
	}

	target := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
	}

	mlp := NewMLP(3, []int{3, 3, 1})

	learningRate := 0.001
	epochs := 10000

	for epoch := 0; epoch < epochs; epoch++ {
		totCost := 0.0

		for i := 0; i < len(input); i++ {

			pred := mlp.Forward(input[i])

			cost := Cost(pred, target[i])
			totCost += cost

			// Backward pass with error from the target
			mlp.Backward(pred, target[i], learningRate)
		}

		if epoch%1000 == 0 {
			fmt.Printf("Epoch %d, Cost: %f\n", epoch, totCost/float64(len(input)))
		}
	}
}
