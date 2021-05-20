package flappybird;

import java.util.*;

public class NeuralNetwork {
	static class FCLayer {
		static Random rand = new Random(), mut = new Random();
		double[][] weights;
		double[] bias;
		int in_size, out_size;
		FCLayer(int in_size, int out_size) {
			this.in_size = in_size; this.out_size = out_size;
			weights = new double[out_size][in_size];
			for (int i = 0; i < out_size; i++)
				for (int j = 0; j < in_size; j++) 
					weights[i][j] = rand.nextGaussian();

			bias = new double[out_size];
			for (int i = 0; i < out_size; i++) bias[i] = rand.nextGaussian();
		}
		FCLayer copy() {
			FCLayer res = new FCLayer(in_size, out_size);
			res.weights = weights.clone();
			res.bias = bias.clone();
			return res;
		}
		double[] forward(double[] input) {
			assert input.length == in_size;
			double[] res = new double[out_size];
			for (int i = 0; i < out_size; i++) {
				res[i] = bias[i];
				for (int j = 0; j < in_size; j++) res[i] += weights[i][j] * input[j];
			}
			return res;
		}

		static double randFactor() {
			return (mut.nextDouble() - 0.5) * 3 + (mut.nextDouble() - 0.5) + 1;
		}
		void mutate(double rate) {
			for (int i = 0; i < out_size; i++)
				for (int j = 0; j < in_size; j++) 
					if (mut.nextDouble() < rate) weights[i][j] *= randFactor();

			for (int i = 0; i < out_size; i++) 
				if (mut.nextDouble() < rate) bias[i] *= randFactor();
		}
		void show() {
			System.out.println("weights:");
			for (int i = 0; i < out_size; i++) System.out.println(Arrays.toString(weights[i]));
			System.out.print("\nbias: "); System.out.println(Arrays.toString(bias));
			System.out.println();
		}
	}

	static class SigmoidLayer {
		double[] forward(double[] input) {
			double[] res = new double[input.length];
			for (int i = 0; i < input.length; i++) res[i] = 1.0 / (Math.exp(-input[i]) + 1.0);
			return res;
		}
	}

	static SigmoidLayer classify = new SigmoidLayer();
	FCLayer[] layers;
	int[] sizes;
	int depth;
	NeuralNetwork(int[] sizes) {
		this.sizes = sizes.clone();
		this.depth = sizes.length;
		layers = new FCLayer[depth - 1];
		for (int i = 0; i < depth - 1; i++) layers[i] = new FCLayer(sizes[i], sizes[i + 1]);
	}
	NeuralNetwork copy() {
		NeuralNetwork res = new NeuralNetwork(sizes.clone());
		for (int i = 0; i < depth - 1; i++) res.layers[i] = layers[i].copy();
		return res;
	}
	double[] normalize(double[] input) {
		double sum = 0;
		for (int i = 0; i < input.length; i++) sum += input[i];
		double[] res = new double[input.length];
		for (int i = 0; i < input.length; i++) res[i] = input[i] / sum;
		return res;
	}
	double[] forward(double[] input) {
		assert input.length == sizes[0];
		double[] res = normalize(input);
		for (int i = 0; i < depth - 1; i++) res = layers[i].forward(res);
		return classify.forward(res);
	}
	void mutate(double rate) {
		for (int i = 0; i < depth - 1; i++) layers[i].mutate(rate);
	}
/*
	public static void main(String[] args) {
		NeuralNetwork net1 = new NeuralNetwork(new int[]{3, 10, 1}), net2 = new NeuralNetwork(new int[]{3, 10, 1});
		System.out.println(net1.forward(new double[]{0.5, 0.5, 0.5})[0]);
		//for (int i = 0; i < net1.depth - 1; i++) net1.layers[i].show();
		System.out.println(net2.forward(new double[]{0.5, 0.5, 0.5})[0]);
		//for (int i = 0; i < net2.depth - 1; i++) net2.layers[i].show();
	}*/
}