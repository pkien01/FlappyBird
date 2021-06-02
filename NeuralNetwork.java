package flappybird;

import java.util.*;

public class NeuralNetwork {
	static class SigmoidLayer {
		static double[] forward(double[] input) {
			double[] res = new double[input.length];
			for (int i = 0; i < input.length; i++) res[i] = 1.0 / (Math.exp(-input[i]) + 1.0);
			return res;
		}
	}
	static class FCLayer {
		static Random rand = new Random();
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
			res = SigmoidLayer.forward(res);
			return res;
		}

		void mutate(double rate) {
			for (int i = 0; i < out_size; i++)
				for (int j = 0; j < in_size; j++) 
					if (Math.random() < rate) weights[i][j] += rand.nextGaussian();

			for (int i = 0; i < out_size; i++) 
				if (Math.random() < rate) bias[i] += rand.nextGaussian();
		}
		void crossOverAll(FCLayer mate) {
		
			boolean[] choice = new boolean[out_size];
			int cut = rand.nextInt(out_size);
			for (int i = 0; i < out_size; i++) choice[i] = i >= out_size;

			for (int i = 0; i < out_size; i++) {
				if (choice[i]) {
					for (int j = 0; j < in_size; j++) 
						weights[i][j] = mate.weights[i][j];
				}
			}

			for (int i = 0; i < out_size; i++)
				if (choice[i]) bias[i] = mate.bias[i];
		}
		void show() {
			System.out.println("weights:");
			for (int i = 0; i < out_size; i++) System.out.println(Arrays.toString(weights[i]));
			System.out.print("\nbias: "); System.out.println(Arrays.toString(bias));
			System.out.println();
		}
	}

	static Random rand = new Random();
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
	double[] forward(double[] input) {
		assert input.length == sizes[0];
		double[] res = input.clone();
		for (int i = 0; i < depth - 1; i++) res = layers[i].forward(res);
		return res;
	}
	void mutate(double rate) {
		for (int i = 0; i < depth - 1; i++) layers[i].mutate(rate);
	}
	NeuralNetwork crossOverAll(NeuralNetwork mate) {
		NeuralNetwork res = this.copy();
		for (int i = 0; i < depth - 1; i++) res.layers[i].crossOverAll(mate.layers[i]);
		return res;
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