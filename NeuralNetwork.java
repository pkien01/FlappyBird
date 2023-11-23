import java.util.Random;
public class NeuralNetwork {
	static class Layer {
        Matrix weight, bias;
        int in_size, out_size;
        Layer(int in_size, int out_size) {
            weight = new Matrix(out_size, in_size);
            bias = new Matrix(out_size, 1);
            this.in_size = in_size;
            this.out_size = out_size;
        }
        Matrix linear(Matrix input) {
            return weight.multiply(input).add(bias);
        }
        void initializeRandom(Random rng) {
            double std = Math.sqrt(2./ in_size);
            weight.fill(() -> rng.nextGaussian() * std);
            bias.fill(() -> rng.nextGaussian() * std);
        }
    }
    Layer[] layers;
    NeuralNetwork(int[] sizes) {
        layers = new Layer[sizes.length - 1];
        for (int i = 0; i < sizes.length - 1; i++)
            layers[i] = new Layer(sizes[i], sizes[i + 1]);
    }
    void initializeRandom() {
        Random rand = new Random();
        for (int i = 0; i < layers.length; i++)
            layers[i].initializeRandom(rand);
    }
    Matrix forward(Matrix input) {
        Matrix res = input;
        for (int i = 0; i < layers.length; i++) {
            res = layers[i].linear(res).apply(x -> Math.max(x, 0.));
        }
        return res;
    }
}