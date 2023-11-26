import java.io.*;
import java.util.Random;

public class NeuralNetwork implements Serializable {
	static class Layer implements Serializable {
        Matrix weight, bias;
        int in_size, out_size;
        Matrix input, output;
        Matrix moment1Weight, moment1Bias;
        Matrix moment2Weight, moment2Bias;
        Layer(int in_size, int out_size) {
            weight = new Matrix(out_size, in_size);
            bias = new Matrix(out_size, 1);
            this.in_size = in_size;
            this.out_size = out_size;
            moment1Weight = new Matrix(out_size, in_size, 0.);
            moment2Weight = new Matrix(out_size, in_size, 0.);
            moment1Bias = new Matrix(out_size, 1, 0);
            moment2Bias = new Matrix(out_size, 1, 0);
        }
        Layer(Layer other) {
            weight = new Matrix(other.weight);
            bias = new Matrix(other.bias);
            this.in_size = other.in_size;
            this.out_size = other.out_size;
        }
        Matrix linear(Matrix input) {
            this.input = input;
            output = weight.multiply(input).add(bias);
            return output;
        }
        void initializeRandom(Random rng) {
            double std = Math.sqrt(2./ in_size);
            weight.fill(() -> rng.nextGaussian() * std);
            bias.fill(() -> rng.nextGaussian() * std);
        }
        void computeGrad(Matrix linGrad, double beta1, double beta2) {
            Matrix weightGrad = linGrad.multiply(input.transpose());
            moment1Weight = moment1Weight.add(weightGrad, beta1);
            moment2Weight = moment2Weight.add(weightGrad.multiplyEntrywise(weightGrad), beta2);

            Matrix biasGrad = new Matrix(linGrad);
            moment1Bias = moment1Bias.add(biasGrad, beta1);
            moment2Bias = moment2Bias.add(biasGrad.multiplyEntrywise(biasGrad), beta2);
        }
        void step(double learningRate, double beta1, double beta2, int epoch) {
            double correct1 = 1. + Math.pow(beta1, epoch), correct2 = 1. + Math.pow(beta2, epoch); 
            Matrix correctedMoment1Weight = moment1Weight.divide(correct1);
            Matrix correctedMoment1Bias = moment1Bias.divide(correct1);
            
            Matrix correctedMoment2Weight = moment2Weight.divide(correct2);
            Matrix correctedMoment2Bias = moment2Bias.divide(correct2);

            correctedMoment2Weight.applyInPlace(x -> Math.sqrt(x) + 1e-8);
            correctedMoment2Bias.applyInPlace(x -> Math.sqrt(x) + 1e-8);
            
            Matrix weightUpdate = correctedMoment1Weight.divide(correctedMoment2Weight);
            Matrix biasUpdate = correctedMoment1Bias.divide(correctedMoment2Bias);
            weight.subtractInPlace(weightUpdate.multiply(learningRate));
            bias.subtractInPlace(biasUpdate.multiply(learningRate));
        }
    }
    Layer[] layers;
    long epoch;
    NeuralNetwork(int[] sizes) {
        layers = new Layer[sizes.length - 1];
        Random rand = new Random();
        for (int i = 0; i < sizes.length - 1; i++) {
            layers[i] = new Layer(sizes[i], sizes[i + 1]);
            layers[i].initializeRandom(rand);
        }
        epoch = 0;
    }
    NeuralNetwork(NeuralNetwork other) {
        layers = new Layer[other.layers.length];
        for (int i = 0; i < other.layers.length; i++) {
            layers[i] = new Layer(other.layers[i]);
        }
        epoch = 0;
    }
    Matrix forward(Matrix input) {
        assert input.n == layers[0].in_size;
        Matrix res = input;
        for (int i = 0; i < layers.length; i++) {
            res = layers[i].linear(res);
            assert res.n == layers[i].out_size;
            if (i != layers.length - 1) res.applyInPlace(x -> Math.max(x, 0.));
        }
        return res;
    }
    void backward(Matrix lastGrad, double learningRate, double beta1, double beta2, int epoch) {
        Matrix linGrad = lastGrad;
        layers[layers.length - 1].computeGrad(linGrad, beta1, beta2);
        layers[layers.length - 1].step(learningRate, beta1, beta2, epoch);
        for (int i = layers.length - 2; i >= 0; i--) {
            Matrix actGrad = layers[i+1].weight.transpose().multiply(linGrad);
            linGrad = actGrad.multiplyEntrywise(layers[i].output.apply(x -> x > 0? 1. : 0.));
            layers[i].computeGrad(linGrad, beta1, beta2);
            layers[i].step(learningRate, beta1, beta2, epoch);
        }
    }
    void save(String fileName) {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(fileName))) {
            out.writeObject(this);
        } catch (IOException e) {
            throw new RuntimeException("Error saving neural network to file: " + fileName, e);
        }
    }
    static NeuralNetwork load(String fileName) {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(fileName))) {
            return (NeuralNetwork)in.readObject();
        } catch (IOException | ClassNotFoundException e) {
            throw new RuntimeException("Error loading neural network from file: " + fileName, e);
        }
    }

    static double sigmoid(double x) {
        return 1. / (1. + Math.exp(-x));
    }
    static double stablizeLog(double val) {
        val = Math.max(val, 1e-9);
        val = Math.min(val, 1. - 1e-9);
        return Math.log(val);
    }
}