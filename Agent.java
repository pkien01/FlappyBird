import java.awt.Graphics;
import java.io.Serializable;
import java.util.*;

class Agent implements Entity, Serializable {
    static final int[] architecture = {5, 15, 25, 15, 1};
    NeuralNetwork brain;
    Player player;
    Random rand;
    boolean died;
    State input;
    long distSurvived;
    int epoch;

    Agent(NeuralNetwork net) {
        player = new Player();
        rand = new Random();
        died = false;
        distSurvived = 0;
        epoch = 1;
        brain = net;
    }   
    Agent() {
        this(new NeuralNetwork(architecture));
    }
    Agent(Agent other) {
        this(new NeuralNetwork(other.brain));
    }
    public void draw(Graphics g) {
        player.draw(g);
    }
    public Agent getInput(Enviroment env) {
        if (player.crash(env)) {
            died = true;
            return this;
        }
        input = new State(player, env);
        return this;
    }
    public void update() {
        //if (distSurvived % 100 == 0) System.out.println("input = " + input);
        distSurvived++;
        assert input.size() == architecture[0];
        Matrix lastLayer = brain.forward(input.toMatrix()); // need a 5x1 matrix here
        assert lastLayer.n == architecture[architecture.length - 1];
        assert lastLayer.m == 1;
        int pred = NeuralNetwork.sigmoid(lastLayer.data[0][0]) > 0.7 ? 1 : 0;
        if (pred == 1) player.tap();
        player.update();
    }
    Agent mutate() {
        for (int i = 0; i < brain.layers.length; i++) {
            assert brain.layers[i] != null;
            brain.layers[i].weight.applyInPlace(
                elem -> {
                    return rand.nextDouble() < .1? elem + rand.nextGaussian() : elem;
                }
            );
            
            brain.layers[i].bias.applyInPlace(
                elem -> {
                    return rand.nextDouble() < .1? elem + rand.nextGaussian() : elem;
                }
            );
        }
        return this;
    }

    static Matrix crossOver(Matrix lhs, Matrix rhs, int cutPoint) {
        assert lhs.n == rhs.n;
        assert lhs.m == rhs.m;
        assert cutPoint >= 0;
        assert cutPoint < lhs.n;
        Matrix res = new Matrix(lhs.n, lhs.m);
        for (int i = 0; i < cutPoint; i++)
            for (int j = 0; j < res.m; j++) res.data[i][j] = lhs.data[i][j];
        
        for (int i = cutPoint; i < res.n; i++)
            for (int j = 0; j < res.m; j++) res.data[i][j] = rhs.data[i][j];
        
        return res;
    }
    Agent mate(Agent other) {
        //Random rand = new Random(1225);
        Agent res = new Agent();
        for (int i = 0; i < brain.layers.length; i++) {
            int randCut =  rand.nextInt(brain.layers[i].out_size);
            res.brain.layers[i].weight = crossOver(brain.layers[i].weight, other.brain.layers[i].weight, randCut);
            res.brain.layers[i].bias = crossOver(brain.layers[i].bias, other.brain.layers[i].bias, randCut);
        }
        return res;
    }
    long fitness() {
        return distSurvived;
    }
    boolean isAlive() {
        return !died;
    }
}
