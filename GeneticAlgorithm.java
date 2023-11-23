import javax.swing.*;
import javax.swing.Timer;
import java.awt.*;
import java.awt.event.*;
import java.util.*;


public class GeneticAlgorithm extends JPanel {
    private static final int[] nnArchitecture = {5, 10, 5, 1};
    static class Agent implements Entity {
        NeuralNetwork brain;
        Player player;
        Random rand;
        boolean died;
        Agent() {
            brain = new NeuralNetwork(nnArchitecture);
            player = new Player();
            rand = new Random();
            died = false;
        }
        void initialize() {
            brain.initializeRandom();
        }
        public void draw(Graphics g) {
            player.draw(g);
        }
        public void update() {
            Matrix jumpProb = brain.forward(); // need a 5x1 matrix here

        }
        Agent mutate() {
            Agent res = new Agent();
            for (int i = 0; i < brain.layers.length; i++) {
                Matrix weightNoise = brain.layers[i].weight.sameSize();
                weightNoise.fill(() -> rand.nextGaussian());
                res.brain.layers[i].weight = brain.layers[i].weight.add(weightNoise);
                
                Matrix biasNoise = brain.layers[i].bias.sameSize();
                biasNoise.fill(() -> rand.nextGaussian());
                res.brain.layers[i].bias = brain.layers[i].bias.add(biasNoise);
            }
            return res;
        }
        Agent mate(Agent other) {
            Agent res = new Agent();
            for (int i = 0; i < brain.layers.length; i++) 
                res.brain.layers[i].weight = brain.layers[i].weight.add(other.brain.layers[i].weight, rand.nextDouble());
            return res;
        }
        double fitness() {
            return player.score;
        }
        boolean isAlive() {
            return !died;
        }
    }
    
    ArrayList<Agent> agents;
    int size;
    double keeps, mutations, combinations;
    Random rand;
    GeneticAlgorithm(int populations, double keeps, double mutations, double combinations) {
        assert keeps + mutations + combinations < 1. - 1e8; 
        this.keeps = keeps; this.mutations = mutations; this.combinations = combinations;
        size = populations;
        agents = new ArrayList<>(size);
        for (int i = 0; i < size; i++) agents.add(new Agent());
        agents.forEach(Agent::initialize);
        rand = new Random();
    }
    Agent selectAgent() {
        double totalFitness = agents.stream().mapToDouble(Agent::fitness).sum();
        double randNum = rand.nextDouble(totalFitness);
        double prefFitness = 0.;
        for (Agent agent: agents) {
            prefFitness += agent.fitness();
            if (randNum <= prefFitness) 
                return agent;
        }
        return null;
    }
    void nextGeneration() {
        agents.sort(new Comparator<Agent>() {
            @Override
            public int compare(Agent lhs, Agent rhs) {
                return Double.compare(lhs.fitness(), rhs.fitness());
            }
        }.reversed());
        int numKeeps = (int)Math.floor(keeps*size);
        int numMutations = (int)Math.floor(mutations*size);
        int numCombinations = (int)Math.floor(combinations*size);
        ArrayList<Agent> children = new ArrayList<>(numKeeps + numMutations + numCombinations);

        for (int i = 0; i < numKeeps; i++) children.add(selectAgent());
        for (int i = 0; i < numMutations; i++) children.add(selectAgent().mutate());
        for (int i = 0; i < numCombinations; i++) children.add(selectAgent().mate(selectAgent()));

        agents.clear();
    }
    void run() {
        while (true) {

        }
    }
    void draw(Graphics g) {
        agents.stream().filter(Agent::isAlive).forEach(agent -> agent.draw(g));
    }
    void update() {
        agents.stream().filter(Agent::isAlive).forEach(Agent::update);
    }
}
