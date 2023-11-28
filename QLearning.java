import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.awt.Graphics;
import java.util.Collections;
import java.util.Deque;
public class QLearning implements Entity {
    static final int[] architecture = {6, 12, 24, 12, 1};
    static final double discountFactor = .9;
    Player player;
    NeuralNetwork brain;
    boolean died;
    Enviroment env;
    int numGenerations, maxScore;
    long distSurvived, maxDistSurvived;
    QLearning(Enviroment env, String paramFile) {
        this.env = env;
        player = new Player();
        brain = paramFile != null?  NeuralNetwork.load(paramFile) : new NeuralNetwork(architecture);
        maxScore = 0;
        numGenerations = 0;
        distSurvived = 0;
        maxDistSurvived = 0;
    }

    double step(ActionStatePair actionStatePair, double learningRate, int epoch, NeuralNetwork targetBrain) {
        Matrix bestNextState = actionStatePair.getNextActionStateInput(getBestAction(actionStatePair.nextState));
        double targetVal = actionStatePair.reward + discountFactor * targetBrain.forward(bestNextState).data[0][0];
        Matrix curPred = brain.forward(actionStatePair.getCurActionStateInput());
        //System.out.println("predicted reward: " + curPred.data[0][0] + " " + ", actual reward: " + reward);
        brain.backward(curPred.subtract(targetVal), learningRate, .9, .999, epoch + 1);
        return Math.pow(curPred.data[0][0] - targetVal, 2);
    }
    int getBestAction(State state) {
        double noTapReward = brain.forward(ActionStatePair.getStateActionInput(state, 0)).data[0][0];
        double tapReward = brain.forward(ActionStatePair.getStateActionInput(state, 1)).data[0][0];
        //System.out.println("noTapReward: " + noTapReward + ", tapReward: " + tapReward);
        return noTapReward >= tapReward? 0 : 1;
    }

    static class CircularQueue<T> {
        List<T> buffer;
        int capacity;
        int front, rear;
        CircularQueue(int capacity) {
            this.capacity = capacity;
            buffer = new ArrayList<>();
            front = -1; rear = -1;
        }
        int size() {
            return buffer.size();
        }
        void add(T elem) {
            if (front == -1 && rear == -1) {
                front = 0; rear = 0;
            }
            else {
                rear = (rear + 1) % capacity;
                if (rear == front) front++;
            }
            while (rear >=buffer.size()) buffer.add(null);
            buffer.set(rear, elem);
        }
        T get(int idx) {
            assert (idx >= 0 && idx < buffer.size());
            return buffer.get((front + idx) % capacity);
        }
    }

    void optimize(double initLearningRate, int maxEpochs, int batchSize, int verboseFreq) {
        System.out.println("Q-function training is starting");

        //System.out.println("posBuffer size: " + posBuffer.size());
        //System.out.println("negBuffer size: " + negBuffer.size());
        final double epsStart = 0.9, epsEnd = 0.05, epsDecay = 1000.;
        final double tau = 0.05;
        final int maxMemorySize = 1000000;

        Random rand = new Random();
        CircularQueue<ActionStatePair> memory = new CircularQueue<>(maxMemorySize);

        double eps = epsStart;
        NeuralNetwork target = new NeuralNetwork(brain);
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            eps = epsEnd + (epsStart - epsEnd) * Math.exp(-1.*epoch / epsDecay);

            double loss = 0.0;
            State prevState = null;
            int prevAction = -1;
            while (player.score <= 100000) {
                State curState = new State(player, env);
                if (prevState != null) {
                    if (player.crash(env)) {
                        //if (curState.penalty < 0.99 || curState.penalty > 1.01) System.out.println("penalty: " + curState.penalty);
                        memory.add(new ActionStatePair(prevState, curState, prevAction,  -curState.penalty));
                        break;
                    } else {
                        memory.add(new ActionStatePair(prevState, curState, prevAction,  1));
                    }

                    for (int i = 0; i < batchSize; i++) {
                        double curLearningRate = initLearningRate;
                        loss += step(memory.get(rand.nextInt(memory.size())), curLearningRate, epoch, target) / batchSize;
                    }

                    for (int i = 0; i < target.layers.length; i++) {
                        target.layers[i].weight.add(brain.layers[i].weight, 1. - tau);
                        target.layers[i].bias.add(brain.layers[i].bias, 1. - tau);
                    }   

                }  

                int pred;
                if (rand.nextDouble() < eps) pred = rand.nextBoolean()? 1 : 0;
                else pred = getBestAction(curState);
                if (pred == 1) player.tap();

                prevState = curState;
                prevAction = pred;

                distSurvived++;
                player.update();
                env.update();
            }

            maxDistSurvived = Math.max(maxDistSurvived, distSurvived);
            maxScore = Math.max(maxScore, player.score);

            if (player.score >= Math.max(2, maxScore*2)) {
                brain.save(String.format(Main.Q_LEARNING_FILE_FORMAT, maxScore,  epoch));
                brain.save(Main.Q_LEARNING_FILE_DEFAULT);
            }
            if (verboseFreq > 0 && epoch % verboseFreq == 0) {
                System.out.println("[Epoch " + epoch + "/" + maxEpochs + "] " + "Q loss: " + loss + ", score: " + maxScore + ", distance survived: " + maxDistSurvived);
            }
            
            distSurvived  = 0;
            player.reset();
            env.reset();
        }
        System.out.println("Q-function training finished");
    }

    public void update() {
        if (died) {
            maxScore = Math.max(maxScore, player.score);
            numGenerations++;
            env.reset();
            player.reset();
            died = false;
            //posBuffer = posBuffer.subList(0, Math.min(1000, posBuffer.size()));
            //negBuffer = negBuffer.subList(0, Math.min(1000, negBuffer.size()));
            distSurvived = 0;
            return;
        }
        died = player.crash(env);
        State curState = new State(player, env);
        if (died) System.out.println("penalty = " + curState.penalty);
        if (getBestAction(curState) == 1) player.tap();
        player.update();
        distSurvived++;
    }
    public void draw(Graphics g) {
        player.draw(g);
    }
}
