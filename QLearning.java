import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.awt.Graphics;

public class QLearning implements Entity {
    static final int[] architecture = {6, 12, 18, 12, 6, 1};
    static final double discountFactor = .5;
    List<ActionStatePair> posBuffer, negBuffer;
    Player player;
    NeuralNetwork brain;
    boolean died;
    Enviroment env;
    State prevState;
    int prevAction;
    int numGenerations, maxScore;
    QLearning() {
        player = new Player();
        brain = new NeuralNetwork(architecture);
        prevState = null;
        posBuffer = new ArrayList<>();
        negBuffer = new ArrayList<>();
        maxScore = 0;
        numGenerations = 0;
    }

    double step(ActionStatePair actionStatePair, double learningRate, double runningFactor) {
        Matrix nextPredNoTap = brain.forward(actionStatePair.getNextActionStateInput(0));
        Matrix nextPredTap = brain.forward(actionStatePair.getNextActionStateInput(1));
        double reward = (double)actionStatePair.reward + discountFactor * Math.max(nextPredNoTap.data[0][0], nextPredTap.data[0][0]);
        Matrix curPred = brain.forward(actionStatePair.getCurActionStateInput());
        //System.out.println("predicted reward: " + curPred.data[0][0] + " " + ", actual reward: " + reward);
        brain.backward(curPred.subtract(reward), runningFactor);
        brain.step(learningRate);
        return Math.pow(curPred.data[0][0] - reward, 2);
    }
    void optimize(double initLearningRate, int maxEpochs, int batchSize, double runningFactor) {
        System.out.println("Q-function training is starting");

        Collections.shuffle(posBuffer);
        Collections.shuffle(negBuffer);
        //System.out.println("posBuffer size: " + posBuffer.size());
        //System.out.println("negBuffer size: " + negBuffer.size());
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            double loss = 0.0;
            for (int i = 0; i < batchSize; i++) {
                double curLearningRate = initLearningRate;
                loss += step(posBuffer.get(i % posBuffer.size()), curLearningRate, runningFactor) / batchSize;
                loss += step(negBuffer.get(i % negBuffer.size()), curLearningRate, runningFactor) / batchSize;
            }
            if (epoch % 20 == 0) System.out.println("Q loss: " + loss);
        }
        System.out.println("Q-function training finished");
    }
    QLearning getEnvironment(Enviroment env) {
        this.env = env;
        return this;
    }
    public void update() {
        if (died) {
            optimize(1e-4, 100, 1000, .9);
            maxScore = Math.max(maxScore, player.score);
            numGenerations++;
            env.reset();
            player.reset();
            died = false;
            Collections.shuffle(posBuffer);
            Collections.shuffle(negBuffer);
            //posBuffer = posBuffer.subList(0, Math.min(1000, posBuffer.size()));
            //negBuffer = negBuffer.subList(0, Math.min(1000, negBuffer.size()));
            return;
        }
        died = player.crash(env);
        State curState = new State(player, env);
        if (died) System.out.println("penalty = " + curState.penalty);
        ActionStatePair curActionState = new ActionStatePair(prevState, curState, prevAction, died? -curState.penalty*10. : 1.);
        if (prevState != null) {
            if (died) negBuffer.add(curActionState);
            else posBuffer.add(curActionState);
        }

        double noTapReward = brain.forward(curActionState.getNextActionStateInput(0)).data[0][0];
        double tapReward = brain.forward(curActionState.getNextActionStateInput(1)).data[0][0];

        //System.out.println("no tap: " + curActionState.getNextActionStateInput(0) + " -> reward: " + noTapReward);
        //System.out.println("tap: " + curActionState.getNextActionStateInput(1) + " -> reward: " + tapReward);
        int pred = noTapReward >= tapReward? 0 : 1;

        if (pred == 1) player.tap();
        player.update();
        prevState = curState;
        prevAction = pred;
    }
    public void draw(Graphics g) {
        player.draw(g);
    }
}
