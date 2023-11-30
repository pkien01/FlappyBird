import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.awt.Graphics;
public class QLearning implements Entity {
    static final int[] architecture = {6, 12, 24, 12, 1};
    static final double discountFactor = .99;
    Player player;
    NeuralNetwork brain;
    Enviroment env;
    QLearning(Enviroment env, String paramFile) {
        this.env = env;
        player = new Player();
        brain = paramFile != null?  NeuralNetwork.load(paramFile) : new NeuralNetwork(architecture);
    }

    double step(ActionStatePair actionStatePair, NeuralNetwork targetBrain, double learningRate, int epoch, int batchSize) {
        double targetVal = actionStatePair.reward;
        if (actionStatePair.nextState != null) {
            Matrix bestNextState = actionStatePair.getNextActionStateInput(getBestAction(actionStatePair.nextState));
            targetVal += discountFactor * targetBrain.forward(bestNextState).data[0][0];
        }
        //Matrix targetVal = new Matrix(actionStatePair.reward);
        Matrix curPred = brain.forward(actionStatePair.getCurActionStateInput());
        brain.backward(curPred.subtract(targetVal).divide(batchSize), learningRate, .9, .999, epoch + 1);
        return Math.pow(curPred.data[0][0] - targetVal, 2) / batchSize;
    }
    int getBestAction(State state) {
        double noTapReward = brain.forward(ActionStatePair.getStateActionInput(state, 0)).data[0][0];
        double tapReward = brain.forward(ActionStatePair.getStateActionInput(state, 1)).data[0][0];
        //System.out.println("noTapReward: " + noTapReward + ", tapReward: " + tapReward);
        return noTapReward >= tapReward? 0 : 1;
    }

    void evaluate() {
        player.reset();
        env.reset();
        while (player.score <= 100000) {
            if (player.crash(env)) return;
            State curState = new State(player, env);
            if (getBestAction(curState) == 1) player.tap();

            player.update();
            env.update();
        }
    }

    void optimize(double initLearningRate, int maxEpochs, int batchSize, int verboseFreq) {
        System.out.println("Q-function training is starting");

        //System.out.println("posBuffer size: " + posBuffer.size());
        //System.out.println("negBuffer size: " + negBuffer.size());
        final double epsStart = 0.9, epsEnd = 0.05, epsDecay = 1000.;
        final double tau = 0.01;
        final double lrDecayRate = 0.96, lrDecayStep = 10000.;
        final int maxMemorySize = 1000000;

        Random rand = new Random();
        CircularBuffer<ActionStatePair> memory = new CircularBuffer<>(maxMemorySize);

        int maxScore = 0;
        long maxDistSurvived = 0;
        NeuralNetwork target = new NeuralNetwork(brain);
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            //System.out.println("posBuffer size: " + posBuffer.size());
            //System.out.println("negBuffer size: " + negBuffer.size());
            double eps = epsEnd + (epsStart - epsEnd) * Math.exp(-1.*epoch / epsDecay);
            double curLearningRate = initLearningRate * Math.pow(lrDecayRate, (double)epoch / lrDecayStep);
            player.reset();
            env.reset();
            
            double loss = 0.0;
            while (player.score <= 100000) {
                State curState = new State(player, env);
                
                int action;
                double curEps = eps;
                if (player.distSurvived > maxDistSurvived) curEps = Math.min(epsStart, eps*1.25);
                if (rand.nextDouble() < eps) action = rand.nextBoolean()? 1 : 0;
                else action = getBestAction(curState);
                if (action == 1) player.tap();

                player.update();
                env.update();
                
                if (player.crash(env)) {
                    memory.add(new ActionStatePair(curState, null, action, -curState.penalty*100.)); 
                    break;
                }
                State nextState = new State(player, env);
                memory.add(new ActionStatePair(curState, nextState, action, 1.));

                for (int i = 0; i < batchSize; i++)
                    loss += step(memory.get(rand.nextInt(memory.size())), target, curLearningRate, epoch, batchSize);

                for (int i = 0; i < target.layers.length; i++) {
                    target.layers[i].weight.add(brain.layers[i].weight, 1. - tau);
                    target.layers[i].bias.add(brain.layers[i].bias, 1. - tau);
                }
            }

            loss /= player.distSurvived;
            maxDistSurvived = Math.max(maxDistSurvived, player.distSurvived);
            evaluate();
            if (verboseFreq > 0 && epoch % verboseFreq == 0) {
                System.out.println("[Epoch " + epoch + "/" + maxEpochs + "] " + "Q loss: " + loss + ", score: " + player.score + ", distance survived: " + player.distSurvived);
            }   

            if (player.score > Math.max(2, maxScore)) {
                brain.save(String.format(Main.Q_LEARNING_FILE_FORMAT, maxScore,  epoch));
                brain.save(Main.Q_LEARNING_FILE_DEFAULT);
                maxScore = player.score;
            }
        }
        System.out.println("Q-function training finished");
    }
    public void update() {
        if (player.crash(env)) {
            player.reset();
            env.reset();
            return;
        }
        State curState = new State(player, env);
        if (getBestAction(curState) == 1) player.tap();
        player.update();
    }
    public void draw(Graphics g) {
        player.draw(g);
    }
}
