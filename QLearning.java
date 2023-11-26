import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.awt.Graphics;

public class QLearning implements Entity {
    static final int[] architecture = {6, 12, 24, 12, 1};
    static final double discountFactor = .7;
    Player player;
    NeuralNetwork brain;
    boolean died;
    Enviroment env;
    int numGenerations, maxScore;
    QLearning(Enviroment env, String paramFile) {
        this.env = env;
        player = new Player();
        brain = paramFile != null?  NeuralNetwork.load(paramFile) : new NeuralNetwork(architecture);
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
    int getBestAction(State state) {
        double noTapReward = brain.forward(ActionStatePair.getStateActionInput(state, 0)).data[0][0];
        double tapReward = brain.forward(ActionStatePair.getStateActionInput(state, 1)).data[0][0];
        return noTapReward >= tapReward? 0 : 1;
    }

    void optimize(double initLearningRate, int maxEpochs, int batchSize, double runningFactor, int verboseFreq) {
        System.out.println("Q-function training is starting");

        //System.out.println("posBuffer size: " + posBuffer.size());
        //System.out.println("negBuffer size: " + negBuffer.size());
        final double epsStart = 0.9, epsEnd = 0.05, epsDecay = 1000.;

        Random rand = new Random();
        List<ActionStatePair> posBuffer = new ArrayList<>();
        List<ActionStatePair> negBuffer = new ArrayList<>();

        final int maxLen = 1000000;

        double eps = 0.5;
        int overallMaxScore = 0;
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            //System.out.println("posBuffer size: " + posBuffer.size());
            //System.out.println("negBuffer size: " + negBuffer.size());
            eps = epsEnd + (epsStart - epsEnd) * Math.exp(-1.*epoch / epsDecay);
            //eps *= .9999;
            //if (epoch % 10 == 0) System.out.println("epoch " + epoch + ": " + eps);
            for (int i = 0; i < batchSize; i++) {
                State prevState = null;
                int prevAction = -1;
                while (player.score <= 1000) {
                    State curState = new State(player, env);
                    int pred;
                    if (rand.nextDouble() < eps) pred = rand.nextBoolean()? 1 : 0;
                    else pred = getBestAction(curState);
                    //System.out.println("curState: " + curState.toMatrix() + ", prevAction: " + prevAction + ", died: " + emulator.died);
                    if (pred == 1) player.tap();
                    if (prevState != null) {
                        if (player.crash(env)) {
                            negBuffer.add(new ActionStatePair(prevState, curState, prevAction,  -5));
                            break;
                        } else 
                            posBuffer.add(new ActionStatePair(prevState, curState, prevAction,  1));
                    }  
                    prevState = curState;
                    prevAction = pred;
                    player.update();
                    env.update();
                }
                maxScore = Math.max(maxScore, player.score);
                player.reset();
                env.reset();
            }
            
            if (epoch % 1000 == 0 || maxScore >= Math.max(2, overallMaxScore*2)) {
                brain.save(String.format(Main.Q_LEARNING_FILE_FORMAT, maxScore,  epoch));
                brain.save(Main.Q_LEARNING_FILE_DEFAULT);
            }

            double loss = 0.0;
            for (int i = 0; i < batchSize; i++) {
                double curLearningRate = initLearningRate / Math.sqrt(epoch + 1);
                if (!posBuffer.isEmpty()) loss += step(posBuffer.get(rand.nextInt(posBuffer.size())), curLearningRate, runningFactor) / batchSize;
                if (!negBuffer.isEmpty()) loss += step(negBuffer.get(rand.nextInt(negBuffer.size())), curLearningRate, runningFactor) / batchSize;
            }
            if (verboseFreq > 0 && epoch % verboseFreq == 0) {
                System.out.println("[Epoch " + epoch +"] " + "Q loss: " + loss + ", max score: " + maxScore);
                maxScore = 0;
            }

            while (posBuffer.size() > maxLen) posBuffer.remove(rand.nextInt(posBuffer.size()));
            while (negBuffer.size() > maxLen) negBuffer.remove(rand.nextInt(negBuffer.size()));
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
            return;
        }
        died = player.crash(env);
        State curState = new State(player, env);
        if (died) System.out.println("penalty = " + curState.penalty);
        if (getBestAction(curState) == 1) player.tap();
        player.update();
    }
    public void draw(Graphics g) {
        player.draw(g);
    }
}
