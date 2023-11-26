import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.awt.Graphics;
import java.util.Collections;
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
        final double epsStart = 0.9, epsEnd = 0.01, epsDecay = 5000.;

        Random rand = new Random();
        List<ActionStatePair> posMemory = new ArrayList<>();
        List<ActionStatePair> negMemory = new ArrayList<>();

        final int maxMemorySize = 1000000;

        double eps = epsStart;
        int overallMaxScore = 0;
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            //System.out.println("posBuffer size: " + posBuffer.size());
            //System.out.println("negBuffer size: " + negBuffer.size());
            eps = epsEnd + (epsStart - epsEnd) * Math.exp(-1.*epoch / epsDecay);
            //eps *= .9999;
            //if (epoch % 100 == 0) System.out.println("epoch " + epoch + ": " + eps);
             List<ActionStatePair> posBatch = new ArrayList<>();
            List<ActionStatePair> negBatch = new ArrayList<>();
            for (int i = 0; i < batchSize; i++) {
                State prevState = null;
                int prevAction = -1;
                while (player.score <= 100000) {
                    State curState = new State(player, env);
                    int pred;
                    if (rand.nextDouble() < eps) pred = rand.nextBoolean()? 1 : 0;
                    else pred = getBestAction(curState);
                    //System.out.println("curState: " + curState.toMatrix() + ", prevAction: " + prevAction + ", died: " + emulator.died);
                    if (pred == 1) player.tap();
                    if (prevState != null) {
                        if (player.crash(env)) {
                            if (curState.penalty < 0.99 || curState.penalty > 1.01) System.out.println("penalty: " + curState.penalty);
                            negBatch.add(new ActionStatePair(prevState, curState, prevAction,  -curState.penalty*5));
                            break;
                        } else 
                            posBatch.add(new ActionStatePair(prevState, curState, prevAction,  1));
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
                overallMaxScore = maxScore;
            }
            Collections.shuffle(posBatch);
            Collections.shuffle(negBatch);

            posMemory.addAll(posBatch.subList(0, batchSize));
            negMemory.addAll(negBatch.subList(0, batchSize));

            double loss = 0.0;
            for (int i = 0; i < batchSize; i++) {
                double curLearningRate = initLearningRate / Math.sqrt(epoch + 1);
                if (!posMemory.isEmpty()) loss += step(posMemory.get(rand.nextInt(posMemory.size())), curLearningRate, runningFactor) / batchSize;
                if (!negMemory.isEmpty()) loss += step(negMemory.get(rand.nextInt(negMemory.size())), curLearningRate, runningFactor) / batchSize;
            }
            if (verboseFreq > 0 && epoch % verboseFreq == 0) {
                System.out.println("[Epoch " + epoch +"] " + "Q loss: " + loss + ", max score: " + maxScore);
                maxScore = 0;
            }

            if (posMemory.size() > maxMemorySize) posMemory.subList(0, posMemory.size() - maxMemorySize).clear();
            if (negMemory.size() > maxMemorySize) negMemory.subList(0, negMemory.size() - maxMemorySize).clear();
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