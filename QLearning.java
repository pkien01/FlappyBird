import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.awt.Graphics;
import java.io.*;
import java.util.Collections;
public class QLearning implements Entity {
    static final int[] bodyArchitecture = {5, 15, 25};
    static final int[] criticArchitecture = {25, 15, 1};
    static final int[] actorArchitecture = {25, 15, 1};
    static final double discountFactor = .9;
    Player player;
    NeuralNetwork body, actor, critic;
    boolean died;
    Enviroment env;

    void loadModel(String fileName) {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(fileName))) {
            body = NeuralNetwork.load(in);
            actor = NeuralNetwork.load(in);
            critic = NeuralNetwork.load(in);
        } catch (IOException e) {
            throw new RuntimeException("Cannot open the file: " + fileName, e);
        }
    }
    void saveModel(String fileName) {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(fileName))) {
            body.save(out);
            actor.save(out);
            critic.save(out);
        } catch (IOException e) {
            throw new RuntimeException("Cannot open the file: " + fileName, e);
        }
    }
    QLearning(Enviroment env, String fromFile) {
        this.env = env;
        player = new Player();
        if (fromFile != null) {
            loadModel(fromFile);
        } else {
            body = new NeuralNetwork(bodyArchitecture);
            actor = new NeuralNetwork(actorArchitecture);
            critic = new NeuralNetwork(criticArchitecture);
        }
    }

    double step(ActionStatePair actionStatePair, double learningRate, int epoch) {
        /*Matrix bestNextState = actionStatePair.getNextActionStateInput(getBestAction(actionStatePair.nextState));
        double targetVal = actionStatePair.reward + discountFactor * targetBrain.forward(bestNextState).data[0][0];
        Matrix curPred = brain.forward(actionStatePair.getCurActionStateInput());
        //System.out.println("predicted reward: " + curPred.data[0][0] + " " + ", actual reward: " + reward);
        brain.backward(curPred.subtract(targetVal), learningRate, .9, .999, epoch + 1);
        return Math.pow(curPred.data[0][0] - targetVal, 2);*/

        Matrix nextFeature = body.forward(actionStatePair.nextState.toMatrix());
        double nextCriticOutput = critic.forward(nextFeature).data[0][0];
        Matrix feature = body.forward(actionStatePair.curState.toMatrix());
        
        double criticOutput = critic.forward(feature).data[0][0];
        double advantage = actionStatePair.reward + discountFactor * nextCriticOutput - criticOutput;
        Matrix criticGrad = critic.backward(new Matrix(-advantage), learningRate, .9, .999, epoch+1);
        body.backward(criticGrad, learningRate, .9, .999, epoch + 1);
        
        double actorOutput = actor.forward(feature).data[0][0];
        Matrix actorGrad = actor.backward(new Matrix(-advantage * NeuralNetwork.sigmoid(-actorOutput)), learningRate, .9, .999, epoch + 1);
        body.backward(actorGrad, learningRate, .9, .999, epoch + 1);
        
        return Math.pow(advantage, 2) - NeuralNetwork.stablizeLog(NeuralNetwork.sigmoid(actorOutput))*advantage;
    }
    int getBestAction(State state) {
        Matrix feature = body.forward(state.toMatrix());
        double actorOutput = NeuralNetwork.sigmoid(actor.forward(feature).data[0][0]);
        return actorOutput > .5? 1 : 0;
    }
    int getStochasticAction(State state, Random rand) {
        Matrix feature = body.forward(state.toMatrix());
        double actorOutput = NeuralNetwork.sigmoid(actor.forward(feature).data[0][0]);
        return rand.nextDouble() < actorOutput? 1 : 0;
    } 

    void evaluate() {
        player.reset();
        env.reset();
        while (player.score <= 100000) {
            State curState = new State(player, env);
            if (player.crash(env)) return;
            if (getBestAction(curState) == 1) player.tap();

            player.update();
            env.update();
        }
    }

    void optimize(double initLearningRate, int maxEpochs, int batchSize, int verboseFreq) {
        System.out.println("Q-function training is starting");

        //System.out.println("posBuffer size: " + posBuffer.size());
        //System.out.println("negBuffer size: " + negBuffer.size());
        final double epsStart = 0.9, epsEnd = 0.05, epsDecay = 5000.;
        final double tau = 0.03;
        final double lrDecayRate = 0.96, lrDecayStep = 10000.;

        Random rand = new Random();
        List<ActionStatePair> posMemory = new ArrayList<>();
        List<ActionStatePair> negMemory = new ArrayList<>();

        final int maxMemorySize = 1000000;

        double eps = epsStart;
        int maxScore = 0;
        long maxDistSurvived = 0;
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            //System.out.println("posBuffer size: " + posBuffer.size());
            //System.out.println("negBuffer size: " + negBuffer.size());
            eps = epsEnd + (epsStart - epsEnd) * Math.exp(-1.*epoch / epsDecay);
            //eps *= .9999;
            //if (epoch % 100 == 0) System.out.println("epoch " + epoch + ": " + eps);
            List<ActionStatePair> posBatch = new ArrayList<>();
            List<ActionStatePair> negBatch = new ArrayList<>();
            for (int i = 0; i < batchSize; i++) {
                player.reset();
                env.reset();
                State prevState = null;
                int prevAction = -1;
                while (player.score <= 100000) {
                    State curState = new State(player, env);
                    if (prevState != null) {
                        if (player.crash(env)) {
                            //if (curState.penalty < 0.99 || curState.penalty > 1.01) System.out.println("penalty: " + curState.penalty);
                            negBatch.add(new ActionStatePair(prevState, curState, prevAction,  -curState.penalty*3));
                            break;
                        } else 
                            posBatch.add(new ActionStatePair(prevState, curState, prevAction,  1));
                    }  
                    int pred = getStochasticAction(curState, rand);

                    prevState = curState;
                    prevAction = pred;

                    player.update();
                    env.update();
                }
            }
        
            Collections.shuffle(posBatch);
            Collections.shuffle(negBatch);

            posMemory.addAll(posBatch.subList(0, batchSize));
            negMemory.addAll(negBatch.subList(0, batchSize));

            double loss = 0.0;
            double curLearningRate = initLearningRate * Math.pow(lrDecayRate, (double)epoch / lrDecayStep);
            
            for (int i = 0; i < batchSize; i++) {
                if (!posMemory.isEmpty()) loss += step(posMemory.get(rand.nextInt(posMemory.size())), curLearningRate, epoch) / batchSize/ 2;
                if (!negMemory.isEmpty()) loss += step(negMemory.get(rand.nextInt(negMemory.size())), curLearningRate, epoch) / batchSize / 2;
            }
           /*  for (int i = 0; i < batchSize; i++) {
                double curLearningRate = initLearningRate / Math.sqrt(epoch + 1);
                if (!posBatch.isEmpty()) loss += step(posBatch.get(i % posBatch.size()), curLearningRate) / batchSize;
                if (!negBatch.isEmpty()) loss += step(negBatch.get(i % negBatch.size()), curLearningRate) / batchSize;
            }*/

            evaluate();
            if (verboseFreq > 0 && epoch % verboseFreq == 0) {
                System.out.println("[Epoch " + epoch + "/" + maxEpochs + "] " + "Q loss: " + loss + ", score: " + player.score + ", distance survived: " + player.distSurvived);
            }

            if (player.score >= Math.max(2, maxScore*2)) {
                saveModel(String.format(Main.Q_LEARNING_FILE_FORMAT, maxScore,  epoch));
                saveModel(Main.Q_LEARNING_FILE_DEFAULT);
                maxScore = player.score;
                maxDistSurvived = player.distSurvived;
            }

            if (posMemory.size() > maxMemorySize) posMemory.subList(0, posMemory.size() - maxMemorySize).clear();
            if (negMemory.size() > maxMemorySize) negMemory.subList(0, negMemory.size() - maxMemorySize).clear();
        }
        System.out.println("Q-function training finished");
    }
    public void update() {
        if (died) {
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
