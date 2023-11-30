import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.awt.Graphics;
import java.io.*;
public class QLearning implements Entity {
    static final int[] criticArchitecture = {6, 128, 1};
    static final int[] actorArchitecture = {5, 128, 1};
    static final double discountFactor = .99;
    Player player;
    NeuralNetwork actor, critic;
    boolean died;
    Enviroment env;
    Random rand;

    void loadModel(String fileName) {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(fileName))) {
            actor = NeuralNetwork.load(in);
            critic = NeuralNetwork.load(in);
        } catch (IOException e) {
            throw new RuntimeException("Cannot open the file: " + fileName, e);
        }
    }
    void saveModel(String fileName) {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(fileName))) {
            actor.save(out);
            critic.save(out);
        } catch (IOException e) {
            throw new RuntimeException("Cannot open the file: " + fileName, e);
        }
    }
    QLearning(Enviroment env, String fromFile) {
        this.env = env;
        player = new Player();
        rand = new Random();
        if (fromFile != null) {
            loadModel(fromFile);
        } else {
            actor = new NeuralNetwork(actorArchitecture);
            critic = new NeuralNetwork(criticArchitecture);
        }
    }
    
    double[] step(ActionStatePair actionStatePair, NeuralNetwork targetNet, double actorLearningRate, double criticLearningRate, int epoch, int batchSize) {
        /*Matrix bestNextState = actionStatePair.getNextActionStateInput(getBestAction(actionStatePair.nextState));
        double targetVal = actionStatePair.reward + discountFactor * targetBrain.forward(bestNextState).data[0][0];
        Matrix curPred = brain.forward(actionStatePair.getCurActionStateInput());
        //System.out.println("predicted reward: " + curPred.data[0][0] + " " + ", actual reward: " + reward);
        brain.backward(curPred.subtract(targetVal), learningRate, .9, .999, epoch + 1);
        return Math.pow(curPred.data[0][0] - targetVal, 2);*/
      
        double advantage = actionStatePair.reward;
        if (actionStatePair.nextState != null) 
            advantage += discountFactor*targetNet.forward(actionStatePair.getNextActionStateInput(getStochasticAction(actionStatePair.nextState))).data[0][0];

        advantage -= critic.forward(actionStatePair.getCurActionStateInput()).data[0][0];
        critic.backward(new Matrix(-advantage / 2 / batchSize), criticLearningRate, .9, .999, epoch+1);
        
        double actorOutput = actor.forward(actionStatePair.curState.toMatrix()).data[0][0];
        double actionProb = NeuralNetwork.sigmoid(actorOutput);
        actor.backward(new Matrix(-advantage * (1. - actionProb) / batchSize), actorLearningRate, .9, .999, epoch + 1);
        
        return new double[]{advantage*advantage / batchSize, -NeuralNetwork.stablizeLog(actionProb)*advantage / batchSize};
    }
    int getBestAction(State state) {
        double actorOutput = NeuralNetwork.sigmoid(actor.forward(state.toMatrix()).data[0][0]);
        return actorOutput > .5? 1 : 0;
    }
    int getStochasticAction(State state) {
        double actorOutput = NeuralNetwork.sigmoid(actor.forward(state.toMatrix()).data[0][0]);
        return rand.nextDouble() < actorOutput? 1 : 0;
    } 

    void evaluate() {
        Random rand = new Random();
        player.reset();
        env.reset();
        while (player.score <= 100000) {
            State curState = new State(player, env);
            if (player.crash(env)) return;
            if (getStochasticAction(curState) == 1) player.tap();

            player.update();
            env.update();
        }
    }

    void optimize(double initActorLearningRate, double initCriticLearningRate, int maxEpochs, int batchSize, int verboseFreq) {
        System.out.println("Q-function training is starting");

        //System.out.println("posBuffer size: " + posBuffer.size());
        //System.out.println("negBuffer size: " + negBuffer.size());
        //final double epsStart = 0.9, epsEnd = 0.05, epsDecay = 5000.;
        final double tau = 0.03;
        final double lrDecayRate = 0.95, lrDecayStep = 100000.;
        final int maxMemorySize = 1000000;

        CircularBuffer<ActionStatePair> memory = new CircularBuffer<>(maxMemorySize);

        //double eps = epsStart;
        int maxScore = 0;
        long maxDistSurvived = 0;
        NeuralNetwork target = new NeuralNetwork(critic);
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            //System.out.println("posBuffer size: " + posBuffer.size());
            //System.out.println("negBuffer size: " + negBuffer.size());
            //eps = epsEnd + (epsStart - epsEnd) * Math.exp(-1.*epoch / epsDecay);
            //eps *= .9999;
            //if (epoch % 100 == 0) System.out.println("epoch " + epoch + ": " + eps);
            player.reset();
            env.reset();

            double curLrDecay =  Math.pow(lrDecayRate, (double)epoch / lrDecayStep);
            double curActorLearningRate = initActorLearningRate * curLrDecay * Math.pow(discountFactor, epoch);
            double curCriticLearningRate = initCriticLearningRate * curLrDecay;
            
            while (player.score <= 100000) {
                State curState = new State(player, env);
                
                if (player.crash(env)) {
                    memory.add(new ActionStatePair(curState, null, 0, -curState.penalty*100)); 
                    break;
                }
                int action = getStochasticAction(curState);
                if (action == 1) player.tap();
                player.update();
                env.update();
                
                State nextState = new State(player, env);
                memory.add(new ActionStatePair(curState, nextState, action, 1.));
            }
            //criticLoss /= player.distSurvived;
            //actorLoss /= player.distSurvived;
            double actorLoss = 0.0, criticLoss = 0.0;
            for (int i = 0; i < batchSize; i++) {
                double[] losses = step(memory.get(rand.nextInt(memory.size())), target, curActorLearningRate, curCriticLearningRate, epoch, batchSize);
                criticLoss += losses[0];
                actorLoss += losses[1];
            }
            for (int i = 0; i < target.layers.length; i++) {
                target.layers[i].weight.add(critic.layers[i].weight, 1. - tau);
                target.layers[i].bias.add(critic.layers[i].bias, 1. - tau);
            }
            evaluate();
            if (verboseFreq > 0 && epoch % verboseFreq == 0) {
                System.out.println("[Epoch " + epoch + "/" + maxEpochs + "] " + "   critic loss: " + criticLoss + ", actor loss: " + actorLoss  + ", score: " + player.score + ", distance survived: " + player.distSurvived);
            }

            if (player.score >= Math.max(2, maxScore*2)) {
                saveModel(String.format(Main.Q_LEARNING_FILE_FORMAT, maxScore,  epoch));
                saveModel(Main.Q_LEARNING_FILE_DEFAULT);
                maxScore = player.score;
                maxDistSurvived = player.distSurvived;
            }
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
