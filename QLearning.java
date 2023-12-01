import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.awt.Graphics;
import java.io.*;
public class QLearning implements Entity {
    static final int[] criticArchitecture = {5, 64, 64, 1};
    static final int[] actorArchitecture = {5, 64, 64, 1};
    static final double discountFactor = .99;
    Player player;
    NeuralNetwork actor, critic;
    boolean died;
    Enviroment env;

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
        if (fromFile != null) {
            loadModel(fromFile);
        } else {
            actor = new NeuralNetwork(actorArchitecture);
            critic = new NeuralNetwork(criticArchitecture);
        }
    }
    
    double[] computeLoss(ActionStatePair actionStatePair) {
        /*Matrix bestNextState = actionStatePair.getNextActionStateInput(getBestAction(actionStatePair.nextState));
        double targetVal = actionStatePair.reward + discountFactor * targetBrain.forward(bestNextState).data[0][0];
        Matrix curPred = brain.forward(actionStatePair.getCurActionStateInput());
        //System.out.println("predicted reward: " + curPred.data[0][0] + " " + ", actual reward: " + reward);
        brain.backward(curPred.subtract(targetVal), learningRate, .9, .999, epoch + 1);
        return Math.pow(curPred.data[0][0] - targetVal, 2);*/
      
        double criticOutput = critic.forward(actionStatePair.curState.toMatrix()).data[0][0];
        double advantage = actionStatePair.reward - criticOutput;
        critic.backward(new Matrix(-advantage *.5));
        
        double actorOutput = actor.forward(actionStatePair.curState.toMatrix()).data[0][0];
        double actionProb = NeuralNetwork.sigmoid(actorOutput);
        actor.backward(new Matrix(-advantage * (1. - actionProb)));
        
        return new double[]{advantage*advantage *.5, -NeuralNetwork.stablizeLog(actionProb)*advantage};
    }
    int getBestAction(State state) {
        double actorOutput = NeuralNetwork.sigmoid(actor.forward(state.toMatrix()).data[0][0]);
        return actorOutput > .5? 1 : 0;
    }
    int getStochasticAction(State state, Random rand) {
        double actorOutput = NeuralNetwork.sigmoid(actor.forward(state.toMatrix()).data[0][0]);
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

    void optimize(double initActorLearningRate, double initCriticLearningRate, int maxEpochs, int batchSize, int verboseFreq) {
        System.out.println("Q-function training is starting");

        //System.out.println("posBuffer size: " + posBuffer.size());
        //System.out.println("negBuffer size: " + negBuffer.size());
        //final double epsStart = 0.9, epsEnd = 0.05, epsDecay = 5000.;
        //final double tau = 0.03;
        final double lrDecayRate = 0.95, lrDecayStep = 10000.;

        Random rand = new Random();

        //double eps = epsStart;
        int maxScore = 0;
        long maxDistSurvived = 0;
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            //System.out.println("posBuffer size: " + posBuffer.size());
            //System.out.println("negBuffer size: " + negBuffer.size());
            //eps = epsEnd + (epsStart - epsEnd) * Math.exp(-1.*epoch / epsDecay);
            //eps *= .9999;
            //if (epoch % 100 == 0) System.out.println("epoch " + epoch + ": " + eps);
            List<ActionStatePair> episode = new ArrayList<>();
            player.reset();
            env.reset();
            
            while (player.score <= 100000) {
                State curState = new State(player, env);
                
                if (player.crash(env)) {
                    episode.add(new ActionStatePair(curState, null, 0, -curState.penalty*100)); 
                    break;
                }
                int action = getStochasticAction(curState, rand);
                if (action != 0) player.tap();
                player.update();
                env.update();
                
                State nextState = new State(player, env);
                episode.add(new ActionStatePair(curState, nextState, action, 1.));
            }

            for (int j = episode.size() - 2; j >= 0; j--) 
                episode.get(j).reward += discountFactor * episode.get(j + 1).reward;

            double actorLoss = .0, criticloss = .0;
            for (int i = 0; i < episode.size(); i++) {
                double[] curLosses = computeLoss(episode.get(i));
                criticloss += curLosses[0];
                actorLoss += curLosses[1];
            }
            criticloss /= episode.size();
            actorLoss /= episode.size();

            double curLrDecay =  Math.pow(lrDecayRate, (double)epoch / lrDecayStep);
            double curActorLearningRate = initActorLearningRate * curLrDecay;
            double curCriticLearningRate = initCriticLearningRate * curLrDecay;
            critic.step(curCriticLearningRate, .9, .999, epoch, 1);
            actor.step(curActorLearningRate, .9, .999, epoch, 1);

            evaluate();
            if (verboseFreq > 0 && epoch % verboseFreq == 0) {
                System.out.println("[Epoch " + epoch + "/" + maxEpochs + "] " + "   critic loss: " + criticloss + ", actor loss: " + actorLoss  + ", score: " + player.score + ", distance survived: " + player.distSurvived);
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
