import java.util.ArrayList;
import java.util.List;
import java.util.OptionalDouble;
import java.util.PriorityQueue;
import java.util.Random;
import java.awt.Graphics;
import java.io.*;
import java.util.Comparator;
public class QLearning implements Entity {
    static final int[] criticArchitecture = {5, 256, 1};
    static final int[] actorArchitecture = {5, 256, 1};
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
    
    double[] computeLoss(ActionStatePair actionStatePair,int batchSize, double entropyFactor) {
        /*Matrix bestNextState = actionStatePair.getNextActionStateInput(getBestAction(actionStatePair.nextState));
        double targetVal = actionStatePair.reward + discountFactor * targetBrain.forward(bestNextState).data[0][0];
        Matrix curPred = brain.forward(actionStatePair.getCurActionStateInput());
        //System.out.println("predicted reward: " + curPred.data[0][0] + " " + ", actual reward: " + reward);
        brain.backward(curPred.subtract(targetVal), learningRate, .9, .999, epoch + 1);
        return Math.pow(curPred.data[0][0] - targetVal, 2);*/
      
        double criticOutput = critic.forward(actionStatePair.curState.toMatrix()).data[0][0];
        double advantage = actionStatePair.reward - criticOutput;
        critic.backward(new Matrix(-advantage / batchSize));
        
        double actorOutput = actor.forward(actionStatePair.curState.toMatrix()).data[0][0];
        double actionProb = NeuralNetwork.sigmoid(actorOutput);
        double expOutput = Math.exp(-actorOutput);
        double entropyGrad = expOutput*actionProb*actionProb*(1. - NeuralNetwork.stablizeLog(expOutput + 1.));
        actor.backward(new Matrix((-advantage * (1. - actionProb) - entropyFactor*entropyGrad) / batchSize));
        
        double entropyLoss = actionProb * NeuralNetwork.stablizeLog(actionProb);
        return new double[]{advantage*advantage *.5 / batchSize, (-NeuralNetwork.stablizeLog(actionProb)*advantage - entropyFactor*entropyLoss) / batchSize};
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

    static class EnviromentPlayerPair {
        Enviroment enviroment;
        Player player;
        EnviromentPlayerPair(Enviroment enviroment, Player player) {
            this.enviroment = new Enviroment(enviroment);
            this.player = new Player(player);
        }
    }

    void optimize(double initCriticLearningRate, double initActorLearningRate,  int maxEpochs, int batchSize, int verboseFreq) {
        System.out.println("Q-function training is starting");

        //System.out.println("posBuffer size: " + posBuffer.size());
        //System.out.println("negBuffer size: " + negBuffer.size());
        final double epsStart = .9, epsEnd = .05, epsDecay = 5000.;
        final double tau = 0.03;
        final double lrDecayRate = 0.95, lrDecayStep = 10000.;
        final int maxMemoryCapacity = 1000000;

        Random rand = new Random();

        //double eps = epsStart;
        int maxScore = 0;
        long maxDistSurvived = 0;
        CircularBuffer<EnviromentPlayerPair> memory = new CircularBuffer<>(maxMemoryCapacity);
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            critic.zeroGrad();
            actor.zeroGrad();
            
            double eps = epsEnd + (epsStart - epsEnd) * Math.exp(-1.*epoch / epsDecay);
            if (memory.size() == 0 || rand.nextDouble() < eps) {
                player.reset();
                env.reset();
            } else {
                EnviromentPlayerPair prevGame = memory.get(rand.nextInt(memory.size()));
                player = new Player(prevGame.player);
                env = new Enviroment(prevGame.enviroment);
            }
            List<ActionStatePair> episode = new ArrayList<>();
            CircularBuffer<EnviromentPlayerPair> window = new CircularBuffer<>(100);
            while (player.score <= 100000) {
                State curState = new State(player, env);
                window.add(new EnviromentPlayerPair(env, player));
                
                if (player.crash(env)) {
                    episode.add(new ActionStatePair(curState, null, 0, -Math.max(curState.penalty*100, 1.))); 
                    break;
                }
                int action = getStochasticAction(curState, rand);
                if (action == 1) player.tap();
                
                player.update();
                env.update();
                
                State nextState = new State(player, env);
                episode.add(new ActionStatePair(curState, nextState, action, 1.));
            }
            
            for (int i = (int)(.4*window.size()); i <= (int)(.92 * window.size()); i++) 
                memory.add(window.get(i));

            for (int i = episode.size() - 2; i >= 0; i--) 
                episode.get(i).reward += discountFactor * episode.get(i + 1).reward;

            double actorLoss = .0, criticloss = .0;
            for (int i = 0; i < episode.size(); i++) {
                double[] curLosses = computeLoss(episode.get(i), episode.size(), .1);
                criticloss += curLosses[0];
                actorLoss += curLosses[1];
            }

            double curLrDecay =  Math.pow(lrDecayRate, (double)epoch / lrDecayStep);
            critic.step(initCriticLearningRate * curLrDecay, .9, .999, epoch);
            actor.step(initActorLearningRate * curLrDecay, .9, .999, epoch);

            evaluate();
            if (verboseFreq > 0 && epoch % verboseFreq == 0) {
                System.out.println("[Epoch " + epoch + "/" + maxEpochs + "] " + "   critic loss: " + criticloss + ", actor loss: " + actorLoss  + ", score: " + player.score + ", distance survived: " + player.distSurvived);
            }

            if (player.score >= Math.max(2, maxScore*2.)) {
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
