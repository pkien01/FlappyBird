import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.awt.Graphics;
public class QLearning implements Entity {
    static final int[] architecture = {6, 12, 24, 12, 1};
    static final double discountFactor = .9;
    Player player;
    NeuralNetwork brain;
    Enviroment env;
    QLearning(Enviroment env, String paramFile) {
        this.env = env;
        player = new Player();
        brain = paramFile != null?  NeuralNetwork.load(paramFile) : new NeuralNetwork(architecture);
    }

    double step(ActionStatePair actionStatePair, double learningRate, int epoch, NeuralNetwork targetBrain) {
        //Matrix bestNextState = actionStatePair.getNextActionStateInput(getBestAction(actionStatePair.nextState));
        //double targetVal = actionStatePair.reward + discountFactor * targetBrain.forward(bestNextState).data[0][0];
        Matrix targetVal = new Matrix(actionStatePair.reward);
        Matrix curPred = brain.forward(actionStatePair.getCurActionStateInput());
        //System.out.println("predicted reward: " + curPred.data[0][0] + " " + ", actual reward: " + reward);
        brain.backward(curPred.subtract(targetVal), learningRate, .9, .999, epoch + 1);
        return Math.pow(curPred.data[0][0] - targetVal.data[0][0], 2);
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
        final double epsStart = 0.9, epsEnd = 0.05, epsDecay = 5000.;
        final double tau = 0.05;
        final double lrDecayRate = 0.96, lrDecayStep = 50000.;

        Random rand = new Random();
        List<ActionStatePair> memory = new ArrayList<>();

        final int maxMemorySize = 1000000;

        double eps = epsStart;
        int maxScore = 0;
        NeuralNetwork target = new NeuralNetwork(brain);
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            //System.out.println("posBuffer size: " + posBuffer.size());
            //System.out.println("negBuffer size: " + negBuffer.size());
            eps = epsEnd + (epsStart - epsEnd) * Math.exp(-1.*epoch / epsDecay);
            //eps *= .9999;
            //if (epoch % 100 == 0) System.out.println("epoch " + epoch + ": " + eps);
            for (int i = 0; i < batchSize; i++) {
                List<ActionStatePair> episode = new ArrayList<>();
                player.reset();
                env.reset();
                
                while (player.score <= 100000) {
                    State curState = new State(player, env);
                    
                    int action;
                    if (rand.nextDouble() < eps) action = rand.nextBoolean()? 1 : 0;
                    else action = getBestAction(curState);
                    if (action == 1) player.tap();

                    player.update();
                    env.update();
                    
                    if (player.crash(env)) {
                        episode.add(new ActionStatePair(curState, null, action, -curState.penalty*3)); 
                        break;
                    }
                    State nextState = new State(player, env);
                    episode.add(new ActionStatePair(curState, nextState, action, 1.));
                }

                for (int j = episode.size() - 2; j >= 0; j--) 
                    episode.get(j).reward += discountFactor * episode.get(j + 1).reward;
                    
                int memoryOverflow =  memory.size() + episode.size() - maxMemorySize;
                if (memoryOverflow <= 0) memory.addAll(episode);
                else if (memoryOverflow <= memory.size()) {
                    memory.subList(0, memoryOverflow).clear();
                    memory.addAll(episode);
                } else {
                    memory.clear();
                    memoryOverflow -= memory.size();
                    episode.subList(0, memoryOverflow).clear();
                    memory.addAll(episode);
                }
            }

            double loss = 0.0;
            double curLearningRate = initLearningRate * Math.pow(lrDecayRate, (double)epoch / lrDecayStep);
            
            if (!memory.isEmpty()) {
                for (int i = 0; i < batchSize; i++) 
                    loss += step(memory.get(rand.nextInt(memory.size())), curLearningRate, epoch, target) / batchSize;
            }

            /*for (int i = 0; i < target.layers.length; i++) {
                target.layers[i].weight.add(brain.layers[i].weight, 1. - tau);
                target.layers[i].bias.add(brain.layers[i].bias, 1. - tau);
            }
             for (int i = 0; i < batchSize; i++) {
                double curLearningRate = initLearningRate / Math.sqrt(epoch + 1);
                if (!posBatch.isEmpty()) loss += step(posBatch.get(i % posBatch.size()), curLearningRate) / batchSize;
                if (!negBatch.isEmpty()) loss += step(negBatch.get(i % negBatch.size()), curLearningRate) / batchSize;
            }*/
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
