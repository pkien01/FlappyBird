import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.awt.Graphics;

public class QLearning implements Entity {
    static final int[] architecture = { 6, 128, 1 };
    static final double discountFactor = .99;
    double[] qValues;
    Player player;
    NeuralNetwork brain;
    Enviroment env;

    QLearning(Enviroment env, String paramFile) {
        this.env = env;
        player = new Player();
        brain = paramFile != null ? NeuralNetwork.load(paramFile) : new NeuralNetwork(architecture);
        qValues = new double[2];
    }

    double computeLoss(ActionStatePair actionStatePair, NeuralNetwork targetBrain) {
        double targetVal = actionStatePair.reward;
        if (actionStatePair.nextState != null) {
            Matrix bestNextState =
            actionStatePair.getNextActionStateInput(getBestAction(actionStatePair.nextState));
            targetVal += discountFactor * targetBrain.forward(bestNextState).data[0][0];
        }
        //Matrix targetVal = new Matrix(actionStatePair.reward);
        Matrix curPred = brain.forward(actionStatePair.getCurActionStateInput());
        brain.backward(curPred.subtract(targetVal));
        return Math.pow(curPred.data[0][0] - targetVal, 2) / 2;
    }

    int getBestAction(State state) {
        qValues[0] = brain.forward(ActionStatePair.getStateActionInput(state, 0)).data[0][0];
        qValues[1] = brain.forward(ActionStatePair.getStateActionInput(state, 1)).data[0][0];
        // System.out.println("noTapReward: " + noTapReward + ", tapReward: " +
        // tapReward);
        return qValues[0] >= qValues[1] ? 0 : 1;
    }

    void evaluate() {
        player.reset();
        env.reset();
        while (player.score <= Main.terminalScore) {
            if (player.crash(env))
                return;
            State curState = new State(player, env);
            if (getBestAction(curState) == 1)
                player.tap();

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

    static void standardizeReward(List<ActionStatePair> batch) {
        double meanReward = batch.stream().mapToDouble(x -> x.reward).average().getAsDouble();
        double stdReward = Math.sqrt(
                batch.stream().mapToDouble(x -> Math.pow(x.reward - meanReward, 2)).average().getAsDouble());

        for (int i = 0; i < batch.size(); i++)
            batch.get(i).reward = (batch.get(i).reward - meanReward) / (stdReward + 1e-9);
    }

    void optimize(double initLearningRate, int maxEpochs, int batchSize, int verboseFreq) {
        System.out.println("Q-function training is starting");

        // System.out.println("posBuffer size: " + posBuffer.size());
        // System.out.println("negBuffer size: " + negBuffer.size());
        final double epsStart = 0.9, epsEnd = 0.05, epsDecay = 20000.;
        final double tau = 0.005;
        final double lrDecayRate = 0.95, lrDecayStep = 50000.;
        final int maxMemorySize = 1000000;

        Random rand = new Random();
        CircularBuffer<ActionStatePair> stateMemory = new CircularBuffer<>(maxMemorySize);
        CircularBuffer<EnviromentPlayerPair> gameMemory = new CircularBuffer<>(maxMemorySize);

        int maxScore = 0;
        // long maxDistSurvived = 0;
        NeuralNetwork target = new NeuralNetwork(brain);
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            // System.out.println("posBuffer size: " + posBuffer.size());
            // System.out.println("negBuffer size: " + negBuffer.size());
            double eps = epsEnd + (epsStart - epsEnd) * Math.exp(-1. * epoch / epsDecay);
        
            if (gameMemory.size() == 0 || rand.nextDouble() < .25) {
                player.reset();
                env.reset();
            } else {
                EnviromentPlayerPair prevGame = gameMemory.get(rand.nextInt(gameMemory.size()));
                player = new Player(prevGame.player);
                env = new Enviroment(prevGame.enviroment);
            }
            
            double curLearningRate = initLearningRate * Math.pow(lrDecayRate, (double) epoch / lrDecayStep);
            double loss = 0.0;

            CircularBuffer<EnviromentPlayerPair> window = new CircularBuffer<>(100);
            while (player.score <= Main.terminalScore) {
                State curState = new State(player, env);
                window.add(new EnviromentPlayerPair(env, player));

                int action;
                if (rand.nextDouble() < eps)
                    action = rand.nextBoolean() ? 1 : 0;
                else
                    action = getBestAction(curState);
                if (action == 1)
                    player.tap();

                player.update();
                env.update();

                if (player.crash(env)) {
                    stateMemory.add(new ActionStatePair(curState, null, action, -100.));
                    break;
                }

                State nextState = new State(player, env);
                stateMemory.add(new ActionStatePair(curState, nextState, action, 0.));

                List<ActionStatePair> batch = new ArrayList<>(batchSize);
                for (int i = 0; i < batchSize; i++) batch.add(stateMemory.get(rand.nextInt(stateMemory.size())));
                standardizeReward(batch);
                brain.zeroGrad();
                for (int i = 0; i < batchSize; i++) {
                    loss += computeLoss(batch.get(i), target) / batchSize;
                }
                brain.step(curLearningRate, .9, .999, epoch, batchSize);

                for (int i = 0; i < target.layers.length; i++) {
                    target.layers[i].weight.add(brain.layers[i].weight, 1. - tau);
                    target.layers[i].bias.add(brain.layers[i].bias, 1. - tau);
                }
            }
            if (window.size() == 100) {
                for (int i = 30; i <= 90; i++)
                    gameMemory.add(window.get(i));
            }
            loss /= player.distSurvived;
            

            evaluate();
            if (verboseFreq > 0 && epoch % verboseFreq == 0) {
                System.out.println("[Epoch " + epoch + "/" + maxEpochs + "] " + "Q loss: " + loss + ", score: "
                        + player.score + ", distance survived: " + player.distSurvived);
            }

            if (player.score >= Main.terminalScore || player.score >= Math.max(2, maxScore * 2)) {
                brain.save(String.format(Main.Q_LEARNING_FILE_FORMAT, player.score, epoch));
                brain.save(Main.Q_LEARNING_FILE_DEFAULT);
                maxScore = player.score;
            }
            if (maxScore >= Main.terminalScore)
                break;
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
        if (getBestAction(curState) == 1)
            player.tap();
        player.update();
    }

    public void draw(Graphics g) {
        player.draw(g);
    }
}
