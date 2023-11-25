import java.awt.Graphics;
import java.util.*;



public class GeneticAlgorithm implements Entity{
    private static final int maxCapacity = 1000;
    
    ArrayList<Agent> agents;
    Queue<Agent> bestAgents;
    int maxScore;
    int size;
    int numKeeps, numMutations, numCombinations;
    int numGenerations;
    Random rand;
    long totalFitness;
    Enviroment env;
    GeneticAlgorithm(int populations, int numKeeps, int numMutations, int numCombinations) {
        this.numKeeps = numKeeps; this.numMutations = numMutations; this.numCombinations = numCombinations;
        size = populations;
        maxScore = 0;
        agents = new ArrayList<>(size);
        for (int i = 0; i < size; i++) agents.add(new Agent());
        rand = new Random();
        bestAgents = new PriorityQueue<>(maxCapacity,
            new Comparator<Agent>() {
                @Override
                public int compare(Agent lhs, Agent rhs) {
                    return Long.compare(lhs.fitness(), rhs.fitness());
                }
            }
        );
        totalFitness = 0;
    }
    GeneticAlgorithm getEnvironment(Enviroment env) {
        this.env = env;
        return this;
    }
    Agent selectAgent() {
        double randNum = rand.nextDouble()*totalFitness;
        double prefFitness = 0.;
        for (Agent agent: bestAgents) {
            prefFitness += agent.fitness();
            if (randNum <= prefFitness) 
                return new Agent(agent);
        }
        throw new RuntimeException("this cannot happen in selectAgent");
    }
    void nextGeneration() {
        if (agents.isEmpty()) return;

        for (Agent agent: agents) {
            if (bestAgents.size() < maxCapacity) {
                bestAgents.add(agent);
                totalFitness += agent.fitness();
            }
        }
        ArrayList<Agent> children = new ArrayList<>(numKeeps + numMutations + numCombinations);

        for (int i = 0; i < numKeeps; i++) children.add(selectAgent());
        for (int i = 0; i < numMutations; i++) children.add(selectAgent().mutate());   
        for (int i = 0; i < numCombinations; i++) children.add(selectAgent().mate(selectAgent()));

        agents.clear();
        agents = children;
        size = agents.size();
    }
    public void draw(Graphics g) {
        agents.stream().filter(Agent::isAlive).forEach(agent -> agent.draw(g));
    }
    public void update() {
        if (agents.isEmpty()) return;
        int numAlives = 0;
        for (Agent agent: agents) {
            if (agent.isAlive()) {
                maxScore = Math.max(maxScore, agent.player.score);
                agent.getInput(env).update();
                numAlives++;
            }
        }
        if (numAlives == 0) {
            env.reset();
            nextGeneration();
            numGenerations++;
        }
    }
}
