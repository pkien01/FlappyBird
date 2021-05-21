package flappybird;

import javax.swing.*;
import javax.swing.Timer;
import java.awt.*;
import java.awt.event.*;
import java.util.*;

public class GeneticsAlgorithm extends Game {
	static class AIPlayer extends Player implements Comparable<AIPlayer> {
		NeuralNetwork brain;
        int distTravelled;
        double distToNextHole;
		boolean alive;
        static double distance(Point p1, Point p2) {
            int diff_x = p1.x - p2.x, diff_y = p1.y - p2.y;
            return Math.sqrt((double)diff_x * diff_x + diff_y * diff_y);
        }
        double distanceTo(Point p) {
            return distance(new Point(displayPos, (int)Math.round(height)), p);
        }
        double distanceTo(Enviroment.Pillar pl) {
            return distanceTo(new Point(pl.top.x + pl.displayWidth / 2, pl.top.height + pl.holeLen / 2));
        }
		AIPlayer(NeuralNetwork brain) {
			super();
			this.brain = brain;
            this.distTravelled = 0;
            this.distToNextHole = distanceTo(new Point(Main.width, 0));
			alive = true;
		}
        @Override 
        public void reset() {
            super.reset();
            this.distTravelled = 0;
            this.distToNextHole = distanceTo(new Point(Main.width, 0));
        }
        @Override 
        public void update() {
            super.update();
            distTravelled += Enviroment.speed;
        }
        double fitness() {
            return (double)distTravelled - distTravelled;
        }
		@Override
		public int compareTo(AIPlayer other) {
			if (other.fitness() == fitness()) return 0;
            if (other.fitness() < fitness()) return -1;
            return 1;
		}
	}


	int population_size, remain_size;
	double mutate_rate;

	ArrayList<AIPlayer> ai_birds;
	int cntAlive, generations;
	GeneticsAlgorithm(int population_size, int remain_size, double mutate_rate) {
		super();
		this.population_size = population_size;
		this.remain_size = remain_size;
		this.mutate_rate = mutate_rate;

		ai_birds = new ArrayList<>();
		for (int i = 0; i < population_size; i++) ai_birds.add(new AIPlayer(new NeuralNetwork(new int[]{3, 10, 1})));	

		cntAlive = population_size;
		generations = 0;
	}
	void nextGeneration() {
		generations++;
		Collections.sort(ai_birds);


		for (int i = 0; i < remain_size; i++) ai_birds.set(i, new AIPlayer(ai_birds.get(i).brain.copy()));
		for (int i = remain_size; i < population_size; i++) {
			NeuralNetwork curBrain = ai_birds.get(i % remain_size).brain.copy();
			curBrain.mutate(mutate_rate);
			ai_birds.set(i, new AIPlayer(curBrain));
		}

		cntAlive = population_size;
	}

	@Override
    protected void paintComponent(Graphics g) {
    	super.paintComponent(g);
    	
    	g.setColor(Color.CYAN.brighter());
    	g.fillRect(0, 0, Main.width, Main.height);
    	env.draw(g);
    	int maxScore = 0;
    	for (AIPlayer bird: ai_birds) {
    		if (!bird.alive) continue;
    		bird.draw(g);
    		maxScore = Math.max(maxScore, bird.score);
    	}
    	g.setColor(Color.GREEN.darker());
    	g.setFont(new Font("Helvetica", Font.BOLD, 15)); 
    	g.drawString("Score: " + maxScore, Main.width - 110, 20);

    	g.setColor(Color.RED.darker());
    	g.setFont(new Font("Helvetica", Font.BOLD, 15)); 
    	g.drawString("Number of generations: " + generations, 20, 20);
    }

	@Override
    public void actionPerformed(ActionEvent event) {
    	if (!gameOver) {
    		if (control.curKey == KeyEvent.VK_ENTER) {
    			if (!gameStarted) gameStarted = true;
    		}
    		if (gameStarted) {
    			for (int i = 0; i < ai_birds.size(); i++) {
    				if (!ai_birds.get(i).alive) continue;
    				int nextIdx = env.nearestPillarIndex(ai_birds.get(i));
    				Enviroment.Pillar nextPillar = new Enviroment.Pillar(Main.width, Main.height);
    				if (nextIdx < env.pillars.size() && env.pillars.get(nextIdx).top.x < Main.width) 
    					nextPillar = env.pillars.get(nextIdx);

                    ai_birds.get(i).distToNextHole = ai_birds.get(i).distanceTo(nextPillar);
    			
    				double[] features = new double[]{(double)ai_birds.get(i).height, (double)nextPillar.top.x, (double)nextPillar.top.height};
                    double[] pred = ai_birds.get(i).brain.forward(features);
                    //System.out.println(pred[0]);
    				if (pred[0] > 0.5) ai_birds.get(i).tap();
    				ai_birds.get(i).update();
    				if (ai_birds.get(i).crash() || !env.check(ai_birds.get(i))) {
    					ai_birds.get(i).alive = false;
    					cntAlive--;
    				}
    				if (cntAlive == 0) {
    					env.reset();
    					nextGeneration();
    				}
    			}
    			env.update();
    		}
    	}
    	//System.out.println(cntAlive);
    	repaint();
    }
}