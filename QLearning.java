package flappybird;

import javax.swing.*;
import javax.swing.Timer;
import java.awt.*;
import java.awt.event.*;
import java.util.*;

public class QLearning extends Game {
	double alpha, gamma, epsilon;
	Random rand;

	double[][][] q;
	QLearning(double alpha, double gamma, double epsilon) {
		super();
		this.alpha = alpha; this.gamma = gamma; this.epsilon = epsilon;
		rand = new Random();
		q = new double[Main.width - bird.displayPos + 1][(Main.height - Enviroment.groundHeight) * 2 + 1][2];
	}

	int[] getState() {
		int nextIdx = env.nearestPillarIndex(bird);
		Enviroment.Pillar nextPillar = new Enviroment.Pillar(Main.width, (Main.height - Enviroment.groundHeight - Enviroment.Pillar.holeLen) / 2);
    	if (nextIdx < env.pillars.size() && env.pillars.get(nextIdx).top.x < Main.width) nextPillar = env.pillars.get(nextIdx);
    	int x_diff = nextPillar.top.x - bird.displayPos;
    	int h_diff = nextPillar.top.height + nextPillar.holeLen / 2 - (int)Math.round(bird.height) + Main.height - Enviroment.groundHeight;
		return new int[]{x_diff, h_diff};
	}

	public void actionPerformed(ActionEvent event) {
    	if (!gameOver) {
    		if (control.curKey == KeyEvent.VK_ENTER) {
    			if (!gameStarted) gameStarted = true;
    		}
    		if (gameStarted) {
    			int[] curState = getState();
    			int action = q[curState[0]][curState[1]][0] >= q[curState[0]][curState[1]][1]? 0 : 1;
    			//System.out.println(q[curState[0]][curState[1]][0] + " " + q[curState[0]][curState[1]][1]);
    			if (Math.random() < epsilon) action = rand.nextBoolean()? 1 : 0;
    			if (action == 1) bird.tap();
    			//System.out.println(action);
    			bird.update();
    			int[] nextState = getState();
    			double maxNextQ = Math.max(q[nextState[0]][nextState[1]][0], q[nextState[0]][nextState[1]][1]);
    			boolean alive = !bird.crash() && env.check(bird); 
    			q[curState[0]][curState[1]][action] += alpha * ((alive? 1.0 : -1000.0) + gamma*maxNextQ - q[curState[0]][curState[1]][action]);
    			if (!alive) {
    				bird.reset();
    				env.reset();
    			}
    			env.update();
    		}
    	}
    	repaint();
    }
}
