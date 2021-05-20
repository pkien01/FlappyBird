package flappybird;

import javax.swing.*;
import javax.swing.Timer;
import java.awt.*;
import java.awt.event.*;
import java.util.*;


public class Game extends JPanel implements ActionListener {
	static final int deltaTime = 5;

	static class Controller extends KeyAdapter {
		int curKey;
		Controller() {
			curKey = -1;
		}
		void reset() {
			curKey = -1;
		}
		@Override
		public void keyPressed(KeyEvent event) {
    		curKey = event.getKeyCode();
    	}
    	@Override
    	public void keyReleased(KeyEvent event) {
    		curKey = -1;
    	}
	}

	Timer timer;
	Player bird;
	Enviroment env;
	Controller control;
	boolean gameStarted, gameOver;
	public Game() {
		bird = new Player();
		env = new Enviroment();
		gameStarted = gameOver = false;

		control = new Controller();
		addKeyListener(control);
		setFocusable(true);
		timer = new Timer(deltaTime, this);
		timer.start();
	}

	@Override
    protected void paintComponent(Graphics g) {
    	super.paintComponent(g);
    	
    	g.setColor(Color.CYAN.brighter());
    	g.fillRect(0, 0, Main.width, Main.height);
    	env.draw(g);
    	bird.draw(g);
    	
    	if (gameOver) {
    		g.setColor(Color.RED);
    		g.setFont(new Font("Helvetica", Font.BOLD, 30)); 
    		g.drawString("GAME OVER!", Main.width / 2 - 100 , Main.height / 2 - 30);
    	}
    	g.setColor(Color.GREEN.darker());
    	g.setFont(new Font("Helvetica", Font.BOLD, 15)); 
    	g.drawString("Score: " + bird.score, Main.width - 110, 20);
    }
    @Override
    public void actionPerformed(ActionEvent event) {
    	if (!gameOver) {
    		if (control.curKey == KeyEvent.VK_SPACE) {
    			gameStarted = true;
    			bird.tap();
    			control.reset();
    		}
    		if (gameStarted) {
    			bird.update();
                env.update();
                if (bird.crash() || !env.check(bird)) gameOver = true;
    		}
    	}
    	else if (control.curKey == KeyEvent.VK_SPACE) {
    		bird.reset();
    		env.reset();
    		control.reset();
    		gameStarted = gameOver = false;
    	}
    	repaint();
    }

}
