import javax.swing.*;
import javax.swing.Timer;
import java.awt.*;
import java.awt.event.*;
import java.util.List;

public class Game extends JPanel implements ActionListener {
	enum Mode { NORMAL, GENETIC, QLEARNING}

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
	GeneticAlgorithm genetic;
	QLearning qlearning;
	Enviroment env;
	Controller control;
	boolean gameStarted, gameOver;
	Mode mode;
	public Game(Mode mode) {
		env = new Enviroment();
		switch (mode) {
			case NORMAL: bird = new Player(); break;
			case GENETIC:  genetic = new GeneticAlgorithm(env, Main.listFiles(Main.GENETIC_FOLDER_DEFAULT)); break;
			case QLEARNING: qlearning = new QLearning(env, Main.Q_LEARNING_FILE_DEFAULT); break;
			default: throw new RuntimeException("Mode " + mode + " does not exist");
		}
		gameStarted = gameOver = false;
		this.mode = mode;

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
    	if (mode == Mode.NORMAL) bird.draw(g);
		else if (mode == Mode.GENETIC) genetic.draw(g);
		else qlearning.draw(g);
    	
    	if (gameOver) {
    		g.setColor(Color.RED);
    		g.setFont(new Font("Helvetica", Font.BOLD, 30)); 
    		g.drawString("GAME OVER!", Main.width / 2 - 100 , Main.height / 2 - 30);
    	}
    	g.setColor(Color.GREEN.darker());
    	g.setFont(new Font("Helvetica", Font.BOLD, 15)); 

		switch (mode) {
			case NORMAL: 
				g.drawString("Score: " + bird.score, Main.width - 110, 20);
				break;
			case GENETIC:
				g.drawString("Generation #: " + genetic.numGenerations, Main.width - 150, 20);
				g.drawString("Max score : " + genetic.maxScore, Main.width - 150, 40);
				break;
			case QLEARNING:
				g.drawString("Generation #: " + qlearning.numGenerations, Main.width - 150, 20);
				g.drawString("Max score : " + qlearning.maxScore, Main.width - 150, 40);
				break;
		}
    }
    @Override
    public void actionPerformed(ActionEvent event) {
		if (mode == Mode.NORMAL) {
			if (!gameOver) {
				if (control.curKey == KeyEvent.VK_SPACE) {
					gameStarted = true;
					bird.tap();
					control.reset();
				}
				if (gameStarted) {
					bird.update();
					env.update();
					if (bird.crash(env)) gameOver = true;
				}
			}
			else if (control.curKey == KeyEvent.VK_SPACE) {
				bird.reset();
				env.reset();
				control.reset();
				gameStarted = gameOver = false;
			}
		} else if (mode == Mode.GENETIC) {
			if (control.curKey == KeyEvent.VK_SPACE) {
				gameStarted = true;
				control.reset();
			}
			if (gameStarted) {
				genetic.update();
				env.update();
			}
		} else if (mode == Mode.QLEARNING) {
			if (control.curKey == KeyEvent.VK_SPACE) {
				gameStarted = true;
				control.reset();
			}
			if (gameStarted) {
				qlearning.update();
				env.update();
			}
		}
    	repaint();
    }

}
