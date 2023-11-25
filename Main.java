import javax.swing.*;
import java.awt.*;


public class Main extends JFrame {
	static final int width = 600, height = 600;  
    public static void main(String[] args) { 
        Game.Mode gameMode = Game.Mode.NORMAL;
        if (args.length > 1) {
            throw new RuntimeException("Too many arguments");
        }   
        if (args.length == 1) {
            if (args[0].equals("-g") || args[0].equals("--genetic"))
                gameMode = Game.Mode.GENETIC;
            else if (args[0].equals("-q") || args[0].equals("--qlearning")) {
                gameMode = Game.Mode.QLEARNING;
            }
            else {
                throw new RuntimeException("Invalid argument: " + args[0]);
            }
        }

      	JFrame frame = new JFrame("Flappy Bird");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        //frame.setLocationRelativeTo(null);
        
		frame.setSize(width, height);
		frame.add(new Game(gameMode)); 
        frame.setVisible(true);
    }
}