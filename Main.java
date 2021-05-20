package flappybird;

import javax.swing.*;
import java.awt.*;


public class Main extends JFrame {
	static final int width = 600, height = 600;  
    public static void main(String[] args) {    
      	JFrame frame = new JFrame("Flappy Bird");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        //frame.setLocationRelativeTo(null);
        
		frame.setSize(width, height);
		//frame.add(new Game()); 
		frame.add(new GeneticsAlgorithm(10, 0.2));
        frame.setVisible(true);
    }
}