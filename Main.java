import java.io.*;
import java.nio.file.Files;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Calendar;
import java.util.stream.Collectors;
import java.util.List;

public class Main {
	static final int width = 600, height = 600;  

    static final String Q_LEARNING_FILE_FORMAT = "./qlearning/score-%d-epoch-%d"; 
    static final String Q_LEARNING_FILE_DEFAULT = "./qlearning/default";

    static final String GENETIC_FILE_FORMAT = "./genetic/maxscore-%d-generation-%d/score-%d-bird-%d";
    static final String GENETIC_FOLDER_DEFAULT = "./genetic/default";
    static final String GENETIC_FILE_DEFAULT = GENETIC_FOLDER_DEFAULT + "/score-%d-bird-%d";

    public static void play(Game.Mode gameMode) {
        javax.swing.JFrame frame = new javax.swing.JFrame("Flappy Bird");
        frame.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
        //frame.setLocationRelativeTo(null);
        
	frame.setSize(width, height);
	frame.add(new Game(gameMode)); 
        frame.setVisible(true);
    }
    public static List<String> listFiles(String folder) {
        File dir = new File(folder);
        assert dir.isDirectory();
        return Arrays.stream(dir.listFiles()).filter(File::isFile).map(File::getPath).collect(Collectors.toList());
    }
    public static void deleteDir(File dir) {
        File[] contents = dir.listFiles();
        if (contents != null) {
            for (File f : contents) {
                if (! Files.isSymbolicLink(f.toPath())) {
                    deleteDir(f);
                }
            }
        }
        dir.delete();
    }
    public static void train(Game.Mode gameMode, int iterations, int verboseFreq) {
        Enviroment emulator = new Enviroment();
        if (gameMode == Game.Mode.GENETIC) {
            File defaultParent = new File(GENETIC_FOLDER_DEFAULT);
            deleteDir(defaultParent);
            defaultParent.mkdirs();
            GeneticAlgorithm geneticAlgorithm = new GeneticAlgorithm(emulator, 100, 20, 30, 50);
            int prevGeneration = -1;
            int overallMaxScore = 0;
            while (geneticAlgorithm.maxScore <= 1000000 && geneticAlgorithm.numGenerations <= iterations) {
                geneticAlgorithm.update();
                emulator.update();
                if (geneticAlgorithm.maxScore >= 1000000 || geneticAlgorithm.maxScore >= Math.max(2, overallMaxScore*2)) {
                    String folderName = String.format((new File(GENETIC_FILE_FORMAT)).getParent(), geneticAlgorithm.maxScore, geneticAlgorithm.numGenerations);
                    File folder = new File(folderName);
                    if (!folder.exists()) folder.mkdirs();

                    for (int i = 0; i < geneticAlgorithm.agents.size(); i++) {
                        Agent agent = geneticAlgorithm.agents.get(i);
                        agent.brain.save(String.format(GENETIC_FILE_FORMAT, geneticAlgorithm.maxScore, geneticAlgorithm.numGenerations, agent.player.score, i));
                        agent.brain.save(String.format(GENETIC_FILE_DEFAULT, agent.player.score, i));
                    }
                    overallMaxScore = geneticAlgorithm.maxScore;
                }

                if (geneticAlgorithm.numGenerations != prevGeneration && verboseFreq > 0 && geneticAlgorithm.numGenerations % verboseFreq == 0) {
                    System.out.print("[Generation " + geneticAlgorithm.numGenerations + "]: ");
                    System.out.println("max score = " + geneticAlgorithm.maxScore + ", max distance survived = " + geneticAlgorithm.maxDistSurvived);
                    prevGeneration = geneticAlgorithm.numGenerations;
                    geneticAlgorithm.maxScore = 0;
                    geneticAlgorithm.maxDistSurvived = 0;
                }
            }
            System.out.print("[Generation " + geneticAlgorithm.numGenerations + "]: ");
            System.out.println("max score = " + geneticAlgorithm.maxScore + ", max distance survived = " + geneticAlgorithm.maxDistSurvived);
        }
        else if (gameMode == Game.Mode.QLEARNING) {
            File defaultParent = new File((new File(Q_LEARNING_FILE_DEFAULT)).getParent());
            deleteDir(defaultParent);
            defaultParent.mkdirs();
            QLearning qLearning = new QLearning(emulator, null);
            qLearning.optimize(1e-3, iterations, 128, verboseFreq);
        } 
    }
    public static void main(String[] args) { 
        Game.Mode gameMode = Game.Mode.NORMAL; 
        if (args.length == 0) {
            play(Game.Mode.NORMAL);
        }
        else {
            int verboseFreq = 0, iterations = 0;
            if (args[0].equals("-g") || args[0].equals("--genetic")) {
                gameMode = Game.Mode.GENETIC;
                verboseFreq = 10;
                iterations = 3000000;
            }
            else if (args[0].equals("-q") || args[0].equals("--qlearning")) {
                gameMode = Game.Mode.QLEARNING;
                verboseFreq = 10;
                iterations = 1000000;
            }
            else {
                throw new RuntimeException("Invalid mode: " + args[0]);
            }

            if (args.length == 1) {
                play(gameMode);
            }
            else if (args.length == 2) {
                if (args[1].equals("-t") || args[1].equals("--train")) 
                    train(gameMode, iterations, verboseFreq);
            }
            else 
                throw new RuntimeException("Too many arguments");
        }  
    }  

}
