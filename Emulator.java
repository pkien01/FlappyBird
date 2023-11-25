public class Emulator {
    Enviroment env;
    Player player;

    Emulator() {
        env = new Enviroment();
        player = new Player();
    }
    void tap() {
        player.tap();
    }
    void update() {
        player.update();
        env.update();
    }
}
