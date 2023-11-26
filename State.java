import java.util.*;

public class State {
    List<Double> state;
    double penalty;
    State(Player player, Enviroment env) {
        state = new ArrayList<>();
        int idx = env.nearestPillarIndex(player);
        if (idx >= 0 && idx < env.pillars.size() && env.pillars.get(idx).top.x < Main.width - Enviroment.Pillar.displayWidth / 2) {
            Enviroment.Pillar curPillar = env.pillars.get(idx);
            state.add(player.distanceTo(curPillar.top) / Main.width);
            state.add(player.distanceTo(curPillar.bottom) / Main.width);
            double midPillarHeight = (curPillar.top.y + curPillar.top.height + curPillar.bottom.y) / 2;
            penalty = player.distanceTo(curPillar.top.x + curPillar.displayWidth, midPillarHeight) / Math.hypot(Enviroment.pillarsGap, Main.height);
        } else {
            state.add(1.);
            state.add(1.);
            penalty = 1.;
        }
        state.add(player.distanceToCeiling() / Main.height);
        state.add(player.distanceToGround() / Main.height);
        state.add(player.vertSpeed / Math.sqrt(2.*player.gravity*Main.height));
    } 
    Matrix toMatrix() {
        return new Matrix(state);
    }
    List<Double> toList() {
        return new ArrayList<>(state);
    }
    void clear() {
        state.clear();
    }
    int size() {
        return state.size();
    }
}
