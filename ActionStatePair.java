import java.util.List;

public class ActionStatePair {
    State curState, nextState;
    int action;
    double reward;
    ActionStatePair(State curState, State nextState, int action, double reward) {
        this.curState = curState;
        this.nextState = nextState;
        this.action = action;
        this.reward = reward;
    }
    ActionStatePair(ActionStatePair other) {
        curState = other.curState;
        nextState = other.nextState;
        action = other.action;
        reward = other.reward;
    }
    Matrix getCurActionStateInput() {
        List<Double> res = curState.toList();
        res.add((double)action);
        return new Matrix(res);
    }
    Matrix getNextActionStateInput(int nextAction) {
        List<Double> res = nextState.toList();
        res.add((double)nextAction);
        return new Matrix(res);
    }
}
