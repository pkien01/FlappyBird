import java.util.ArrayList;
import java.util.List;

class CircularBuffer<T> {
    List<T> buffer;
    int capacity, front;
    CircularBuffer(int capacity) {
        this.capacity = capacity;
        buffer = new ArrayList<>(capacity);
        front = 0;
    }
    void add(T elem) {
        if (buffer.size() == capacity) {
            buffer.set(front, elem);
            front = (front + 1) % capacity;
        }
        else {
            buffer.add(elem);
        }
    }
    T get(int idx) {
        return buffer.get((front + idx) % buffer.size());
    }
    int size() {
        return buffer.size();
    }
}