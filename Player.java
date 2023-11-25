import java.awt.*;
import java.awt.geom.AffineTransform;

public class Player implements Entity {
	static final int displayRadius = 22;
	static final int initHeight = (Main.height - Enviroment.groundHeight) / 2;
	static final double gravity = 0.016, tapSpeed = -1.26;
	static final int displayPos = Main.width / 4;
	static final Color color = Color.YELLOW.darker();
	static final int wingsLen = displayRadius * 4 / 3, maxWingsAngle = 15;


	int wingsAngle, wingsSpeed;
	double height, vertSpeed;
	int score;

	Player() {
		this.height = (double)initHeight;
		vertSpeed = 0.0;
		score = 0;
		wingsAngle = 0; wingsSpeed = 5;
	}
	void reset() {
		this.height = (double)initHeight;
		vertSpeed = 0.0;
		score = 0;
		wingsAngle = 0; wingsSpeed = 5;
	}
	void tap() {
		vertSpeed = tapSpeed;
	}
	double distanceTo(double x, double y) {
		double diff_x = x - displayPos, diff_y = y - height;
		return Math.sqrt(diff_x * diff_x + diff_y * diff_y);
	}
	double distanceTo(Rectangle rec) {
		double x_c = displayPos, y_c = height;
		double x_nearest = Math.max(rec.x, Math.min(x_c, rec.x + rec.width));
		double y_nearest = Math.max(rec.y, Math.min(y_c, rec.y + rec.height));
		double diff_x = x_nearest - x_c, diff_y = y_nearest - y_c;
		return Math.sqrt(diff_x * diff_x + diff_y * diff_y);
	}
	double distanceToGround() {
		return Main.height - Enviroment.groundHeight - (height + displayRadius);
	}
	double distanceToCeiling() {
		return height - displayRadius;
	}
	boolean crash(Enviroment.Pillar pillar) {
		return distanceTo(pillar.top) <= displayRadius || distanceTo(pillar.bottom) <= displayRadius;
	}
	boolean crash(Enviroment env) {
		if (distanceToGround() < 0 || distanceToCeiling() < 0) return true;
		if (!env.pillars.isEmpty()) {
            int idx = env.nearestPillarIndex(this);
            if (crash(env.pillars.get(idx))) return true;
            if (!env.pillars.get(idx).flag && env.pillars.get(idx).passOver(this)) {
                env.pillars.get(idx).flag = true;
                //pillars.get(idx).color = Color.RED;
                score++;
            }
        }
        return false;
	}
	
	public void update() {
		height += vertSpeed * Game.deltaTime;
		//if ((int)Math.round(height) - displayRadius < 0) height = displayRadius;
		//if ((int)Math.round(height) + displayRadius >= Main.height) height = Main.height - displayRadius;
		vertSpeed += gravity * Game.deltaTime;

		if (Math.abs(wingsAngle + wingsSpeed) > maxWingsAngle) wingsSpeed = -wingsSpeed;
		wingsAngle += wingsSpeed;
	}
	public void draw(Graphics g) {
		//if (outOfScreen()) return;
		if (height + displayRadius < 0) return;
		//body
		g.setColor(color);
		g.fillOval(displayPos- displayRadius, (int)Math.round(height) - displayRadius, displayRadius * 2, displayRadius * 2);

		// eye
		Point eye = new Point(displayPos + displayRadius * 2 / 3, (int)Math.round(height) - displayRadius / 3);
		g.setColor(Color.WHITE); g.fillOval(eye.x - 6, eye.y - 6, 12, 12);
		g.setColor(Color.BLACK); g.fillOval(eye.x - 3, eye.y - 3, 6, 6);

		//beak
		g.setColor(Color.RED);
		g.fillOval(eye.x, (int)Math.round(height) - 4, displayRadius * 5 / 6, 8);

		// wing
		g.setColor(color.darker());
		Graphics2D g2d = (Graphics2D)g;
		AffineTransform old = g2d.getTransform();
		g2d.rotate(Math.toRadians(wingsAngle), displayPos, (int)Math.round(height));
		g2d.fillArc(displayPos - wingsLen, (int)Math.round(height) - displayRadius / 2, wingsLen, displayRadius, 180 - wingsAngle, 180);
		g2d.setTransform(old);
	}
}