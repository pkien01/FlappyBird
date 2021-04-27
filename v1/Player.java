package v1;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.AffineTransform;

public class Player {
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
	/*
	boolean outOfScreen() {
		return (int)Math.round(height) + displayRadius < 0 || (int)Math.round(height) - displayRadius >= Main.height;
	}*/
	boolean crash(Rectangle rec) {
		int x_c = displayPos, y_c = (int)Math.round(height);
		int x_nearest = Math.max(rec.x, Math.min(x_c, rec.x + rec.width));
		int y_nearest = Math.max(rec.y, Math.min(y_c, rec.y + rec.height));
		int diff_x = x_nearest - x_c, diff_y = y_nearest - y_c;
		return diff_x * diff_x + diff_y * diff_y <= displayRadius * displayRadius;
	}
	boolean crash(Enviroment.Pillar pillar) {
		return crash(pillar.top) || crash(pillar.bottom);
	}
	boolean crash() {
		return height + displayRadius >= Main.height - Enviroment.groundHeight;
	}
	
	void update() {
		height += vertSpeed * Game.deltaTime;
		//if ((int)Math.round(height) - displayRadius < 0) height = displayRadius;
		//if ((int)Math.round(height) + displayRadius >= Main.height) height = Main.height - displayRadius;
		vertSpeed += gravity * Game.deltaTime;

		if (Math.abs(wingsAngle + wingsSpeed) > maxWingsAngle) wingsSpeed = -wingsSpeed;
		wingsAngle += wingsSpeed;
	}
	void draw(Graphics g) {
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