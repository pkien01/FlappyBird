package v1;

import java.util.*;
import javax.swing.*;
import java.awt.*;

public class Enviroment {
	static final int groundHeight = 100;
	static class Pillar {
		static final int holeLen = Player.displayRadius * 6, minHoleHeight = holeLen / 4;
		static final int displayWidth = Player.displayRadius * 9 / 2;
		static final Color color = Color.GRAY;

		static Random rand;

		boolean flag;
		Rectangle top, bottom;
		Pillar(int pos, int holeHeight) {
			this.top = new Rectangle(pos, 0, displayWidth, holeHeight);
			this.bottom = new Rectangle(pos, holeHeight + holeLen, displayWidth, Main.height - holeHeight - holeLen - groundHeight);
			this.flag = false;
		}
		//generate random gap
		static Pillar generate(int pos) {
			if (rand == null) rand = new Random();
			return new Pillar(pos, minHoleHeight + rand.nextInt(Main.height - holeLen - minHoleHeight * 2 - groundHeight));
		}
		void draw(Graphics g) {
			g.setColor(color);
			g.fillRect(top.x, top.y, Math.min(top.width, Main.width - top.x), top.height);
			g.fillRect(bottom.x, bottom.y, Math.min(bottom.width, Main.width - bottom.x), bottom.height);
		}
	}

	static final int pillarsGap = Main.width / 3;
	static int speed = 2;

	ArrayList<Pillar> pillars;
	Enviroment() {
		pillars = new ArrayList<>();
	}	
	void reset() {
		pillars.clear();
	}
	// returns true if no crash, false otherwise;
	boolean update(Player player) {
		ArrayList<Pillar> newPillars = new ArrayList<>();
		if (!pillars.isEmpty()) {
			for (Pillar curPillar: pillars) {
				curPillar.top.x -= speed;
				curPillar.bottom.x -= speed;
				if (curPillar.top.x + Pillar.displayWidth < 0) continue;

				if (player.crash(curPillar)) return false;
				if (!curPillar.flag && curPillar.top.x + Pillar.displayWidth <= player.displayPos) {
					player.score++;
					curPillar.flag = true;
				}
				newPillars.add(curPillar);
			}
		}

		int lastPos = newPillars.isEmpty()? Main.width : newPillars.get(newPillars.size() - 1).top.x;
		while (lastPos < Main.width + (pillarsGap + Pillar.displayWidth) * 2) {
			lastPos += pillarsGap + Pillar.displayWidth;
			newPillars.add(Pillar.generate(lastPos));
		}
		pillars = newPillars;
		return true;
	}
	void draw(Graphics g) {
		g.setColor(new Color(102, 51, 0));
		g.fillRect(0, Main.height - groundHeight, Main.width, groundHeight);
		for (Pillar curPillar: pillars) curPillar.draw(g);
	}
}