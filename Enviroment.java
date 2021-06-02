package flappybird;

import java.util.*;
import javax.swing.*;
import java.awt.*;

public class Enviroment {
	static final int groundHeight = 100;
	static Random rand;
	static class Pillar {
		static final int holeLen = Player.displayRadius * 6;
		static final int minHoleHeight = holeLen / 4, maxHoleHeight = Main.height - groundHeight - holeLen - minHoleHeight;
		static final int displayWidth = Player.displayRadius * 9 / 2;
		Color color = Color.GRAY;

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
			return new Pillar(pos, minHoleHeight + rand.nextInt(maxHoleHeight - minHoleHeight));
		}
		boolean passOver(Player player) {
			return top.x + displayWidth <= player.displayPos;
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
		//rand = new Random(123);
	}	
	void reset() {
		//rand = new Random(123);
		pillars.clear();
	}
	int nearestPillarIndex(Player player) {
		for (int i = 0; i < pillars.size(); i++) 
			if (pillars.get(i).top.x + Pillar.displayWidth > player.displayPos - player.displayRadius) 
				return i;
		return 0;
	}
	// returns 0 if no crash, 1 if no crash and pass through, -1 otherwise;
	boolean check(Player player) {
		if (!pillars.isEmpty()) {
            int idx = nearestPillarIndex(player);
            if (player.crash(pillars.get(idx))) return false;
            if (!pillars.get(idx).flag && pillars.get(idx).passOver(player)) {
                pillars.get(idx).flag = true;
                //pillars.get(idx).color = Color.RED;
                player.score++;
            }
        }
        return true;
	}
	boolean update() {
		ArrayList<Pillar> newPillars = new ArrayList<>();
		if (!pillars.isEmpty()) {
			for (Pillar curPillar: pillars) {
				curPillar.top.x -= speed;
				curPillar.bottom.x -= speed;
				if (curPillar.top.x + Pillar.displayWidth < 0) continue;
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
		g.setColor(new Color(102, 51, 0)); // BROWN
		g.fillRect(0, Main.height - groundHeight, Main.width, groundHeight);
		for (Pillar curPillar: pillars) curPillar.draw(g);
	}
}