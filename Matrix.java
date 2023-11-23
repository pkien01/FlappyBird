import java.util.function.Function;
import java.util.function.Supplier;

public class Matrix {
	int n, m;
	double[][] data;
	Matrix(int rows, int cols) {
		data = new double[rows][cols];
		n = rows;
		m = cols;
	}
	Matrix multiply(Matrix other) {
		assert m == other.n;

		Matrix res = new Matrix(n, other.m);
		for (int i = 0; i < n; i++) 
			for (int k = 0; k < m; k++)
				for (int j = 0; j < other.m; j++)
					res.data[i][j] += data[i][k] * other.data[k][j];
		return res;
	}
	Matrix transpose() {
		Matrix res = new Matrix(m, n);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++) res.data[j][i] = data[i][j]; 
		return res;
	}
	Matrix add(Matrix other) {
		assert n == other.n && m == other.m;
		Matrix res = new Matrix(n, m);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++) res.data[i][j] = data[i][j] + other.data[i][j];
		return res;
	} 
	Matrix add(Matrix other, double alpha) {
		assert n == other.n && m == other.m;
		Matrix res = new Matrix(n, m);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++) res.data[i][j] = alpha*data[i][j] + (1. - alpha)*other.data[i][j];
		return res;
	} 
	Matrix apply(Function<Double, Double> func) {
		Matrix res = new Matrix(n, m);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++) res.data[i][j] = func.apply(data[i][j]);
		return res;
	}
	void fill(Supplier<Double> func) {
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++) data[i][j] = func.get();
	}
	Matrix sameSize() {
		return new Matrix(n, m);
	}
}