import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

public class Matrix implements Serializable {
	int n, m;
	double[][] data;
	Matrix(int rows, int cols) {
		data = new double[rows][cols];
		n = rows;
		m = cols;
	}
	Matrix(int rows, int cols, double val) {
		this(rows, cols);
		for (int i = 0; i < rows; i++)
			for (int j = 0; j < cols; j++) data[i][j] = val;
	}
	Matrix(List<Double> data) {
		n = data.size(); m = 1;
		this.data = new double[n][m];
		for (int i = 0; i < n; i++) 
			this.data[i][0] = data.get(i);
	}
	Matrix(Matrix other) {
		this(other.n, other.m);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++) data[i][j] = other.data[i][j];
	}
	Matrix(double ...elem) {
		n = elem.length; m = 1;
		this.data = new double[n][m];
		for (int i = 0; i < n; i++) 
			this.data[i][0] = elem[i];
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
		assert n == other.n;
		assert m == other.m;
		Matrix res = new Matrix(n, m);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++) res.data[i][j] = data[i][j] + other.data[i][j];
		return res;
	} 
	Matrix subtract(Matrix other) {
		assert n == other.n;
		assert m == other.m;
		Matrix res = new Matrix(n, m);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++) res.data[i][j] = data[i][j] - other.data[i][j];
		return res;
	}
	Matrix subtract(double subtrahend) {
		Matrix res = new Matrix(n, m);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++) res.data[i][j] = data[i][j] - subtrahend;
		return res;
	}
	void subtractInPlace(Matrix other) {
		assert n == other.n;
		assert m == other.m;
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++) data[i][j] -= other.data[i][j];
	}
	Matrix multiplyEntrywise(Matrix other) {
		assert n == other.n;
		assert m == other.m;
		Matrix res = new Matrix(n, m);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++) res.data[i][j] = data[i][j] * other.data[i][j];
		return res;
	}
	Matrix multiply(double factor) {
		Matrix res = new Matrix(n, m);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++) res.data[i][j] = data[i][j] * factor;
		return res;
	}
	Matrix divide(Matrix other) {
		assert n == other.n;
		assert m == other.m;
		Matrix res = new Matrix(n, m);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++) res.data[i][j] = data[i][j] / other.data[i][j];
		return res;
	}
	Matrix divide(double factor) {
		Matrix res = new Matrix(n, m);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++) res.data[i][j] = data[i][j] / factor;
		return res;
	}
	Matrix add(Matrix other, double alpha) {
		assert n == other.n;
		assert m == other.m;
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
	void applyInPlace(Function<Double, Double> func) {
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++) data[i][j] = func.apply(data[i][j]);
	}
	void fill(Supplier<Double> func) {
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++) data[i][j] = func.get();
	}
	Matrix concatRow(List<Double> row) {
		assert row.size() == m;
		Matrix res = new Matrix(n + 1, m);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++) res.data[i][j] = data[i][j];
		for (int j = 0; j < m; j++) res.data[n][j] = row.get(j);
		return res;
	}
	public String toString() {
		return "[" + String.join(", ", Arrays.stream(data).map(Arrays::toString).collect(Collectors.toList())) +"]";
	}
}