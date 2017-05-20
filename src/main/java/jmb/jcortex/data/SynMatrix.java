package jmb.jcortex.data;

import jmb.jcortex.mapfunctions.MatrixFunction;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

import static java.lang.String.format;
import static org.jblas.ranges.RangeUtils.interval;

/**
 * A wrapper class around the jblas DoubleMatrix class. This abstracts the particular linear algebra library
 * used, and adds lots of convenience methods useful for neural nets.
 * <p>
 * James Brundege
 * Date: 2017-04-09
 * MIT license: https://opensource.org/licenses/MIT
 */
public class SynMatrix implements Copyable<SynMatrix>, Serializable {

    public static SynMatrix ones(int numRows, int numCols) {
   		return new SynMatrix(DoubleMatrix.ones(numRows, numCols));
   	}

    private DoubleMatrix internalMatrix;

    /**
     * Creates a column vector with the given values.
     */
    public SynMatrix(double... data) {
        internalMatrix = new DoubleMatrix(data);
    }

    /**
     * Creates a matrix with the given values.
     */
    public SynMatrix(double[][] data) {
        internalMatrix = new DoubleMatrix(data);
    }

    protected SynMatrix(DoubleMatrix internalMatrix) {
        this.internalMatrix = internalMatrix;
    }

    /**
     * Creates a matrix of the given size, filled with zeros.
     */
    public SynMatrix(int rows, int cols) {
        this.internalMatrix = new DoubleMatrix(rows, cols);
    }

    public SynMatrix(int rows, int cols, double fillValue) {
        this.internalMatrix = new DoubleMatrix(rows, cols);
        Arrays.fill(internalMatrix.data, fillValue);
    }

    public int numCols() {
        return internalMatrix.getColumns();
    }

    public int numRows() {
        return internalMatrix.getRows();
    }

    public double[] getAll() {
        return internalMatrix.toArray();
    }

    @Override
    public SynMatrix copy() {
        return new SynMatrix(internalMatrix.dup());
    }

    public double[] getRow(int rowNum) {
        return internalMatrix.getRow(rowNum).toArray();
    }

    public DoubleStream getRowStream(int rowNum) {
        return Arrays.stream(getRow(rowNum));
    }

    public double[] getCol(int colNum) {
        return internalMatrix.getColumn(colNum).toArray();
    }

    public DoubleStream getColStream(int colNum) {
        return Arrays.stream(getCol(colNum));
    }

    public Stream<double[]> getStreamOfRows() {
        return Arrays.stream(internalMatrix.toArray2());
    }

    public Stream<double[]> getStreamOfCols() {
        return this.transpose().getStreamOfRows();
    }

    public SynMatrix sliceRows(int startRow, int endRow) {
        return new SynMatrix(internalMatrix.getRows(new IntervalRange(startRow, endRow)));
    }

    public double[][] getData() {
        return internalMatrix.toArray2();
    }

    public List<double[]> getRows() {
        return Arrays.asList(internalMatrix.toArray2());
    }

    /**
     * Apply (1 - value) to every value in the matrix.
     * @return A new SynMatrix with the resulting values. The original SynMatrix is not altered.
     */
    public SynMatrix oneMinusThis() {
        DoubleMatrix result = DoubleMatrix.ones(numRows(), numCols()).sub(internalMatrix);
        return new SynMatrix(result);
    }

    /**
     * Add a column of ones to a copy of the current matrix. Does not modify the current matrix.
     * @return A new SynMatrix that is a copy of the current matrix prepended by a column of ones.
     */
    public SynMatrix addBiasColumn() {
        return addBiasColumn(SynMatrix.ones(internalMatrix.getRows(), 1));
    }

    /**
     * Add a column of the given values to a copy of the current matrix. Does not modify the current matrix.
     * @return A new SynMatrix that is a copy of the current matrix prepended by the given bias column.
     */
    public SynMatrix addBiasColumn(SynMatrix biasColumn) {
        if (biasColumn.numCols() != 1) throw new IllegalArgumentException("Wrong number of columns: " + biasColumn.numCols());
        return new SynMatrix(DoubleMatrix.concatHorizontally(biasColumn.internalMatrix, this.internalMatrix));
    }

    /**
     * Add a row of ones to a copy of the current matrix. Does not modify the current matrix.
     * @return A new SynMatrix that is a copy of the current matrix prepended by a row of ones.
     */
    public SynMatrix addBiasRow() {
        return addBiasRow(SynMatrix.ones(1, internalMatrix.getColumns()));
    }

    /**
     * Add a row of the given values to a copy of the current matrix. Does not modify the current matrix.
     * @return A new SynMatrix that is a copy of the current matrix prepended by the given bias row.
     */
    public SynMatrix addBiasRow(SynMatrix biasRow) {
        if (biasRow.numRows() != 1) throw new IllegalArgumentException("Wrong number of rows: " + biasRow.numRows());
        return new SynMatrix(DoubleMatrix.concatVertically(biasRow.internalMatrix, this.internalMatrix));
    }

    public SynMatrix getBiasRow() {
        return this.extractMatrix(0, 1, 0, this.numCols());
    }

    public SynMatrix removeBiasRow() {
        if (this.isVector()) {
            // for vectors, remove the first item
            return this.extractMatrix(1, this.numRows(), 0, this.numCols());
        } else {
            // for matrices, remove the first row
            return this.extractMatrix(1, this.numRows(), 0, this.numCols());
        }
    }

    public SynMatrix getBiasColumn() {
        return this.extractMatrix(0, this.numRows(), 0, 1);
    }

    @SuppressWarnings("UnusedDeclaration")
    public SynMatrix removeBiasColumn() {
        if (this.isVector()) {
            // for vectors, remove the first item
            return this.extractMatrix(1, this.numRows(), 0, this.numCols());
        } else {
            // for matrices, remove the first column
            return this.extractMatrix(0, this.numRows(), 1, this.numCols());
        }
    }

    /**
     * Extract a sub matrix from the this matrix. Indexes are 0-based, and they start inclusive and end exclusive.
     * This is the same standard used to create substrings and subarrays in Java.
     *
     * Example:
     * [1,  2,  3,  4 ]
     * [5,  6,  7,  8 ]
     * [9,  10, 11, 12]
     * [13, 14, 15, 16]
     *
     * synMatrix.extractMatrix(1, 3, 1, 3)
     *
     * returns:
     * [6,  7 ]
     * [10, 11]
     */
    public SynMatrix extractMatrix(int startRowInclusive, int endRowExclusive, int startColumnInclusive, int endColumnExclusive) {
        return new SynMatrix(internalMatrix.get(interval(startRowInclusive, endRowExclusive), interval(startColumnInclusive, endColumnExclusive)));
    }

    public boolean isVector() {
        return numRows() == 1 || numCols() == 1;
    }

//    public SynMatrix square() {
//        return new SynMatrix(internalMatrix.mul(internalMatrix));
//    }
//
//    public SynMatrix squareInPlace() {
//        double[] data = internalMatrix.data;
//        for (int i = 0; i < data.length; i++) {
//            data[i] = data[i] * data[i];
//        }
//        return this;
//    }

    public SynMatrix plus(SynMatrix matrix) {
        assertSameSize(matrix);
        return new SynMatrix(internalMatrix.add(matrix.internalMatrix));
    }

    public SynMatrix plusInPlace(SynMatrix matrix) {
        assertSameSize(matrix);
        internalMatrix.addi(matrix.internalMatrix);
        return this;
    }

    private void assertSameSize(SynMatrix other) {
        if (this.numRows() != other.numRows() || this.numCols() != other.numCols()) {
            throw new IllegalArgumentException(
                    format("Matrices must have same dimensions. This matrix is %s x %s, the passed matrix is %s x %s",
                            this.numRows(), this.numCols(), other.numRows(), other.numCols()));
        }
    }

    public SynMatrix minus(SynMatrix matrix) {
        assertSameSize(matrix);
        return new SynMatrix(internalMatrix.sub(matrix.internalMatrix));
    }

    public SynMatrix minusInPlace(SynMatrix matrix) {
        assertSameSize(matrix);
        internalMatrix.subi(matrix.internalMatrix);
        return this;
    }

    public SynMatrix elementMult(SynMatrix matrix) {
        assertSameSize(matrix);
        return new SynMatrix(internalMatrix.mul(matrix.internalMatrix));
    }

    public SynMatrix elementMultInPlace(SynMatrix matrix) {
        assertSameSize(matrix);
        internalMatrix.muli(matrix.internalMatrix);
        return this;
    }

    public SynMatrix elementMult(double value) {
        return new SynMatrix(internalMatrix.mul(value));
    }

    public SynMatrix elementMultInPlace(double value) {
        internalMatrix.muli(value);
        return this;
    }

    public SynMatrix multiply(SynMatrix matrix) {
        return new SynMatrix(internalMatrix.mmul(matrix.internalMatrix));
    }

    @SuppressWarnings("UnusedDeclaration")
    public SynMatrix elementDivide(double value) {
        return new SynMatrix(internalMatrix.div(value));
    }

    public SynMatrix elementDivideInPlace(double value) {
        internalMatrix.divi(value);
        return this;
    }

    public SynMatrix transpose() {
        return new SynMatrix(internalMatrix.transpose());
    }

    /**
     * Get a value from the matrix by row and col index.
     */
    public double get(int row, int col) {
        return internalMatrix.get(row, col);
    }

    /**
     * Mutable Setter*. Set the given value on the matrix at the given row/col
     */
    public void set(int row, int col, double value) {
        internalMatrix.put(row, col, value);
    }

    /**
     * Get the value at the given linear index of the matrix. The linear index is 0 to numElements()-1
     */
    public double get(int index) {
        return internalMatrix.get(index);
    }

    /**
     * *Mutable Setter*. Set the given value on the matrix at the given linear index. The linear index is 0 to numElements()-1
     */
    public void set(int index, double value) {
        internalMatrix.put(index, value);
    }

    public void setRow(int rowIndex, double[] row) {
        internalMatrix.putRow(rowIndex, new DoubleMatrix(row));
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        SynMatrix synMatrix = (SynMatrix) o;

        // Compare the two matrices allowing the indicated tolerace for rounding differences
        return internalMatrix.compare(synMatrix.internalMatrix, 0.000000000000001);
    }

    @Override
    public int hashCode() {
        return internalMatrix.hashCode();
    }

    public int numElements() {
        return internalMatrix.getLength();
    }

    public SynMatrix apply(DoubleUnaryOperator function) {
   		return this.copy().applyInPlace(function);
   	}

   	public SynMatrix applyInPlace(DoubleUnaryOperator function) {
   		for (int i = 0; i < internalMatrix.data.length; i++) {
   			internalMatrix.data[i] = function.applyAsDouble(internalMatrix.data[i]);
   		}
   		return this;
   	}

    public SynMatrix apply(MatrixFunction matrixFunction) {
        return matrixFunction.apply(this);
    }

    /**
   	 * Returns a vector of the sum of the cells in each column. Vector size is numCols x 1;
   	 */
   	public SynMatrix sumColumns() {
   	    double[] colSums = new double[numCols()];
        for (int i = 0; i < colSums.length; i++) {
            colSums[i] = getColStream(i).sum();
        }
        return new SynMatrix(colSums);
   	}

   	/**
   	 * Returns a vector of the sum of the cells in each row. Vector size is numRows x 1;
   	 */
    public SynMatrix sumRows() {
   	    double[] rowSums = new double[numRows()];
        for (int i = 0; i < rowSums.length; i++) {
            rowSums[i] = getRowStream(i).sum();
        }
        return new SynMatrix(rowSums);
   	}

    /**
   	 * Returns a vector of the mean value of the cells in each column. Vector size is numCols x 1;
   	 */
   	public SynMatrix getColMeans() {
   	    return sumColumns().applyInPlace(x -> x/numRows());
   	}

   	/**
   	 * Returns a vector of the mean value of the cells in each row. Vector size is numRows x 1;
   	 */
    public SynMatrix getRowMeans() {
        return sumRows().applyInPlace(x -> x/numCols());
   	}

   	@Override
    public String toString() {
        return internalMatrix.toString();
    }
}
