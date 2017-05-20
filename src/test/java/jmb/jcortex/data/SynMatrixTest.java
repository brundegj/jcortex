/*
 * James Brundege
 * Date: 2017-04-15
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.data;

import jmb.jcortex.mapfunctions.DoublingActivationFunction;
import org.assertj.core.data.Offset;
import org.junit.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

public class SynMatrixTest {
    // The precision required for double calculations
    private Offset<Double> precision = Offset.offset(0.00000000001);

    @Test
    public void testConstructColumnVector() {
        double[] values = {1.0, 2.0, 3.0};
        SynMatrix synMatrix = new SynMatrix(values);
        assertThat(synMatrix.numRows()).isEqualTo(3);
        assertThat(synMatrix.numCols()).isEqualTo(1);
        assertThat(synMatrix.getCol(0)).containsExactly(values);
    }

    @Test
    public void testConstructEmptyMatrix() {
        SynMatrix synMatrix = new SynMatrix(3, 4);
        assertThat(synMatrix.numRows()).isEqualTo(3);
        assertThat(synMatrix.numCols()).isEqualTo(4);
        synMatrix.getStreamOfRows().forEach(row -> assertThat(row).containsExactly(0, 0, 0, 0));
    }

    @Test
    public void testConstructMatrix() {
        double[][] values = new double[][] {
                {1, 2},
                {3, 4},
                {5, 6}
        };
        SynMatrix synMatrix = new SynMatrix(values);
        assertThat(synMatrix.numRows()).isEqualTo(3);
        assertThat(synMatrix.numCols()).isEqualTo(2);
        assertThat(synMatrix.getRow(0)).containsExactly(1, 2);
        assertThat(synMatrix.getRow(1)).containsExactly(3, 4);
        assertThat(synMatrix.getRow(2)).containsExactly(5, 6);
    }

    @Test
    public void copy_ReturnsCopyWithNoSharedState() {
        SynMatrix original = new SynMatrix(2, 2);
        SynMatrix copy = original.copy();
        assertThat(copy).isNotSameAs(original);
        assertThat(copy).isEqualTo(original);
        copy.set(0, 0, 1.0);
        copy.set(0, 1, 2.0);
        copy.set(1, 0, 3.0);
        copy.set(1, 1, 4.0);
        assertThat(original.getAll()).containsExactly(0, 0, 0, 0);
        assertThat(copy.getAll()).containsExactly(1, 3, 2, 4);
    }

    @Test
    public void sliceRows_ReturnsASubset() {
        double[][] values = new double[][] {
                {1, 2},
                {3, 4},
                {5, 6},
                {7, 8},
                {9, 10},
        };
        SynMatrix synMatrix = new SynMatrix(values);
        SynMatrix subset = synMatrix.sliceRows(1, 4);
        assertThat(subset.numRows()).isEqualTo(3);
        assertThat(subset.numCols()).isEqualTo(2);
        assertThat(subset.getRow(0)).containsExactly(3, 4);
        assertThat(subset.getRow(1)).containsExactly(5, 6);
        assertThat(subset.getRow(2)).containsExactly(7, 8);
    }

    @Test
    public void getRows_ReturnsAListOfArrays() {
        double[][] values = new double[][] {
                {1, 2},
                {3, 4},
                {5, 6}
        };
        SynMatrix synMatrix = new SynMatrix(values);
        List<double[]> rows = synMatrix.getRows();
        assertThat(rows).hasSize(3);
        assertThat(rows.get(0)).containsExactly(1, 2);
        assertThat(rows.get(1)).containsExactly(3, 4);
        assertThat(rows.get(2)).containsExactly(5, 6);
    }

    @Test
    public void getData_Returns2DArray() {
        double[][] values = new double[][] {
                {1, 2},
                {3, 4},
                {5, 6}
        };
        SynMatrix synMatrix = new SynMatrix(values);
        double[][] data = synMatrix.getData();
        assertThat(data[0]).containsExactly(values[0]);
        assertThat(data[1]).containsExactly(values[1]);
        assertThat(data[2]).containsExactly(values[2]);
    }

    @Test
    public void oneMinusThis_SubtractsEveryValueFromOne() {
        double[][] values = new double[][] {
                {-1.5, 0.5},
                {1.3, 2.7}
        };
        SynMatrix synMatrix = new SynMatrix(values);
        SynMatrix result = synMatrix.oneMinusThis();
        assertThat(result.getRow(0)).containsExactly(new double[]{2.5, 0.5}, precision);
        assertThat(result.getRow(1)).containsExactly(new double[]{-0.3, -1.7}, precision);
    }

    @Test
    public void addBiasColumn_WithNoParams_PrependsColumnOfOnes() {
        double[][] values = new double[][] {
                {1, 2},
                {3, 4}
        };
        SynMatrix synMatrix = new SynMatrix(values);
        SynMatrix withBiasCol = synMatrix.addBiasColumn();
        assertThat(withBiasCol.numRows()).isEqualTo(2);
        assertThat(withBiasCol.numCols()).isEqualTo(3);
        assertThat(withBiasCol.getRow(0)).containsExactly(1, 1, 2);
        assertThat(withBiasCol.getRow(1)).containsExactly(1, 3, 4);
        assertThat(withBiasCol.getBiasColumn().getAll()).containsExactly(1, 1);
    }

    @Test
    public void addBiasColumn_WithParam_PrependsGivenColumn() {
        double[][] values = new double[][] {
                {1, 2},
                {3, 4}
        };
        SynMatrix synMatrix = new SynMatrix(values);
        SynMatrix biasCol = new SynMatrix(6.0, 7.0);
        SynMatrix withBiasCol = synMatrix.addBiasColumn(biasCol);
        assertThat(withBiasCol.numRows()).isEqualTo(2);
        assertThat(withBiasCol.numCols()).isEqualTo(3);
        assertThat(withBiasCol.getRow(0)).containsExactly(6, 1, 2);
        assertThat(withBiasCol.getRow(1)).containsExactly(7, 3, 4);
        assertThat(withBiasCol.getBiasColumn().getAll()).containsExactly(6, 7);
    }

    @Test
    public void addBiasRow_WithNoParams_PrependsRowOfOnes() {
        double[][] values = new double[][] {
                {1, 2},
                {3, 4}
        };
        SynMatrix synMatrix = new SynMatrix(values);
        SynMatrix withBiasRow = synMatrix.addBiasRow();
        assertThat(withBiasRow.numRows()).isEqualTo(3);
        assertThat(withBiasRow.numCols()).isEqualTo(2);
        assertThat(withBiasRow.getRow(0)).containsExactly(1, 1);
        assertThat(withBiasRow.getRow(1)).containsExactly(1, 2);
        assertThat(withBiasRow.getRow(2)).containsExactly(3, 4);
        assertThat(withBiasRow.getBiasRow().getAll()).containsExactly(1, 1);
    }

    @Test
    public void addBiasRow_WithParam_PrependsGivenRow() {
        double[][] values = new double[][] {
                {1, 2},
                {3, 4}
        };
        SynMatrix synMatrix = new SynMatrix(values);
        SynMatrix biasRow = new SynMatrix(6.0, 7.0).transpose();
        SynMatrix withBiasRow = synMatrix.addBiasRow(biasRow);
        assertThat(withBiasRow.numRows()).isEqualTo(3);
        assertThat(withBiasRow.numCols()).isEqualTo(2);
        assertThat(withBiasRow.getRow(0)).containsExactly(6, 7);
        assertThat(withBiasRow.getRow(1)).containsExactly(1, 2);
        assertThat(withBiasRow.getRow(2)).containsExactly(3, 4);
        assertThat(withBiasRow.getBiasRow().getAll()).containsExactly(6, 7);
    }

    @Test
    public void removeBiasColumn_RemovesFirstColumn() {
        double[][] values = new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        };
        SynMatrix synMatrix = new SynMatrix(values);
        SynMatrix withoutBiasCol = synMatrix.removeBiasColumn();
        assertThat(withoutBiasCol.numRows()).isEqualTo(2);
        assertThat(withoutBiasCol.numCols()).isEqualTo(2);
        assertThat(withoutBiasCol.getRow(0)).containsExactly(2, 3);
        assertThat(withoutBiasCol.getRow(1)).containsExactly(5, 6);
    }

    @Test
    public void removeBiasColumn_OnVector_RemovesFirstElement() {
        double[][] values = new double[][] {
                {2}, {3}, {4}
        };
        SynMatrix synMatrix = new SynMatrix(values);
        SynMatrix withoutBiasCol = synMatrix.removeBiasColumn();
        assertThat(withoutBiasCol.numRows()).isEqualTo(2);
        assertThat(withoutBiasCol.numCols()).isEqualTo(1);
        assertThat(withoutBiasCol.getRow(0)).containsExactly(3);
        assertThat(withoutBiasCol.getRow(1)).containsExactly(4);
    }

    @Test
    public void removeBiasRow_RemovesFirstRow() {
        double[][] values = new double[][] {
                {1, 2},
                {3, 4},
                {5, 6}
        };
        SynMatrix synMatrix = new SynMatrix(values);
        SynMatrix withoutBiasRow = synMatrix.removeBiasRow();
        assertThat(withoutBiasRow.numRows()).isEqualTo(2);
        assertThat(withoutBiasRow.numCols()).isEqualTo(2);
        assertThat(withoutBiasRow.getRow(0)).containsExactly(3, 4);
        assertThat(withoutBiasRow.getRow(1)).containsExactly(5, 6);
    }

    @Test
    public void removeBiasRow_OnVector_RemovesFirstElement() {
        double[][] values = new double[][] {
                {2}, {3}, {4}
        };
        SynMatrix synMatrix = new SynMatrix(values);
        SynMatrix withoutBiasRow = synMatrix.removeBiasRow();
        assertThat(withoutBiasRow.numRows()).isEqualTo(2);
        assertThat(withoutBiasRow.numCols()).isEqualTo(1);
        assertThat(withoutBiasRow.getRow(0)).containsExactly(3);
        assertThat(withoutBiasRow.getRow(1)).containsExactly(4);
    }

    @Test
    public void extractMatrix_ReturnsSubsetMatrix() {
        double[][] values = new double[][]{
                {1,  2,  3,  4,  5 },
                {6,  7,  8,  9,  10},
                {11, 12, 13, 14, 15},
                {16, 17, 18, 19, 20},
                {21, 22, 23, 24, 25}
        };
        SynMatrix synMatrix = new SynMatrix(values);
        SynMatrix subMatrix = synMatrix.extractMatrix(1, 4, 1, 4);
        assertThat(subMatrix.numRows()).isEqualTo(3);
        assertThat(subMatrix.numCols()).isEqualTo(3);
        assertThat(subMatrix.getRow(0)).containsExactly(7, 8, 9);
        assertThat(subMatrix.getRow(1)).containsExactly(12, 13, 14);
        assertThat(subMatrix.getRow(2)).containsExactly(17, 18, 19);
    }

    @Test
    public void testIsVector() {
        assertThat(new SynMatrix(1, 1).isVector()).isTrue();
        assertThat(new SynMatrix(1, 7).isVector()).isTrue();
        assertThat(new SynMatrix(7, 1).isVector()).isTrue();
        assertThat(new SynMatrix(2, 2).isVector()).isFalse();
        assertThat(new SynMatrix(2, 7).isVector()).isFalse();
    }

    @Test
    public void plusInPlace_AddsCorrespondingValuesOfTwoMatricesOfSameDimension() {
        double[][] values1 = new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        };
        SynMatrix synMatrix1 = new SynMatrix(values1);
        double[][] values2 = new double[][] {
                {3, -2, 1.5},
                {-2.2, 5, 7}
        };
        SynMatrix synMatrix2 = new SynMatrix(values2);

        SynMatrix result = synMatrix1.plusInPlace(synMatrix2);
        assertThat(result).isSameAs(synMatrix1);    // plus in place mutates the original matrix
        assertThat(result.numRows()).isEqualTo(2);
        assertThat(result.numCols()).isEqualTo(3);  // no change in dimensions
        assertThat(result.getRow(0)).containsExactly(new double[]{4, 0, 4.5}, precision);
        assertThat(result.getRow(1)).containsExactly(new double[]{1.8, 10, 13}, precision);
    }

    @Test
    public void minusInPlace_SubtractsCorrespondingValuesOfTwoMatricesOfSameDimension() {
        double[][] values1 = new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        };
        SynMatrix synMatrix1 = new SynMatrix(values1);
        double[][] values2 = new double[][] {
                {3, -2, 1.5},
                {-2.2, 5, 7}
        };
        SynMatrix synMatrix2 = new SynMatrix(values2);

        SynMatrix result = synMatrix1.minusInPlace(synMatrix2);
        assertThat(result).isSameAs(synMatrix1);    // plus in place mutates the original matrix
        assertThat(result.numRows()).isEqualTo(2);
        assertThat(result.numCols()).isEqualTo(3);  // no change in dimensions
        assertThat(result.getRow(0)).containsExactly(new double[]{-2, 4, 1.5}, precision);
        assertThat(result.getRow(1)).containsExactly(new double[]{6.2, 0, -1}, precision);
    }

    @Test
    public void elementMult_WithMatrix_MultipliesEachCorrespondingElement() {
        // NOTE: This is for scalar multiplication on each element.
        // Matrix multiplication is handled by SynMatrix.multiply()

        double[][] values1 = new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        };
        SynMatrix synMatrix1 = new SynMatrix(values1);
        double[][] values2 = new double[][] {
                {3, -2, 1.5},
                {-2.2, 5, 7}
        };
        SynMatrix synMatrix2 = new SynMatrix(values2);

        SynMatrix result = synMatrix1.elementMult(synMatrix2);
        assertThat(result.numRows()).isEqualTo(2);
        assertThat(result.numCols()).isEqualTo(3);  // no change in dimensions
        assertThat(result.getRow(0)).containsExactly(new double[]{3, -4, 4.5}, precision);
        assertThat(result.getRow(1)).containsExactly(new double[]{-8.8, 25, 42}, precision);
    }

    @Test
    public void elementMult_WithNumber_MultipliesEachElementByTheNumber() {
        double[][] values1 = new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        };
        SynMatrix synMatrix = new SynMatrix(values1);

        SynMatrix result = synMatrix.elementMult(2);
        assertThat(result.numRows()).isEqualTo(2);
        assertThat(result.numCols()).isEqualTo(3);  // no change in dimensions
        assertThat(result.getRow(0)).containsExactly(new double[]{2, 4, 6}, precision);
        assertThat(result.getRow(1)).containsExactly(new double[]{8, 10, 12}, precision);
    }

    @Test
    public void elementDivide_WithNumber_DividesEachElementByTheNumber() {
        double[][] values1 = new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        };
        SynMatrix synMatrix = new SynMatrix(values1);

        SynMatrix result = synMatrix.elementDivide(2);

        assertThat(result.numRows()).isEqualTo(2);
        assertThat(result.numCols()).isEqualTo(3);  // no change in dimensions
        assertThat(result.getRow(0)).containsExactly(new double[]{0.5, 1, 1.5}, precision);
        assertThat(result.getRow(1)).containsExactly(new double[]{2, 2.5, 3}, precision);
    }

    @Test
    public void elementMath_ThrowsException_IfMatricesHaveDifferentDimensions() {
        double[][] values1 = new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        };
        SynMatrix synMatrix1 = new SynMatrix(values1);
        double[][] values2 = new double[][] {
                {1, 2},
                {3, 4},
                {5, 6}
        };
        SynMatrix synMatrix2 = new SynMatrix(values2);

        String exceptionMessage = "Matrices must have same dimensions";
        assertThatThrownBy(() -> synMatrix1.plus(synMatrix2))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining(exceptionMessage);
        assertThatThrownBy(() -> synMatrix1.plusInPlace(synMatrix2))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining(exceptionMessage);
        assertThatThrownBy(() -> synMatrix1.minus(synMatrix2))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining(exceptionMessage);
        assertThatThrownBy(() -> synMatrix1.minusInPlace(synMatrix2))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining(exceptionMessage);
        assertThatThrownBy(() -> synMatrix1.elementMult(synMatrix2))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining(exceptionMessage);
        assertThatThrownBy(() -> synMatrix1.elementMultInPlace(synMatrix2))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining(exceptionMessage);
    }

    @Test
    public void multiply_DoesMatrixMultiplication() {
        double[][] values1 = new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        };
        SynMatrix synMatrix1 = new SynMatrix(values1);
        double[][] values2 = new double[][] {
                {2, 3},
                {4, 5},
                {6, 7}
        };
        SynMatrix synMatrix2 = new SynMatrix(values2);

        SynMatrix result = synMatrix1.multiply(synMatrix2);

        assertThat(result.numRows()).isEqualTo(2);
        assertThat(result.numCols()).isEqualTo(2);
        assertThat(result.numElements()).isEqualTo(4);
        assertThat(result.getRow(0)).containsExactly(new double[]{28, 34}, precision);
        assertThat(result.getRow(1)).containsExactly(new double[]{64, 79}, precision);
    }

    @Test
    public void transpose_FlipsDimensions() {
        double[][] values = new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        };
        SynMatrix synMatrix = new SynMatrix(values);

        SynMatrix transposed = synMatrix.transpose();

        assertThat(transposed.numRows()).isEqualTo(3);
        assertThat(transposed.numCols()).isEqualTo(2);
        assertThat(transposed.getRow(0)).containsExactly(1, 4);
        assertThat(transposed.getRow(1)).containsExactly(2, 5);
        assertThat(transposed.getRow(2)).containsExactly(3, 6);
    }

    @Test
    public void set_MutatesTheMatrix() {
        double[][] values = new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        };
        SynMatrix synMatrix = new SynMatrix(values);

        synMatrix.set(0, 2.5);      // set by linear index
        synMatrix.set(1, 1, 3.5);   // set by row/col

        assertThat(synMatrix.numRows()).isEqualTo(2);
        assertThat(synMatrix.numCols()).isEqualTo(3);
        assertThat(synMatrix.getRow(0)).containsExactly(2.5, 2, 3);
        assertThat(synMatrix.getRow(1)).containsExactly(4, 3.5, 6);
    }

    @Test
    public void setRow_ReplacesRowWithNewValues() {
        double[][] values = new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        };
        SynMatrix synMatrix = new SynMatrix(values);

        synMatrix.setRow(1, new double[]{7, 8, 9});

        assertThat(synMatrix.numRows()).isEqualTo(2);
        assertThat(synMatrix.numCols()).isEqualTo(3);
        assertThat(synMatrix.getRow(0)).containsExactly(1, 2, 3);
        assertThat(synMatrix.getRow(1)).containsExactly(7, 8, 9);
    }

    @Test
    public void apply_MapsAllValues() {
        double[][] values = new double[][] {
                {1, 2},
                {3, 4}
        };
        SynMatrix synMatrix = new SynMatrix(values);

        SynMatrix result = synMatrix.apply(new DoublingActivationFunction().getFunction());

        assertThat(result.getRow(0)).containsExactly(new double[]{2, 4}, precision);
        assertThat(result.getRow(1)).containsExactly(new double[]{6, 8}, precision);
    }

    @Test
    public void sumRows_ReturnsVectorOfSums() {
        double[][] values = new double[][] {
                {1, 2},
                {3, 4},
                {5, 6}
        };
        SynMatrix synMatrix = new SynMatrix(values);
        SynMatrix sums = synMatrix.sumRows();
        assertThat(sums.numRows()).isEqualTo(3);
        assertThat(sums.numCols()).isEqualTo(1);
        assertThat(sums.get(0, 0)).isEqualTo(3);
        assertThat(sums.get(1, 0)).isEqualTo(7);
        assertThat(sums.get(2, 0)).isEqualTo(11);
    }

    @Test
    public void sumCols_ReturnsVectorOfSums() {
        double[][] values = new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        };
        SynMatrix synMatrix = new SynMatrix(values);
        SynMatrix sums = synMatrix.sumColumns();
        assertThat(sums.numRows()).isEqualTo(3);
        assertThat(sums.numCols()).isEqualTo(1);
        assertThat(sums.get(0, 0)).isEqualTo(5);
        assertThat(sums.get(1, 0)).isEqualTo(7);
        assertThat(sums.get(2, 0)).isEqualTo(9);
    }

    @Test
    public void getRowMeans_ReturnsVectorOfMeans() {
        double[][] values = new double[][] {
                {1, 2},
                {3, 4},
                {5, 6}
        };
        SynMatrix synMatrix = new SynMatrix(values);
        SynMatrix sums = synMatrix.getRowMeans();
        assertThat(sums.numRows()).isEqualTo(3);
        assertThat(sums.numCols()).isEqualTo(1);
        assertThat(sums.get(0, 0)).isEqualTo(1.5);
        assertThat(sums.get(1, 0)).isEqualTo(3.5);
        assertThat(sums.get(2, 0)).isEqualTo(5.5);
    }

    @Test
    public void getColMeans_ReturnsVectorOfMeans() {
        double[][] values = new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        };
        SynMatrix synMatrix = new SynMatrix(values);
        SynMatrix sums = synMatrix.getColMeans();
        assertThat(sums.numRows()).isEqualTo(3);
        assertThat(sums.numCols()).isEqualTo(1);
        assertThat(sums.get(0, 0)).isEqualTo(2.5);
        assertThat(sums.get(1, 0)).isEqualTo(3.5);
        assertThat(sums.get(2, 0)).isEqualTo(4.5);
    }

}