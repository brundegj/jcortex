/*
 * James Brundege
 * Date: 2017-04-16
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.mapfunctions;

import jmb.jcortex.data.SynMatrix;
import org.assertj.core.data.Offset;
import org.junit.Test;

import java.util.Arrays;

import static org.assertj.core.api.Assertions.assertThat;

public class SoftMaxActivationFunctionTest {
    private Offset<Double> precision = Offset.offset(0.000000000000001);

    public SoftMaxActivationFunction softmax = new SoftMaxActivationFunction();

    private double[] input = new double[]{3, 0.5, 0, -2, 1.7, -1.3};
    private double[] expectedOutput = new double[]{0.7018970478761761, 0.05761521820903455, 0.034945396309813054, 0.00472934510740423, 0.1912892609829745, 0.009523731514597518};

    @Test
    public void testApplyRow() {
        assertThat(softmax.apply(input)).containsExactly(expectedOutput, precision);
        assertThat(Arrays.stream(softmax.apply(input)).sum()).isEqualTo(1.0);
    }

    @Test
    public void testApplyMatrix() {
        double[][] input2d = new double[][]{
                input
        };
        SynMatrix originalMatrix = new SynMatrix(input2d);

        SynMatrix result = originalMatrix.apply(softmax);

        assertThat(result.numRows()).isEqualTo(1);
        assertThat(result.numCols()).isEqualTo(6);
        assertThat(result.getRow(0)).containsExactly(expectedOutput, precision);
        assertThat(Arrays.stream(result.getAll()).sum()).isEqualTo(1.0);
    }
}