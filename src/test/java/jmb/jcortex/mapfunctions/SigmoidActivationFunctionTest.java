/*
 * James Brundege
 * Date: 2017-04-16
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.mapfunctions;

import org.assertj.core.data.Offset;
import org.junit.Test;

import java.util.function.DoubleUnaryOperator;

import static org.assertj.core.api.Assertions.assertThat;

public class SigmoidActivationFunctionTest {
    private Offset<Double> precision = Offset.offset(0.0001);

    public SigmoidActivationFunction sigmoid = new SigmoidActivationFunction();

    @Test
    public void testFunction() {
        DoubleUnaryOperator function = sigmoid.getFunction();
        assertThat(function.applyAsDouble(-10)).isEqualTo(0, precision);
        assertThat(function.applyAsDouble(-2)).isEqualTo(0.1192, precision);
        assertThat(function.applyAsDouble(-1)).isEqualTo(0.2689, precision);
        assertThat(function.applyAsDouble(0)).isEqualTo(0.5, precision);
        assertThat(function.applyAsDouble(1)).isEqualTo(0.7311, precision);
        assertThat(function.applyAsDouble(2)).isEqualTo(0.8808, precision);
        assertThat(function.applyAsDouble(10)).isEqualTo(1, precision);
    }

    @Test
    public void testDerivative() {
        DoubleUnaryOperator derivative = sigmoid.getDerivative();
        assertThat(derivative.applyAsDouble(-10)).isEqualTo(-110, precision);
        assertThat(derivative.applyAsDouble(-2)).isEqualTo(-6, precision);
        assertThat(derivative.applyAsDouble(-1)).isEqualTo(-2, precision);
        assertThat(derivative.applyAsDouble(0)).isEqualTo(0, precision);
        assertThat(derivative.applyAsDouble(1)).isEqualTo(0, precision);
        assertThat(derivative.applyAsDouble(2)).isEqualTo(-2, precision);
        assertThat(derivative.applyAsDouble(10)).isEqualTo(-90, precision);
    }

}