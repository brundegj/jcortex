/*
 * James Brundege
 * Date: 2017-05-14
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.mapfunctions;

import org.assertj.core.data.Offset;
import org.junit.Test;

import java.util.function.DoubleUnaryOperator;

import static org.assertj.core.api.Assertions.assertThat;

public class RectifiedLinearActivationFunctionTest {
    private Offset<Double> precision = Offset.offset(0.00000000001);

    public RectifiedLinearActivationFunction function = new RectifiedLinearActivationFunction();

    @Test
    public void testFunction() {
        DoubleUnaryOperator function = this.function.getFunction();
        assertThat(function.applyAsDouble(-1)).isEqualTo(0.0, precision);
        assertThat(function.applyAsDouble(-0.5)).isEqualTo(0.0, precision);
        assertThat(function.applyAsDouble(-0.00001)).isEqualTo(0.0, precision);
        assertThat(function.applyAsDouble(0)).isEqualTo(0.0, precision);
        assertThat(function.applyAsDouble(0.00001)).isEqualTo(0.00001, precision);
        assertThat(function.applyAsDouble(0.5)).isEqualTo(0.5, precision);
        assertThat(function.applyAsDouble(1)).isEqualTo(1, precision);
        assertThat(function.applyAsDouble(2.5)).isEqualTo(2.5, precision);
    }

    @Test
    public void testDerivative() {
        DoubleUnaryOperator derivative = function.getDerivative();
        assertThat(derivative.applyAsDouble(-1)).isEqualTo(0.0, precision);
        assertThat(derivative.applyAsDouble(-0.5)).isEqualTo(0.0, precision);
        assertThat(derivative.applyAsDouble(-0.00001)).isEqualTo(0.0, precision);
        assertThat(derivative.applyAsDouble(0)).isEqualTo(0.0, precision);
        assertThat(derivative.applyAsDouble(0.00001)).isEqualTo(1.0, precision);
        assertThat(derivative.applyAsDouble(0.5)).isEqualTo(1.0, precision);
        assertThat(derivative.applyAsDouble(1)).isEqualTo(1.0, precision);
        assertThat(derivative.applyAsDouble(2.5)).isEqualTo(1.0, precision);
    }
}