/*
 * James Brundege
 * Date: 2017-04-23
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.trainers;

import jmb.jcortex.data.SynMatrix;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

public class GradientCalculatorTest {

    @Test
    public void calcGradients_CalculatesAverageGradientAcrossRows() {
        List<SynMatrix> nodeValues = Arrays.asList(
                new SynMatrix(new double[][]{
                        {-0.1, 0.1},
                        {0.15, -0.15}
                }),
                new SynMatrix(new double[][]{
                        {0.2, -0.2},
                        {-0.3, 0.3}
                })
        );
        List<SynMatrix> deltas = Arrays.asList(
                new SynMatrix(new double[][]{
                        {-0.2, 0.2, 0.1},
                        {0.3, -0.2, -0.15}
                }),
                new SynMatrix(new double[][]{
                        {-0.15, -0.2, 0.3},
                        {0.15, 0.3, -0.1}
                })
        );

        List<SynMatrix> expectedGradients = Arrays.asList(
                new SynMatrix(new double[][]{
                        {0.05, 0.0, -0.025},
                        {0.0325, -0.025, -0.01625},
                        {-0.0325, 0.025, 0.01625}
                }),
                new SynMatrix(new double[][]{
                        {0.0, 0.05, 0.1},
                        {-0.0375, -0.065, 0.045},
                        {0.0375, 0.065, -0.045}
                })
        );

        List<SynMatrix> actualGradients = new GradientCalculator().calcGradients(deltas, nodeValues);

        assertThat(actualGradients).isEqualTo(expectedGradients);
    }

}