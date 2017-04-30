/*
 * James Brundege
 * Date: 2017-04-19
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.optimizationstrategies;

import jmb.jcortex.data.SynMatrix;
import org.assertj.core.data.Offset;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

public class MomentumOptimizationStrategyTest {
    private Offset<Double> precision = Offset.offset(0.000000000000001);

    @Test
    public void calcCorrections_AddsPreviousCorrections() {
        MomentumOptimizationStrategy momentumOptimizationStrategy = new MomentumOptimizationStrategy(0.1, 0.5);

        double[][] gradientValues = new double[][] {
                {0.1, -0.2},
                {0.2, -0.1}
        };
        SynMatrix layer1Gradients = new SynMatrix(gradientValues);
        SynMatrix layer2Gradients = new SynMatrix(gradientValues);

        // 1st round, use gradients and learning rate only to calc corrections
        List<SynMatrix> corrections = momentumOptimizationStrategy.calcCorrections(Arrays.asList(layer1Gradients, layer2Gradients));
        assertThat(corrections).hasSize(2);
        SynMatrix layer1Corrections = corrections.get(0);
        SynMatrix layer2Corrections = corrections.get(1);

        assertThat(layer1Corrections.getRow(0)).containsExactly(new double[]{0.01, -0.02}, precision);
        assertThat(layer1Corrections.getRow(1)).containsExactly(new double[]{0.02, -0.01}, precision);
        assertThat(layer2Corrections.getRow(0)).containsExactly(new double[]{0.01, -0.02}, precision);
        assertThat(layer2Corrections.getRow(1)).containsExactly(new double[]{0.02, -0.01}, precision);

        // 2nd round, add previous corrections x momentum constant to current corrections
        List<SynMatrix> corrections2 = momentumOptimizationStrategy.calcCorrections(Arrays.asList(layer1Gradients, layer2Gradients));
        assertThat(corrections2).hasSize(2);
        SynMatrix layer1Corrections2 = corrections2.get(0);
        SynMatrix layer2Corrections2 = corrections2.get(1);

        assertThat(layer1Corrections2.getRow(0)).containsExactly(new double[]{0.015, -0.03}, precision);
        assertThat(layer1Corrections2.getRow(1)).containsExactly(new double[]{0.03, -0.015}, precision);
        assertThat(layer2Corrections2.getRow(0)).containsExactly(new double[]{0.015, -0.03}, precision);
        assertThat(layer2Corrections2.getRow(1)).containsExactly(new double[]{0.03, -0.015}, precision);
    }

}