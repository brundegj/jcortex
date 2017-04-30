/*
 * James Brundege
 * Date: 2017-04-30
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.optimizationstrategies;

import jmb.jcortex.data.SynMatrix;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

public class FixedLearningRateOptimizationStrategyTest {

    @Test
    public void calcCorrections_MultipliesByLearningRateConstant() {
        List<SynMatrix> gradients = Arrays.asList(
                new SynMatrix(new double[][]{
                        {1, 2},
                        {3, 4}
                }),
                new SynMatrix(new double[][]{
                        {5, 6, 7},
                        {9, 10, 11}
                })
        );
        List<SynMatrix> expectedCorrections = Arrays.asList(
                new SynMatrix(new double[][]{
                        {2, 4},
                        {6, 8}
                }),
                new SynMatrix(new double[][]{
                        {10, 12, 14},
                        {18, 20, 22}
                })
        );

        FixedLearningRateOptimizationStrategy optimizationStrategy = new FixedLearningRateOptimizationStrategy(2);
        List<SynMatrix> actualCorrections = optimizationStrategy.calcCorrections(gradients);
        assertThat(actualCorrections).isEqualTo(expectedCorrections);
    }

}