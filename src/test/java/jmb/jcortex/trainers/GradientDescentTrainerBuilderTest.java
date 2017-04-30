/*
 * James Brundege
 * Date: 2017-04-19
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.trainers;

import jmb.jcortex.strategies.batchingstrategies.BatchingStrategy;
import jmb.jcortex.strategies.haltingstrategies.HaltingStrategy;
import jmb.jcortex.strategies.optimizationstrategies.OptimizationStrategy;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import static org.assertj.core.api.Assertions.assertThat;

@RunWith(MockitoJUnitRunner.class)
public class GradientDescentTrainerBuilderTest {

    @Mock
    private BatchingStrategy batchingStrategy;
    @Mock
    private OptimizationStrategy optimizationStrategy;
    @Mock
    private HaltingStrategy haltingStrategy;

    @Test
    public void testBuilder() {
        GradientDescentTrainer trainer = GradientDescentTrainerBuilder.createTrainer()
                .withBatchingStrategy(batchingStrategy)
                .withHaltingStrategy(haltingStrategy)
                .withOptimizationStrategy(optimizationStrategy)
                .build();

        assertThat(trainer.getBatchingStrategy()).isSameAs(batchingStrategy);
        assertThat(trainer.getHaltingStrategy()).isSameAs(haltingStrategy);
        assertThat(trainer.getOptimizationStrategy()).isSameAs(optimizationStrategy);
    }

}