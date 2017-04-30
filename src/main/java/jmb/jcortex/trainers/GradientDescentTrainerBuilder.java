/*
 * James Brundege
 * Date: 2017-04-09
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.trainers;

import jmb.jcortex.strategies.batchingstrategies.BatchingStrategy;
import jmb.jcortex.strategies.batchingstrategies.FullTrainingSetBatchingStrategy;
import jmb.jcortex.strategies.haltingstrategies.HaltingStrategy;
import jmb.jcortex.strategies.optimizationstrategies.MomentumOptimizationStrategy;
import jmb.jcortex.strategies.optimizationstrategies.OptimizationStrategy;

public class GradientDescentTrainerBuilder {

    private BatchingStrategy batchingStrategy;
    private OptimizationStrategy optimizationStrategy;
    private HaltingStrategy haltingStrategy;

    public static GradientDescentTrainerBuilder createTrainer() {
        return new GradientDescentTrainerBuilder();
    }

    public GradientDescentTrainerBuilder() {
        // Set default values
        batchingStrategy = new FullTrainingSetBatchingStrategy();
        optimizationStrategy = new MomentumOptimizationStrategy(0.1, 0.1);
    }

    public GradientDescentTrainerBuilder withBatchingStrategy(BatchingStrategy batchingStrategy) {
        this.batchingStrategy = batchingStrategy;
        return this;
    }

    public GradientDescentTrainerBuilder withOptimizationStrategy(OptimizationStrategy optimizationStrategy) {
        this.optimizationStrategy = optimizationStrategy;
        return this;
    }

    public GradientDescentTrainerBuilder withHaltingStrategy(HaltingStrategy haltingStrategy) {
        this.haltingStrategy = haltingStrategy;
        return this;
    }

    public GradientDescentTrainer build() {
        return new GradientDescentTrainer(batchingStrategy, optimizationStrategy, haltingStrategy);
    }

}
