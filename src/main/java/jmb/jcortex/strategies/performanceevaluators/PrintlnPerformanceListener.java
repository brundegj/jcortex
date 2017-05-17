/*
 * James Brundege
 * Date: 2017-04-30
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.performanceevaluators;

import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static java.lang.String.format;
import static java.lang.String.join;
import static jmb.jcortex.JCortexConstants.TEST_SET_PERCENT_WRONG;
import static jmb.jcortex.JCortexConstants.TRAINING_SET_PERCENT_WRONG;
import static jmb.jcortex.JCortexConstants.VALIDATION_SET_PERCENT_WRONG;

/**
 *
 */
public class PrintlnPerformanceListener implements PerformanceListener {
    public static final NumberFormat ONE_DECIMAL = NumberFormat.getNumberInstance();
    static {
        ONE_DECIMAL.setMinimumFractionDigits(1);
        ONE_DECIMAL.setMaximumFractionDigits(1);
    }

    @Override
    public void performanceEvent(Map<String, Double> data) {
        List<String> metrics = new ArrayList<>();

        Double trainingSetWrong = data.get(TRAINING_SET_PERCENT_WRONG);
        if (trainingSetWrong != null) {
            metrics.add(format("Training set %% wrong: %s", formatPercent(trainingSetWrong)));
        }

        Double validationSetWrong = data.get(VALIDATION_SET_PERCENT_WRONG);
        if (validationSetWrong != null) {
            metrics.add(format("Validation set %% wrong: %s", formatPercent(validationSetWrong)));
        }

        Double testSetWrong = data.get(TEST_SET_PERCENT_WRONG);
        if (testSetWrong != null) {
            metrics.add(format("Test set %% wrong: %s", formatPercent(testSetWrong)));
        }

        if (!metrics.isEmpty()) {
            System.out.println(join("\t", metrics));
        }
    }

    private String formatPercent(Double percent) {
        return ONE_DECIMAL.format(percent * 100);
    }
}
