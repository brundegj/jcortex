/*
 * James Brundege
 * Date: 2017-04-30
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.performanceevaluators;

import java.util.Map;

/**
 *
 */
public interface PerformanceListener {

    void performanceEvent(Map<String, Double> data);

}
