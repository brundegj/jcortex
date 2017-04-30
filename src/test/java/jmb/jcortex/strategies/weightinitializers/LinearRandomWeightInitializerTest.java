/*
 * James Brundege
 * Date: 2017-04-16
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.weightinitializers;

import jmb.jcortex.data.SynMatrix;
import org.junit.Test;

import java.util.Arrays;

import static org.assertj.core.api.Assertions.assertThat;

public class LinearRandomWeightInitializerTest {

    @Test
    public void initialize_RandomizesWeightsAroundZero() {
        LinearRandomWeightInitializer weightInitializer = new LinearRandomWeightInitializer(-0.1, 0.1);
        SynMatrix matrix = new SynMatrix(1000, 1000);
        SynMatrix initialized = weightInitializer.initialize(matrix);
        Arrays.stream(initialized.getAll()).forEach(value -> {
            assertThat(value).isGreaterThan(-0.1);
            assertThat(value).isLessThan(0.1);
        });
    }

}