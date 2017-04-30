/*
 * James Brundege
 * Date: 2017-04-09
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.datasource;


import jmb.jcortex.data.DataSet;
import jmb.jcortex.data.SynMatrix;
import org.junit.Test;

import java.util.List;
import java.util.stream.DoubleStream;

import static org.assertj.core.api.Assertions.assertThat;

public class MnistDigitsDataServiceTest {

    @Test
    public void loadDataFile() throws Exception {
        MnistDigitsDataService mnistDigitsDataService = new MnistDigitsDataService();
        DataSet dataSet = mnistDigitsDataService.loadDataFile();

        SynMatrix features = dataSet.getFeatures();
        assertThat(features.numCols()).isEqualTo(784);
        assertThat(features.numRows()).isEqualTo(60000);
        // All features are pixels normalized to 0-1 range
        DoubleStream.of(features.getAll())
                .forEach(value -> {
                    assertThat(value).isGreaterThanOrEqualTo(0.0);
                    assertThat(value).isLessThanOrEqualTo(1.0);
                });

        SynMatrix labels = dataSet.getLabels();
        assertThat(labels.numCols()).isEqualTo(10);
        assertThat(labels.numRows()).isEqualTo(60000);
        List<double[]> rows = labels.getRows();
        rows.forEach(row -> {
            assertThat(DoubleStream.of(row).sum()).isEqualTo(1.0);
            assertThat(DoubleStream.of(row).filter(value -> value == 0.0).count()).isEqualTo(9);
            assertThat(DoubleStream.of(row).filter(value -> value == 1.0).count()).isEqualTo(1);
        });
    }

}