/*
 * James Brundege
 * Date: 2017-04-30
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.performanceevaluators;

import org.knowm.xchart.XChartPanel;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.style.XYStyler;

import javax.swing.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.toList;
import static jmb.jcortex.JCortexConstants.TRAINING_SET_PERCENT_WRONG;
import static jmb.jcortex.JCortexConstants.VALIDATION_SET_PERCENT_WRONG;

/**
 *
 */
public class ChartingPerformanceListener implements PerformanceListener {

    private List<Integer> iterations = new ArrayList<>();
    private List<Double> trainingPercentError = new ArrayList<>();
    private List<Double> validationPercentError = new ArrayList<>();
    private XChartPanel<XYChart> chartPanel;
    private int numIterations;

    public ChartingPerformanceListener() {
        this.numIterations = 1;
    }

    public ChartingPerformanceListener(int numIterations) {
        this.numIterations = numIterations;
    }

    @Override
    public void performanceEvent(Map<String, Double> data) {
        iterations.add(iterations.size() + 1);
        Double trainingSetWrong = data.get(TRAINING_SET_PERCENT_WRONG);
        if (trainingSetWrong != null) {
            trainingPercentError.add(trainingSetWrong * 100);
        }

        Double validationSetWrong = data.get(VALIDATION_SET_PERCENT_WRONG);
        if (validationSetWrong != null) {
            validationPercentError.add(validationSetWrong * 100);
        }

        if (chartPanel == null) {
            initialize();
        }

        XYChart chart = chartPanel.getChart();
        // This allows the chart to compress if we need to accommadate more than numIterations
        if (iterations.size() > numIterations) {
            chart.getStyler().setXAxisMax((double)iterations.size());
        }
        chart.updateXYSeries("Training Set", iterations, trainingPercentError, null).setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line);
        chart.updateXYSeries("Validation Set", iterations, validationPercentError, null).setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line);
        chartPanel.revalidate();
        chartPanel.repaint();
    }

    private void initialize() {
        XYChart chart = new XYChartBuilder()
                .width(800)
                .height(600)
                .xAxisTitle("Iteration")
                .yAxisTitle("% Wrong")
                .build();

        XYStyler styler = chart.getStyler();
        styler.setXAxisMin(0.0);
        styler.setXAxisMax(100.0);
        styler.setYAxisMin(0.0);
        styler.setYAxisMax(100.0);
        chart.addSeries("Training Set", iterations, trainingPercentError);
        chart.addSeries("Validation Set", iterations, validationPercentError);

        chartPanel = new XChartPanel<>(chart);
        try {
            JFrame chartUI = new JFrame("Training Performance");
            SwingUtilities.invokeAndWait(() -> {
                chartUI.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
                chartUI.add(chartPanel);
                chartUI.pack();
                chartUI.setVisible(true);
            });
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

}
