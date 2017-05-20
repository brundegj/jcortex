# jcortex
A neural network implementation in Java, built on the JBLAS linear algebra library.

This is a simple implementation of a multi-layered perceptron created for my own understanding.
It is a toy implementation for education. It uses a builder pattern to allow the creation and training
of neural nets with swappable optimizations and regularizations.

To run this code you need Java 8+ installed. It uses Maven as a build system. To build:
Install JDK 1.8+
Install Maven
<code>
	git clone https://github.com/brundegj/jcortex.git
	mvn clean test
</code>
This will run the unit tests and some integrations tests, including a demo.

To run the demo in isolation:
<code>
	mvn exec:java
</code>

This demo trains a digit recognizer on a very small subset of the MNSIT data set (1200 training samples).
It demonstrates overfitting (due to the low number of training samples, and mitigates this on a second pass
using multiple regularization techniques.

See https://github.com/brundegj/jcortex/blob/master/src/main/java/jmb/jcortex/demos/mnist/MnistOvertrainingDemo.java
for an example of how to setup and run a neural network.
