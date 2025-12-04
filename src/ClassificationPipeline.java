import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ClassificationPipeline extends ModelFunctions {

    private AbstractClassifier classifier;

    public ClassificationPipeline(AbstractClassifier clf,
                                  String dataPath,
                                  String clfOptions,
                                  String dataOptions) throws Exception {
        super(dataPath, clfOptions, dataOptions);
        this.classifier = clf;
    }

    public void train(String trainFile) throws Exception {
        // Load training data
        setTrainset(trainFile);
        this.trainset.setClassIndex(trainset.numAttributes() - 1);

        // Train classifier
        classifier.setOptions(this.model_options);
        classifier.buildClassifier(trainset);
    }

    public void predict(String inputUnlabeled, String outputLabeled) throws Exception {
        // Load unlabeled data
        DataSource reader = new DataSource(inputUnlabeled);
        Instances unlabeled = reader.getDataSet();
        unlabeled.setClassIndex(unlabeled.numAttributes() - 1);

        // Predict class for each instance
        for (int i = 0; i < unlabeled.numInstances(); i++) {
            double pred = classifier.classifyInstance(unlabeled.instance(i));
            unlabeled.instance(i).setClassValue(pred);
        }

        // Write result to file
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputLabeled))) {
            writer.write(unlabeled.toString());
            writer.newLine();
        }
    }

    public void evaluate(String testFile) throws Exception {
        // Load test data
        setTestset(testFile);
        this.testset.setClassIndex(testset.numAttributes() - 1);

        // Evaluation using 10-fold CV
        int folds = 10;
        Random randomSeed = new Random(1);
        Evaluation evaluator = new Evaluation(trainset);

        evaluator.crossValidateModel(classifier, testset, folds, randomSeed);

        System.out.println("Classifier: " + classifier.getClass().getSimpleName());
        System.out.println(evaluator.toSummaryString("\nEvaluation\n-----------------\n", false));
        System.out.println(evaluator.toClassDetailsString("\nClass Details\n-----------------\n"));
    }
}
