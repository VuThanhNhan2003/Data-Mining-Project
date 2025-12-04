import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.Vote;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.RandomForest;
import weka.core.SelectedTag;

public class VoteEnsemblePipeline extends ModelFunctions {

    private Vote ensembleClassifier;

    public void buildEnsemble(String trainFile) throws Exception {
        // Load training data
        setTrainset(trainFile);
        trainset.setClassIndex(trainset.numAttributes() - 1);

        // Configure individual classifiers
        RandomForest rf = new RandomForest();
        rf.setOptions(weka.core.Utils.splitOptions(
                "-P 100 -I 200 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1"));

        NaiveBayes nb = new NaiveBayes();
        nb.setOptions(weka.core.Utils.splitOptions("-D"));

        // Create voting ensemble
        ensembleClassifier = new Vote();
        Classifier[] classifiers = {rf, nb};
        ensembleClassifier.setClassifiers(classifiers);
        ensembleClassifier.setCombinationRule(
                new SelectedTag(Vote.MAJORITY_VOTING_RULE, Vote.TAGS_RULES));

        // Train ensemble
        ensembleClassifier.buildClassifier(trainset);
    }

    public void evaluateEnsemble(String testFile) throws Exception {
        if (ensembleClassifier == null) {
            throw new Exception("Ensemble model chưa được build. Hãy gọi buildEnsemble() trước evaluateEnsemble().");
        }

        // Load test data
        setTestset(testFile);
        testset.setClassIndex(testset.numAttributes() - 1);

        // 10-fold cross-validation on training data
        Evaluation evaluator = new Evaluation(trainset);
        evaluator.crossValidateModel(ensembleClassifier, trainset, 10, new Random(1));

        // Print evaluation summary
        System.out.println(evaluator.toSummaryString("\nEvaluation\n-----------------\n", false));
        System.out.println(evaluator.toClassDetailsString("\nClass Details\n-----------------\n"));
    }
}
