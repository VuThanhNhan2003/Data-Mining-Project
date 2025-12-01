import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.Vote;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomForest;
import weka.core.SelectedTag;

public class VoteModel extends ModelFunctions {

    private Vote voting;  

    public void Build(String filename) throws Exception {
        setTrainset(filename);
        this.trainset.setClassIndex(this.trainset.numAttributes() - 1);

        RandomForest modelR = new RandomForest();
        String[] optionA = weka.core.Utils.splitOptions(
                "-P 100 -I 200 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1");
        modelR.setOptions(optionA);

        NaiveBayes modelN = new NaiveBayes();
        modelN.setOptions(weka.core.Utils.splitOptions("-D"));

        this.voting = new Vote();
        Classifier[] classifier = {modelR, modelN};

        this.voting.setClassifiers(classifier);
        this.voting.setCombinationRule(
                new SelectedTag(Vote.MAJORITY_VOTING_RULE, Vote.TAGS_RULES));
        this.voting.buildClassifier(trainset);
    }

    public void Evaluate(String filename) throws Exception {
        if (this.voting == null) {
        throw new Exception("Model chưa được build. Hãy gọi Build() trước Evaluate().");
        }
        setTestset(filename);
        this.testset.setClassIndex(this.testset.numAttributes() - 1);

        Random rand = new Random(1);
        int folds = 10;

        Evaluation eval = new Evaluation(this.trainset);
        eval.crossValidateModel(this.voting, this.trainset, folds, rand);

        // Summary
        System.out.println(eval.toSummaryString("\nEvaluation\n-----------------\n", false));

        // Precision, recall, f1...
        System.out.println(eval.toClassDetailsString("\nClass Details\n-----------------\n"));
    }
}
