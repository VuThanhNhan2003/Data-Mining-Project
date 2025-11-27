

import java.util.Random;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Classifier;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.SelectedTag;

public class VoteModel extends ModelFunctions{
      
  public void Build(String filename) throws Exception {
    setTrainset(filename);
    this.trainset.setClassIndex(this.trainset.numAttributes() - 1);
    RandomForest modelR = new RandomForest();
    String [] optionA = weka.core.Utils.splitOptions("-P 100 -I 200 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1");
    modelR.setOptions(optionA);
    
    J48 modelJ48 = new J48();
    String [] option = weka.core.Utils.splitOptions("-C 0.25 -M 2");
    modelJ48.setOptions(option);
    
    // NaiveBayes
    NaiveBayes modelNB = new NaiveBayes();
    String[] optionNB = weka.core.Utils.splitOptions("-D");
    modelNB.setOptions(optionNB);
    
    Vote voting = new Vote();
    Classifier [] classifer = {modelR,modelJ48,modelNB};
    voting.setClassifiers(classifer);
    voting.setCombinationRule(new SelectedTag(Vote.MAJORITY_VOTING_RULE,Vote.TAGS_RULES));
    voting.buildClassifier(trainset);
}
}
