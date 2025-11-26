

import java.util.Random;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.Stacking;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.REPTree;
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
    
    Bagging modelG = new Bagging();
    String [] optionsC = weka.core.Utils.splitOptions("-P 100 -S 1 -num-slots 1 -I 10 -W weka.classifiers.trees.REPTree -- -M 2 -V 0.001 -N 3 -S 1 -L -1 -I 0.0" );
    modelG.setClassifier(new REPTree());
    modelG.setNumIterations(100);
    modelG.setOptions(optionsC);
    
    Vote voting = new Vote();
    Classifier [] classifer = {modelR,modelJ48,modelG};
    voting.setClassifiers(classifer);
    voting.setCombinationRule(new SelectedTag(Vote.MAJORITY_VOTING_RULE,Vote.TAGS_RULES));
    voting.buildClassifier(trainset);
}
  
  
}
