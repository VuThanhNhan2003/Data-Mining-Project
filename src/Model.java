

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Random;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;

public class Model extends ModelFunctions {

  AbstractClassifier model;

  private Bagging modelG;

  public Model(
    AbstractClassifier model,
    String filename,
    String m_options,
    String d_options
  ) throws Exception {
    super(filename, m_options, d_options);
    this.model = model;
  }

  public void Evaluate(String filename) throws Exception {
    // Input test set
    setTestset(filename);
    this.testset.setClassIndex(this.testset.numAttributes() - 1);
    //Evaluate model by 10-fold cross-validation
    Random rnd = new Random(1);
    int folds = 10;
    Evaluation eval = new Evaluation(this.trainset);
    eval.crossValidateModel(this.model, this.testset, folds, rnd);

    System.out.println("Model used: " + model.getClass().getSimpleName());
    // Print basic evaluation metrics
    System.out.println(
      eval.toSummaryString("\nEvaluation\n-----------------\n", false));
    // Print precision, recall, F1-score, and other class details
    System.out.println(
      eval.toClassDetailsString("\nClass Details\n-----------------\n")
    );
  }

  public void Build(String filename) throws Exception {
    // Input train set
    setTrainset(filename);
    this.trainset.setClassIndex(this.trainset.numAttributes() - 1);
    // Train model
    model.setOptions(this.model_options);
    model.buildClassifier(this.trainset);
  }

  public void predictClassLabel(String fileIn, String fileOut)
    throws Exception {
    // Read the unlabel file
    DataSource ds = new DataSource(fileIn);
    Instances unlabel = ds.getDataSet();
    unlabel.setClassIndex(unlabel.numAttributes() - 1);

    //Predict label
    for (int i = 0; i < unlabel.numInstances(); i++) {
      double predict = this.model.classifyInstance(unlabel.instance(i));
      unlabel.instance(i).setClassValue(predict);
    }

    //Export into fileOut file
    BufferedWriter outWriter = new BufferedWriter(new FileWriter(fileOut));
    outWriter.write(unlabel.toString());
    outWriter.newLine();
    outWriter.flush();
    outWriter.close();
  }

}
