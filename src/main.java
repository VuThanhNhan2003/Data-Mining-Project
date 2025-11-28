import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.io.BufferedWriter;
import java.io.FileWriter;

public class main {
  public static void main(String[] args) throws Exception {
    String filename = "Processing/dataset/";
    String trainLocation = filename + "train_data.arff";
    String trainLocationPost = filename + "train_data_post.arff";
    String testLocation = filename + "test_data.arff";
    String testLocationPost = filename + "test_data_post.arff";
    String testLocationLabel = filename + "test_data_label.arff";
    String testLocationLabelPost = filename + "test_data_label_post.arff";
    String predictTestLocation = filename + "test_data_predict.arff";
    String predictTestLocationPost = filename + "test_data_predict_post.arff";
    
    // data clean 
    String evaluationData = "heart_disease_clean.arff";
    
    ModelConfig config = null;
    
    java.util.Scanner scanner = new java.util.Scanner(System.in);
    System.out.println("Choose an option:");
    System.out.println("1. NaiveBayes");
    System.out.println("2. DecisionTree");
    System.out.println("3. RandomForest");
    System.out.println("4. Voting Ensemble");
    System.out.println("5. Model Evaluation (10-Fold CV)");
    System.out.print("Your choice: ");
    
    int choice = scanner.nextInt();
    scanner.close();
    
    switch (choice) {
      case 1:
        config = new ModelConfig(new NaiveBayes(), "-D");
        break;
      case 2:
        config = new ModelConfig(new J48(), "-C 0.25 -M 2");
        break;
      case 3:
        config = new ModelConfig(new RandomForest(), "-P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1");
        break;
      case 4:
        config = null;
        break;
      case 5:
        runEvalution(evaluationData);
        return;
      default:
        throw new IllegalArgumentException("Invalid choice: " + choice);
    }
    
    if (config == null) {
      VoteModel voting = new VoteModel();
      voting.Build(trainLocation);
      System.out.println("Performance of voting");    
    }
  }

  private static void runEvalution(String dataPath) throws Exception {
    System.out.println("\n==========MODEL EVALUATION ==========\n");
    
    // Load data 
    DataSource source = new DataSource(dataPath);
    Instances data = source.getDataSet();
    data.setClassIndex(data.numAttributes() - 1);
    System.out.println("Dataset: " + data.numInstances() + " instances, " + data.numAttributes() + " attributes");
    
    // run models
    NaiveBayes nb = new NaiveBayes();
    nb.setUseSupervisedDiscretization(true);
    
    RandomForest rf = new RandomForest();
    rf.setNumIterations(100);
    
    // evaluate 
    System.out.println("\n- Evaluate Naive Bayes ");
    double[] nbResults = evaluateModel(nb, data, 5, 10);
    
    System.out.println("\n- Evaluating Random Forest");
    double[] rfResults = evaluateModel(rf, data, 5, 10);
    
    // Compare and print results
    printComparison(nbResults, rfResults);
    
  }
  
  private static double[] evaluateModel(AbstractClassifier model, Instances data, int iterations, int folds) throws Exception {
    List<Double> accuracies = new ArrayList<>();
    List<Double> precisions = new ArrayList<>();
    List<Double> recalls = new ArrayList<>();
    List<Double> f1Scores = new ArrayList<>();
    List<Double> trainTimes = new ArrayList<>();
    List<Double> predTimes = new ArrayList<>();
    List<Double> maeList = new ArrayList<>();
    List<Double> rmseList = new ArrayList<>();
    List<Double> mseList = new ArrayList<>();
    
    for (int iter = 0; iter < iterations; iter++) {
      Random rnd = new Random(iter * 42 + 1);
      Instances shuffled = new Instances(data);
      shuffled.randomize(rnd);
      
      // Training Time
      long t1 = System.nanoTime();
      model.buildClassifier(shuffled); 
      double trainTime = (System.nanoTime() - t1) / 1e9; //nano to sec
      trainTimes.add(trainTime);
      
      // Prediction Time
      long t2 = System.nanoTime();
      for (int i = 0; i < shuffled.numInstances(); i++) {
        model.classifyInstance(shuffled.instance(i));
      }
      double predTime = (System.nanoTime() - t2) / 1e9 / shuffled.numInstances();
      predTimes.add(predTime);
      
      // 10-Fold 
      Evaluation eval = new Evaluation(shuffled);
      eval.crossValidateModel(model, shuffled, folds, new Random(iter + 100));
      
      double mae = eval.meanAbsoluteError();
      double rmse = eval.rootMeanSquaredError();
      double mse = rmse * rmse;

      maeList.add(mae);
      rmseList.add(rmse);
      mseList.add(mse);



      accuracies.add(eval.pctCorrect() / 100.0);
      
      // cal weight metrics 
      double[][] cm = eval.confusionMatrix();
      int numClasses = cm.length;
      double total = 0;
      double wPrec = 0, wRec = 0, wF1 = 0;
      
      for (int c = 0; c < numClasses; c++) {
        double classN = 0;
        for (int j = 0; j < numClasses; j++) classN += cm[c][j];
        total += classN;
        double tp = cm[c][c];
        double fp = 0;
        for (int i = 0; i < numClasses; i++) fp += cm[i][c];
        fp -= tp;
        double fn = classN - tp;
        double p = (tp + fp) > 0 ? tp / (tp + fp) : 0;
        double r = (tp + fn) > 0 ? tp / (tp + fn) : 0;
        double f = (p + r) > 0 ? 2 * p * r / (p + r) : 0;
        wPrec += p * classN;
        wRec += r * classN;
        wF1 += f * classN;
      }
      
      precisions.add(total > 0 ? wPrec / total : 0);
      recalls.add(total > 0 ? wRec / total : 0);
      f1Scores.add(total > 0 ? wF1 / total : 0);
    }
    return new double[] {
      mean(accuracies), std(accuracies),      
      mean(precisions), std(precisions),    
      mean(recalls), std(recalls),       
      mean(f1Scores), std(f1Scores),           
      mean(maeList), std(maeList),         
      mean(rmseList), std(rmseList),         
      mean(mseList), std(mseList),            
      mean(trainTimes), std(trainTimes),    
      mean(predTimes), std(predTimes)         
    };
  }
  

  private static void printComparison(double[] nb, double[] rf) {
    System.out.println("\n- MODEL COMPARISON \n ");
    System.out.printf("%-15s %-20s %-20s\n", "Metric", "Naive Bayes", "Random Forest");
      
    System.out.println("Classification:");
    System.out.printf("  %-13s %.4f +/- %.4f     %.4f +/- %.4f\n", "Accuracy", nb[0], nb[1], rf[0], rf[1]);
    System.out.printf("  %-13s %.4f +/- %.4f     %.4f +/- %.4f\n", "Precision", nb[2], nb[3], rf[2], rf[3]);
    System.out.printf("  %-13s %.4f +/- %.4f     %.4f +/- %.4f\n", "Recall", nb[4], nb[5], rf[4], rf[5]);
    System.out.printf("  %-13s %.4f +/- %.4f     %.4f +/- %.4f\n", "F1-Score", nb[6], nb[7], rf[6], rf[7]);
    
    // Regre
    System.out.println("Regression:");
    System.out.printf("  %-13s %.4f +/- %.4f     %.4f +/- %.4f\n", "MAE", nb[8], nb[9], rf[8], rf[9]);
    System.out.printf("  %-13s %.4f +/- %.4f     %.4f +/- %.4f\n", "RMSE", nb[10], nb[11], rf[10], rf[11]);
    System.out.printf("  %-13s %.4f +/- %.4f     %.4f +/- %.4f\n", "MSE", nb[12], nb[13], rf[12], rf[13]);
    
    // Speed
    System.out.println("Speed:");
    System.out.printf("  %-13s %.4f +/- %.4f s   %.4f +/- %.4f s\n", "Train Time", nb[14], nb[15], rf[14], rf[15]);
    System.out.printf("  %-13s %.6f +/- %.6f   %.6f +/- %.6f\n", "Pred Time", nb[16], nb[17], rf[16], rf[17]);
    
    // Gener
    System.out.println("\n- GENERE \n");
    System.out.println("Naive Bayes:   " + (nb[1] < 0.05 ? "Good" : "Poor") + " (Std: " + String.format("%.4f", nb[1]) + ")");
    System.out.println("Random Forest: " + (rf[1] < 0.05 ? "Good" : "Poor") + " (Std: " + String.format("%.4f", rf[1]) + ")");
    
    // Trade-offs
    System.out.println("\n- TRADE-OFFS \n");
    // double speedRatio = rf[10] / nb[10];
    String winner = nb[0] > rf[0] ? "Naive Bayes" : (rf[0] > nb[0] ? "Random Forest" : "Tie");
    System.out.println("Winner (Accuracy): " + winner);
  }
  
  private static double mean(List<Double> vals) {
    return vals.stream().mapToDouble(d -> d).average().orElse(0);
  }
  
  private static double std(List<Double> vals) {
    double m = mean(vals);
    return Math.sqrt(vals.stream().mapToDouble(v -> Math.pow(v - m, 2)).average().orElse(0));
  }
}