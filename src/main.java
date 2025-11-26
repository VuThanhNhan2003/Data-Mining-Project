import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;

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
    String validation = filename + "val_data.arff";
    String validationPost = filename + "val_data_post.arff";
    ModelConfig config = null;
    java.util.Scanner scanner = new java.util.Scanner(System.in);
    System.out.println("Choose a model to use: ");
    System.out.println("1. NaiveBayes");
    System.out.println("2. DecisionTree");
    System.out.println("3. RandomForest");
    int choice = scanner.nextInt();
    scanner.close();
    switch (choice) {
      case 1:
        /**
         * @NaiveBayes main  */
        config = new ModelConfig(new NaiveBayes(), "-D");
        break;
      /**
       * @DecisionTree main  */
      case 2:
        config = new ModelConfig(new J48(), "-C 0.25 -M 2");
        break;
      /**
       * @RandomForest main
       */
      case 3:
        config =
          new ModelConfig(
            new RandomForest(),
            "-P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1"
          );
        break;
      /**
       * @GradientBoosting
       */
      case 4:
        config= null;
        break;
      default:
        throw new IllegalArgumentException("Invalid choice: " + choice);
    }
    if (config == null) {

      VoteModel voting = new VoteModel();
      voting.Build(trainLocation);
      System.out.println("Performance of voting");      // 
    }
    else{
      try {
        Model model = new Model(config.getModel(), "", config.getOptions(), null);
        ///////////////////////////////////////
        /// Step02: Build model
        model.Build(trainLocation);
        model.predictClassLabel(testLocation, testLocationLabel);
      } catch (Exception e) {
        System.err.println("Error during model training: " + e.getMessage());
        e.printStackTrace();
        return;
      }
      //////////////////////////////////////////
      /// Step 03
      model.Build(trainLocationPost); 
      model.predictClassLabel(testLocationPost, predictTestLocationPost);

      
    }
  }
}
