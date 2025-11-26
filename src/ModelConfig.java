import weka.classifiers.AbstractClassifier;

public class ModelConfig {
    AbstractClassifier model;
    String options;
    public ModelConfig(AbstractClassifier model, String options) {
        this.model = model;
        this.options = options;
    }
    public AbstractClassifier getModel() {
        return model;
    }
    public void setModel(AbstractClassifier model) {
        this.model = model;
    }
    public String getOptions() {
        return options;
    }
    public void setOptions(String options) {
        this.options = options;
    }
}
