

/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */

import java.nio.file.Path;
import java.nio.file.Paths;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

public class ModelFunctions {
    DataSource source;
    Instances dataset;
    String [] model_options;
    String [] data_options;
    Instances trainset;
    Instances testset;
    Instances label;
    Instances unlabel; 
    
    public ModelFunctions (){
        
    }
    public ModelFunctions (String filename, String m_options, String d_options) throws Exception{ 
        if (!filename.isEmpty()){
            this.source = new DataSource(filename);
            this.dataset = source.getDataSet();
        }
        if (m_options != null){
            this.model_options = weka.core.Utils.splitOptions(m_options);
                    }
        if (d_options != null){
            this.data_options = weka.core.Utils.splitOptions(d_options);
                    }
    }
    
    
    public Instances dividedatatraditional (Instances originalSet, double percent, boolean isTest ) throws Exception {
        RemovePercentage rp = new RemovePercentage();
        rp.setPercentage(percent);
        rp.setInvertSelection(isTest);
        rp.setInputFormat(originalSet);
        return Filter.useFilter(originalSet, rp);
        
    }

    public void setTrainset (String filename){
//        DataSource trainSource = new DataSource(filename);
//        this.trainset = trainSource.getDataSet();
//        
        try {
            // Create a Path object from the given filename
            Path path = Paths.get(filename);

            // If the path is not absolute, resolve it relative to the current working directory
            if (!path.isAbsolute()) {
                String currentDir = System.getProperty("user.dir");
                path = Paths.get(currentDir, filename).normalize();
            }

            // Convert the Path to a String
            String absolutePath = path.toAbsolutePath().toString();

            // Use the DataSource with the resolved path
            DataSource trainSource = new DataSource(absolutePath);
            this.trainset = trainSource.getDataSet();

            System.out.println("Dataset loaded successfully from: " + absolutePath);
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("Failed to load dataset.");
        }
    }
    
    public void setTestset (String filename) {
           try {
            // Create a Path object from the given filename
            Path path = Paths.get(filename);

            // If the path is not absolute, resolve it relative to the current working directory
            if (!path.isAbsolute()) {
                String currentDir = System.getProperty("user.dir");
                path = Paths.get(currentDir, filename).normalize();
            }

            // Convert the Path to a String
            String absolutePath = path.toAbsolutePath().toString();

            // Use the DataSource with the resolved path
            DataSource testSource = new DataSource(absolutePath);
            this.testset = testSource.getDataSet();

            System.out.println("Dataset loaded successfully from: " + absolutePath);
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("Failed to load dataset.");
        }
    }
    
    @Override
    public String toString() {
        return dataset.toSummaryString();
    }
            
}
