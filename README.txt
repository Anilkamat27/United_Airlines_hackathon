
# **Call Reason Prediction Model Repository**

## **Overview**
This repository contains two Jupyter Notebook files designed for developing and utilizing a call reason prediction model based on customer call transcripts. The notebooks are:

1. **Model Training Notebook.ipynb**: This notebook encompasses the entire process for training the model and making predictions.
2. **Test Engine.ipynb**: This notebook provides the code for utilizing the pre-trained model.

### **Colab Notebook Links**
- [Model Training Notebook.ipynb](link-to-your-training-notebook)
- [Test Engine.ipynb](link-to-your-test-engine-notebook)

## **Getting Started**

### **Requirements**
The notebooks are intended to run in the Google Colab environment. No local Python installation or package management is necessary. 

### **Setup Instructions**
To use the pre-trained model, follow these steps:

1. **Upload Required Files to Google Drive**:
   - Upload the following files to your Google Drive:
     - `vocab.txt`
     - `training_args.bin`
     - `tokenizer_config.json`
     - `special_tokens`
     - `model.safetensors`
     - `config.json`
   - Ensure these files are stored in a folder named **`model`**.

2. **Connect Your Google Drive to Colab**:
   - In both notebooks, you will need to mount your Google Drive. Use the following code snippet to do this:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Update the Model Path**:
   - In the **Test Engine.ipynb** notebook, locate the section of the code where the model path is defined. Change the path to point to the directory where you saved the model files in your Google Drive. For example:

   ```python
   model_save_path = '/content/drive/MyDrive/model/'  # Adjust the path according to your folder structure
   ```

4. **Run the Notebooks**:
   - Start with the **Model Training Notebook.ipynb** to train the model (if not already trained) and save the model files.
   - After that, utilize the **Test Engine.ipynb** to load the pre-trained model and make predictions.

### **Usage**
- Once you run the **Test Engine.ipynb** notebook, you can input your call transcripts to predict the call reasons based on the trained model.

## **License**
This project is licensed under the MIT License.

## **Contributing**
If you wish to contribute to this project, please feel free to open an issue or submit a pull request.

## **Contact**
For any inquiries or feedback, please contact:
- Your Name: [your.email@example.com]
