
# **Call Reason Prediction Model Repository**

## **Overview**
This repository contains two Jupyter Notebook files designed for developing and utilizing a call reason prediction model based on customer call transcripts. The model is built using **BERT (Bidirectional Encoder Representations from Transformers)**, which is a deep learning model pre-trained on vast amounts of text data, such as books and Wikipedia, to understand language in context.

The goal of this model is to classify the reason behind customer calls by analyzing the call transcripts. This helps in automating customer support processes, improving service efficiency, and understanding customer needs based on conversation context.

## **Model Description**
The **BERT model** used in this project is a state-of-the-art transformer-based architecture developed by Google. BERT’s key advantage is that it considers the entire context of a word in a sentence, meaning it looks at words both before and after the target word. This makes it particularly well-suited for tasks involving text classification and understanding, like predicting the reason behind customer calls based on their transcripts.

- **BERT (Bidirectional Encoder Representations from Transformers)**:
  - It is a **pre-trained model** trained on large corpora and then fine-tuned on specific datasets, such as call transcripts in this case.
  - The **BERT-base-uncased** model used here is one of the standard BERT versions with 12 transformer layers and 110 million parameters.
  - We fine-tuned this pre-trained BERT model on our call transcript data to predict various call reasons, such as "Voluntary Cancel," "Booking," "IRROPS," and many more.

## **Overview**
This repository contains two Jupyter Notebook files designed for developing and utilizing a call reason prediction model based on customer call transcripts. The notebooks are:

1. **Model Training Notebook.ipynb**: This notebook encompasses the entire process for training the model and making predictions.
2. **Test Engine.ipynb**: This notebook provides the code for utilizing the pre-trained model.

### **Colab Notebook Links**
- [Model Training Notebook.ipynb](https://colab.research.google.com/drive/1Irht1A6ySmkXTxkgku2C5CAyQYg5bp-i?usp=sharing)
- [Test Engine.ipynb](https://colab.research.google.com/drive/1XmKhRoIws5vpNEGYg75qLERFA5W22d-z?usp=sharing)

## **Getting Started**

## **Data Folder Link**
-[Data Folder of all the preprocessed CSV & Escel Files](https://drive.google.com/drive/folders/1ntbzwZeMSJz0YjVzvSXBa8GTYXRoeaiG?usp=sharing)

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

## **Contributing**
If you wish to contribute to this project, please feel free to open an issue or submit a pull request.

## **Contact**
For any inquiries or feedback, please contact:
- Anil Kamat: [anilkamat.du.or.25@gmail.com]
- Deepak : [Deepak.du.or.25@gmail.com]
