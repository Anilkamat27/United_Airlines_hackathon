
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
## **Why I Chose a BERT Model and Its Benefits**

I chose to use a BERT model to predict the primary reason for customer calls because it’s excellent at understanding language. BERT is unique because it reads sentences both forwards and backwards, allowing it to figure out the meaning of each word based on its surrounding context. This is crucial in customer call transcripts, where the same word can have different meanings depending on the conversation.

### Key Benefits:
1. **Pre-Trained on Extensive Data**: 
   - BERT has already been trained on a massive amount of text data, which saves time and effort. I don't have to start from scratch; I can fine-tune it using our specific dataset of 66,000 labeled call transcripts. This helps the model learn the unique patterns in customer conversations quickly.

2. **Handles Complex Language**: 
   - Thanks to its pre-training, BERT can manage informal or complicated language that customers might use during calls. It understands slang, incomplete sentences, and different tones, which are often found in real-world conversations.

3. **Manages Long Call Transcripts**: 
   - BERT is designed to handle large chunks of text (up to 512 tokens), which is ideal for processing long customer conversations. The model considers the entire transcript, ensuring it doesn’t miss key details when predicting the reason for the call.

These features make BERT an ideal choice for predicting call reasons, as it combines language understanding with the ability to handle real-world, complex, and lengthy data.

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

## **Challenges and Limitations in Model Training and Prediction**

During the training process of the BERT model for predicting call reasons, I encountered significant challenges related to computational resources, which impacted the overall efficiency and performance of the model:

### **Limited Training Epochs**:
- Due to restricted computing units on Google Colab, I could only afford to train the model for 1 epoch. This single epoch took approximately 3 hours to complete.
- With additional training epochs (e.g., 10 or 12 epochs), I believe the model would yield substantially better results and improve its predictive capabilities.

### **Extended Prediction Time**:
- Generating predictions for 5,000 transcripts was a time-consuming process, taking about 2 hours to complete. This delay was also attributed to the limited computational resources available during this phase.
- If I had access to unlimited computing units, I could have significantly accelerated this prediction process.

### **Potential for Improved Performance**:
- Given the constraints on training epochs and prediction speed, the model's current performance could be optimized further. With more computational resources, I could develop a state-of-the-art model that delivers higher accuracy and efficiency in predicting call reasons.

### **Conclusion**:
The experience underscored the importance of adequate computational resources in machine learning projects, especially when working with complex models like BERT. Increasing the training epochs and enhancing the prediction capabilities would allow for the development of a more robust and accurate predictive model.

## **Key Considerations for Future Improvements**

To enhance the model in future iterations, the following strategies should be considered:

1. **Increasing Training Epochs**: 
   - Training the model for more epochs will likely yield better predictive performance, allowing it to learn more from the data and improve its accuracy.

2. **Utilizing Advanced Hardware**: 
   - Accessing better hardware (such as TPUs or GPUs) can significantly reduce both training and prediction time, enabling the model to scale more effectively.

3. **Hyperparameter Tuning**: 
   - Experimenting with different hyperparameters, such as learning rate, batch size, and weight decay, can help optimize the model's performance and stability.

5. **Continuous Monitoring and Updating**: 
   - Regularly updating the model with new data and retraining it can help maintain accuracy as call reasons evolve over time, ensuring the model stays relevant and up-to-date.

By applying these improvements, the model will become more robust, scalable, and accurate, leading to better predictions for future tasks.

## **Contributing**
If you wish to contribute to this project, please feel free to open an issue or submit a pull request.

## **Contact**
For any inquiries or feedback, please contact:
- Anil Kamat: [anilkamat.du.or.25@gmail.com]
- Deepak : [Deepak.du.or.25@gmail.com]
