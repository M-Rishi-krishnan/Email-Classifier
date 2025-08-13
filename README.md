Email Spam Classifier using BERT and PyTorch
This project contains a Jupyter Notebook that walks through the process of building a spam classification system by fine-tuning a pre-trained BERT model. The model is trained on the Kaggle "Email Spam Classification Dataset" to distinguish between spam and non-spam (ham) emails. The entire pipeline is built using PyTorch and the HuggingFace Transformers library.

Project Overview
The core of this project is to take a labeled dataset of emails and fine-tune a powerful language model, bert-base-uncased, to understand the nuances of spam versus legitimate emails. The final model can classify new, unseen emails with high confidence, achieving over 98% accuracy on the validation set.

Key Features
Processes the Kaggle email dataset.

Fine-tunes a bert-base-uncased model for binary text classification.

Saves the trained model and tokenizer for future use.

Dataset
This model is trained on the Email Spam Classification Dataset CSV from Kaggle. You must download the emails.csv file from the following link:

Dataset Link: https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv

How to Run the Notebook
To set up and run this project locally, please follow these steps.

1. Clone the Repository
bash
git clone https://github.com/M-Rishi-krishnan/Email-Classifier.git
cd Email-Classifier

2. Install Dependencies
You will need Python and several packages to run the notebook. It is highly recommended to use a virtual environment.

First, install the required packages:
pip install torch pandas numpy scikit-learn transformers[torch] accelerate jupyter

3. Download the Dataset
Download the emails.csv file from the Kaggle link provided above. Place the emails.csv file in the same directory as the Jupyter Notebook.

4. Launch Jupyter and Run the Code
From your terminal, launch Jupyter Notebook.

The notebook will:

Load and preprocess the data from emails.csv.

Fine-tune the BERT model, showing progress and validation accuracy.

Save the final model to a new directory named spam_classifier_model/.

Demonstrate how to classify new emails using the trained model.


Validation Accuracy: ~99%

This demonstrates the model's strong ability to generalize and accurately classify emails it has not seen during training.
