# Email Spam Detection Using BlazingText on AWS

This project involves building, training, and deploying a machine learning model to detect email spam using the BlazingText algorithm on AWS SageMaker. The model is capable of classifying email messages as either spam or ham (not spam) based on their content.

## Table of Contents

- [Introduction](#introduction)
- [BlazingText Algorithm](#blazingtext-algorithm)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Data Preparation](#data-preparation)
- [Training the Model](#training-the-model)
- [Deploying the Model](#deploying-the-model)
- [Testing the Model](#testing-the-model)
- [Cleaning Up Resources](#cleaning-up-resources)
- [Conclusion](#conclusion)

## Introduction

Email spam is an ongoing problem, and this project demonstrates how to detect spam emails using machine learning. The project leverages AWS SageMaker and the BlazingText algorithm, which is optimized for text classification tasks. The model is trained on a dataset of emails and deployed as an endpoint to perform real-time predictions.

## BlazingText Algorithm

BlazingText is a highly optimized and scalable implementation of the Word2Vec and FastText algorithms, designed by AWS for text classification and word embedding. It supports both supervised and unsupervised training modes. The algorithm can train on large-scale datasets quickly by leveraging multi-core CPUs and GPUs, making it ideal for natural language processing (NLP) tasks such as sentiment analysis, spam detection, and more.

### Key Features of BlazingText:
- **Word2Vec and FastText Implementation:** Supports efficient word embeddings and text classification.
- **Supervised and Unsupervised Modes:** Can be used for both supervised learning (text classification) and unsupervised learning (word embeddings).
- **High Performance:** Optimized for distributed training, enabling faster processing of large datasets.
- **Ease of Use:** Easily integrates with AWS SageMaker for scalable training and deployment.

## Project Structure

```
├── data/                          # Directory to store data files
├── train.csv                      # Processed training dataset
├── validation.csv                 # Processed validation dataset
├── spam_detection.ipynb           # Jupyter Notebook for the project
└── README.md                      # Project README file
```

## Setup and Installation

1. **AWS Account:** Ensure you have an AWS account set up.
2. **AWS SageMaker Notebook Instance:** Create a SageMaker notebook instance and open the `spam_detection.ipynb` Jupyter notebook.
3. **Dependencies:** Install necessary Python libraries in your notebook environment:

    ```python
    %pip install nltk boto3 sagemaker pandas matplotlib
    ```

## Data Preparation

1. **Dataset:** Upload your email dataset (`email_dataset.csv`) to an S3 bucket.
2. **Data Processing:** The dataset is processed to label emails as `1` for spam and `0` for ham. The emails are tokenized and preprocessed using NLTK.
3. **S3 Upload:** The processed data is split into training and validation datasets and uploaded to S3 for model training.

## Training the Model

1. **Set Up SageMaker Estimator:** Define the BlazingText estimator in SageMaker with hyperparameters such as epochs, learning rate, and word n-grams.
2. **Training:** Train the model using the training dataset stored in S3. Monitor the training progress and evaluate the model's accuracy.

## Deploying the Model

1. **Model Deployment:** Deploy the trained BlazingText model as an endpoint using SageMaker.
2. **Endpoint Testing:** Test the endpoint by sending sample email messages and receiving predictions on whether they are spam or ham.

## Testing the Model

- **Sample Messages:** Use the deployed model to classify new email messages. The notebook provides examples of how to send input to the endpoint and interpret the results.

## Cleaning Up Resources

- **Delete Endpoint:** After testing, ensure you delete the SageMaker endpoint to avoid unnecessary charges.

    ```python
    text_classifier.delete_endpoint()
    ```

## Conclusion

The model provides a scalable and efficient solution for real-time spam classification, making it a valuable tool for handling unwanted emails.

---
