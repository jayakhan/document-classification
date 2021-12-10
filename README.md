##### This project focuses on document classification, for which it uses 20-ng-news-corpus dataset available by scikit-learn package. 
##### First part of project (10_source_code/naive_bayes_v1.ipynb) uses Naive Bayes Multinomial classifier to build synthetic dataset. Another part of the project (10_source_code/lstm_v1.ipynb, 10_source_code/lstm_v2.ipynb)) uses Bi-directional LSTM model to fit both real and synthetic dataset. The idea is to explore different impact on accuracy and loss of machine learning models based on the type of datasets we use. For instance, we see a lot of fluctuation in accuracy and loss when the data used to train the model is randomly generated (synthetic data).

#### Real Dataset
![ScreenShot](https://github.com/jayakhan/document-classification/blob/main/assets/lstm_accuracy_real_data.png)
![ScreenShot](https://github.com/jayakhan/document-classification/blob/main/assets/lstm_loss_real_data.png)

#### Synthetic Dataset
![ScreenShot](https://github.com/jayakhan/document-classification/blob/main/assets/lstm_accuracy_synthetic_data.png)
![ScreenShot](https://github.com/jayakhan/document-classification/blob/main/assets/lstm_loss_synthetic_data.png)
