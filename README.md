##### This project focuses on document classification, for which it uses 20-ng-news-corpus dataset available by scikit-learn package. 
##### First part of project (10_source_code/naive_bayes_v1.ipynb) uses Naive Bayes Multinomial classifier to build synthetic dataset. Another part of the project (10_source_code/lstm_v1.ipynb, 10_source_code/lstm_v2.ipynb)) uses Bi-directional LSTM model to fit both real and synthetic dataset. The idea is to explore different impact on accuracy and loss of machine learning models based on the type of datasets we use. For instance, we see a lot of fluctuation in accuracy and loss when the data used to train the model is randomly generated (synthetic data).

<center><head><h4> Real Dataset </h4></head>
[![lstm-accuracy-real-data.png](https://i.postimg.cc/KvC526fL/lstm-accuracy-real-data.png)](https://postimg.cc/CzH8C6WL)
</center>

<center><head><h4> Synthetic Dataset </h4></head>
![Alt text](assets/lstm_accuracy_synthetic_data.png?raw=true "Accuracy on Synthetic Data")
![Alt text](assets/lstm_loss_synthetic_data.png?raw=true "Loss on Synthetic Data")
