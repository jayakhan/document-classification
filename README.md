##### This project focuses on document classification, for which it uses 20-ng-news-corpus dataset available by scikit-learn package. 
##### First part of project (10_source_code/naive_bayes_v1.ipynb) uses Naive Bayes Multinomial classifier to build synthetic dataset. Another part of the project (10_source_code/lstm_v1.ipynb, 10_source_code/lstm_v2.ipynb)) uses Bi-directional LSTM model to fit both real and synthetic dataset. Later, we explore scenarios in which one of these models would perform better than the other.

#### Real Dataset
![Alt text](assets/accuracy_real_data.png?raw=true "Accuracy on Real Data")
![Alt text](assets/loss_real_data.png?raw=true "Loss on Real Data")

#### Synthetic Dataset
![Alt text](assets/accuracy_sd.png?raw=true "Accuracy on Synthetic Data")
![Alt text](assets/loss_sd.png?raw=true "Loss on Synthetic Data")
