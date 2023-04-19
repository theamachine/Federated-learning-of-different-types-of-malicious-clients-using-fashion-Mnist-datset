# Federated-learning-of-different-types-of-malicious-clients-using-fashion-Mnist-datset
In code each client trains its local model with its own data and send modal parameters to the server. After server receives all data, it starts federated aggregation and update global model to new global model and distribute it to the clients. This is called a communication round and it continues until the round count.  
However, there are malicious clients that is used to interfere with the server aggregations. These malicious clients shuffle their label and updates their weights wrongly. There are two kinds of malicious clients.  
* Partially Malicious Client: it shuffles half of its labels. 15% of clients are partially malicious. It is coded by number 1.  
* Fully Malicious Client: it shuffles all of its labels. 5% of clients are Fully Malicious. It is coded by number 2.  
