# Federated-learning-of-different-types-of-malicious-clients-using-fashion-Mnist-datset
In code each client trains its local model with its own data and send modal parameters to the server. After server receives all data, it starts federated aggregation and update global model to new global model and distribute it to the clients. This is called a communication round and it continues until the round count.   

However, there are malicious clients that is used to interfere with the server aggregations. These malicious clients shuffle their label and updates their weights wrongly. There are two kinds of malicious clients.    

* Partially Malicious Client: it shuffles half of its labels. 15% of clients are partially malicious. It is coded by number 1.  
* Fully Malicious Client: it shuffles all of its labels. 5% of clients are Fully Malicious. It is coded by number 2.    
If the loss value of a client is between mean+standard_deviation and mean-standard_deviation [i.e., pink region] this client is considered as benign (coded by 0). (Case 1)  
If case 1 does not hold for the client, and loss value of the client is between mean+2*standard_deviation and mean-2*standard_deviation [i.e., purple region], this client is considered as partially malicious (coded by 1). (Case 2)  
  Otherwise, it is considered as a fully malicious client [i.e., green region] (coded by 2). (Case 3)  
    
    ![image](https://user-images.githubusercontent.com/48517382/233139930-8146392c-0f02-4ef6-a141-cd5f5f389157.png)
  
  
