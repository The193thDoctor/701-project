import matplotlib.pyplot as plt

# for encoding in ["rotation", "amplitude"]:
# with open (f"output/quantum_enc=rotation.js", "w") 


# with open (f"output/quantum_enc=amplitude.js", "w")

c_Train_acc = [0.5035, 0.512, 0.544, 0.5965, 0.6295, 0.6495, 0.655, 0.658, 0.665, 0.668, 0.6745, 0.6755, 0.6785, 0.684, 0.683]
c_Train_loss = [0.6993623445034027, 0.689793184041977, 0.6840084357261658, 0.6799471399784088, 0.6767184207439423, 0.6737015662193299, 0.6707626438140869, 0.6678385334014892, 0.6649009523391723, 0.6619741153717041, 0.6591267943382263, 0.6563150887489319, 0.6535792174339294, 0.6509296319484711, 0.6484073235988617]
c_Val_acc = [0.496, 0.514, 0.568, 0.608, 0.636, 0.65, 0.658, 0.658, 0.674, 0.668, 0.668, 0.674, 0.678, 0.68, 0.684]
c_Val_loss = [0.6949985907191322, 0.6877360864291115, 0.6831765960133265, 0.6793989416152711, 0.6762985218138922, 0.673816546561226, 0.6711787912580702, 0.6683773880913144, 0.6654191480742561, 0.6627688776879084, 0.6600669612960209, 0.6574859165009999, 0.6544767637101431, 0.652335438463423, 0.6500986369829329]
q_train_loss = [0.6708, 0.6342, 0.6200, 0.6129, 0.6078, 0.6042, 0.6018, 0.5993, 0.5978, 0.5966, 0.5954, 0.5946, 0.5943, 0.5937, 0.5935]
q_train_acc = [0.5845, 0.6580, 0.6700, 0.6670, 0.6725, 0.6745, 0.6770, 0.6790, 0.6795, 0.6765, 0.6770, 0.6770, 0.6830, 0.6800, 0.6765]
q_test_loss = [0.6555, 0.6375, 0.6307, 0.6269, 0.6260, 0.6217, 0.6197, 0.6186, 0.6167, 0.6175, 0.6201, 0.6167, 0.6182, 0.6151, 0.6134]
q_test_acc = [0.6760, 0.6620, 0.6660, 0.6680, 0.6680, 0.6740, 0.6700, 0.6660, 0.6720, 0.6720, 0.6760, 0.6800, 0.6760, 0.6800, 0.6800]

epochs = range(len(c_Train_acc))
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, c_Train_loss, label='Train_loss(Neural Network)', color = 'orange')
plt.plot(epochs, c_Val_loss, label='Test_loss(Neural Network)', color = 'orange', linestyle='--')
plt.plot(epochs, q_train_loss, label='Train_loss(Quantum Neural Network)', color = 'blue')
plt.plot(epochs, q_test_loss, label='Test_loss(Quantum Neural Network)',  color = 'blue', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, c_Train_acc, label='Train_acc(Neural Network)', color = 'orange')
plt.plot(epochs, c_Val_acc, label='Test_acc(Neural Network)', color = 'orange', linestyle='--')
plt.plot(epochs, q_train_acc, label='Train_acc(Quantum Neural Network)', color = 'blue')
plt.plot(epochs, q_test_acc, label='Test_acc(Quantum Neural Network)',  color = 'blue', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()