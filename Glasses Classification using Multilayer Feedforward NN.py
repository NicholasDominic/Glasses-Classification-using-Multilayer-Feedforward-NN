import csv, numpy as np, tensorflow as tf
from random import shuffle
from sklearn.decomposition import PCA

number_of_input = 5
number_of_output = 5
label_names = ['1', '2', '3', '5', '6', '7']
number_of_hidden = [4] # Non-Linearly Seperable (using Multi Layer Perceptron)

learning_rate = 1e-5
number_of_epoch = 5000
report_between = int(.02 * number_of_epoch) # every 100 epochs, print current error and epoch number

# Calculate the validation error by passing the validation dataset
# Save the model ONLY IF current validation error is lower than the previous validation error
validation_error = int(.1 * number_of_epoch)

x = tf.placeholder(tf.float32, [None, None, number_of_input])
target_output = tf.placeholder(tf.float32, [None, None, number_of_output])

def load_data(filename):
    raw_dataset = []
    with open(filename) as f:
        next(csv.reader(f))
        for row in csv.reader(f):       
            features = [f for f in row[:3]] + [row[4]] + [f for f in row[6:9]]
            label = row[9]
            raw_dataset.append((features, label))
    return raw_dataset

def preprocess(dataset):
    new_dataset = []
    for xFeatures, yLabel in dataset:
        features_new = [float(f) for f in xFeatures]
        label_new = [0] * len(label_names)
        label_new[label_names.index(yLabel)] = 1
        new_dataset.append((features_new, label_new))
    return new_dataset

def split_data(dataset):
    shuffle(dataset)

    # Training:Validation:Testing = 70:20:10
    split_data1 = int(.7 * len(dataset))
    split_data2 = int(.9 * len(dataset))

    training_data = dataset[:split_data1]
    validation_data = dataset[split_data1:split_data2]
    testing_data = dataset[split_data2:]

    return training_data, validation_data, testing_data

def normalize(dataset):
    features = np.array([data[0] for data in dataset])
    maxF = features.max(axis=0)
    minF = features.min(axis=0)

    new_dataset = []
    for features, label in dataset:
        for i in range(len(features)):
            features[i] = (features[i] - minF[i]) / (maxF[i] - minF[i]) * (1 - 0) + 0
        new_dataset.append((features, label)) 
    return new_dataset

def apply_pca(dataset):
    dataset_pca = []
    pca = PCA(n_components=5)
    feature = [data[0] for data in dataset]
    label = [data[1] for data in dataset]
    new_features = pca.fit_transform(feature)
    new_labels = pca.fit_transform(label)
    dataset_pca.append((new_features, new_labels))
    return dataset_pca

def forward_template(inp, inpCount, outCount):
    weight = tf.Variable(tf.random_normal([inpCount, outCount]))
    bias = tf.Variable(tf.random_normal([outCount]))
    return tf.nn.sigmoid(tf.matmul(inp, weight) + bias) # ActivationFunc(LinearCombination)

def forward_pass(inp):
    layer1 = forward_template(inp, number_of_input, number_of_hidden[0])
    layer2 = forward_template(layer1, number_of_hidden[0], number_of_output)
    return layer2

def optimize(train_data, validation_data, test_data, y):
    error = tf.reduce_mean(.5 * (target_output - y) ** 2)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #################### TRAINING & VALIDATION ####################
        for i in range(1, number_of_epoch+1):
            features = [data[0] for data in train_data]
            label = [data[1] for data in train_data]
            feed = {x: features, target_output: label}

            # Optimizer = backward pass (update weight and bias)
            _, err, acc = sess.run([optimizer, error, accuracy], feed)

            if i % report_between == 0:
                print("Epoch Number-{} | Current Error: {:.4f} %".format(i, err*100))

            if i % 500 == 0:
                val_features = np.array([data[0] for data in validation_data])
                val_label = np.array([data[1] for data in validation_data])
                vfeed_dic = {x: val_features, target_output: val_label}
                v_err, v_acc = sess.run([error, accuracy], vfeed_dic)
                print("VALIDATION - Epoch Number-{} | Current Error: {:.4f} %".format(i, v_err*100))
                tf.train.Saver().save(sess, "Classification_Model/glass.ckpt")
        
        #################### TESTING ####################
        tf.train.Saver().restore(sess, "Classification_Model/glass.ckpt")
        t_features = np.array([data[0] for data in test_data])
        t_label = np.array([data[1] for data in test_data])
        tfeed_dic = {x: t_features, target_output: t_label}
        t_acc = sess.run(accuracy, tfeed_dic)
        print("Accuracy: {:.2f}%".format(t_acc))

################## LOAD & PROCESS DATA ##################
raw_dataset = load_data("O202-COMP7117-KK02-00-classification.csv")
raw_dataset = preprocess(raw_dataset)
new_dataset = normalize(raw_dataset)

train_data, validation_data, test_data = split_data(new_dataset)
pca_applied_train_data = apply_pca(train_data)
pca_applied_validation_data = apply_pca(train_data)
pca_applied_test_data = apply_pca(train_data)

################### TRAINING & TESTING ###################
y = forward_pass(x)
correct_prediction = tf.equal(tf.argmax(target_output, 2), tf.argmax(y, 2))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100
optimize(pca_applied_train_data, pca_applied_validation_data, pca_applied_test_data, y)