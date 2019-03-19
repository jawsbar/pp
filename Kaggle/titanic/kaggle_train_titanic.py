import pandas as pd
import tensorflow as tf
import numpy as np


def MinMaxScaler(data):
    numrator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numrator / (denominator + 1e-7)

#지금 load file로 뽑아온 데이터들을 모두 숫자로 바꿔주는 작업
def load_file(is_test):
    if is_test:
        data_df = pd.read_csv("test.csv")
    else:
        data_df = pd.read_csv("train.csv")

    cols = ["Pclass", "Sex","Age", "Fare", "Embarked_0", "Embarked_1", "Embarked_2" ]


    data_df['Sex'] = data_df['Sex'].map({'female':0, 'male':1}).astype(int) #astype의 경우 ()안의 값으로 형변환
    #fillna는 nan 값을 ()안의 값으로 채운다.
    data_df["Age"] = data_df["Age"].fillna(data_df["Age"].mean())
    data_df["Fare"] = data_df["Fare"].fillna(data_df["Fare"].mean())

    data_df["Embarked"] = data_df["Embarked"].fillna('S')
    data_df["Embarked"] = data_df["Embarked"].map({'S':0, 'C':1, 'Q':2}).astype(int)
    # pd.concat의 경우 매트릭스를 합친다.
    # get_dummies는 더미 매트릭스를 만드는데 ex) a = ['a', 'b', 'c'] 일경우 3x3 매트릭스를 만들고
    # 1 과 0으로 표현한다. prefix의 경우 ex)embarked_1 embarked_2 와 같이 표현한다.
    data_df = pd.concat([data_df, pd.get_dummies(data_df['Embarked'], prefix='Embarked')], axis=1)

    data = data_df[cols].values

    if is_test:
        sing_col = data_df["PassengerId"].values
    else:
        sing_col = data_df["Survived"].values

    return sing_col, data

y_train, x_train = load_file(0)
#말그대로 expand한다. dimension을 확대시킨다.
y_train = np.expand_dims(y_train, 1)
train_len = len(x_train)

passId, x_test = load_file(1)
print(x_train.shape, x_test.shape)

x_all = np.vstack((x_train, x_test))
print(x_all.shape)
#minmaxscaler를 통해 0~1까지의 숫자로 바꿔주는 작업을 한다.
x_min_max_all = MinMaxScaler(x_all)
x_train = x_min_max_all[:train_len]
x_test = x_min_max_all[train_len:]
print(x_train.shape, x_test.shape)

learning_rate = 0.01

n_input = 7
n_hidden_1 = 32
n_hidden_2 = 64

X = tf.placeholder(tf.float32, shape=[None, n_input])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.Variable(tf.truncated_normal([n_input, 20], stddev=0.1), name='weight1')
b1 = tf.Variable(tf.truncated_normal([20], stddev=0.1), name='bias1')
L1 = tf.nn.relu(tf.matmul(X, W1)+b1)

W2 = tf.Variable(tf.truncated_normal([20, 1], stddev=0.1), name='weight2')
b2 = tf.Variable(tf.truncated_normal([1], stddev=0.1), name='bias2')

hypothesis = tf.sigmoid(tf.matmul(L1, W2)+b2)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

training_epoch = 15
batch_size = 32
display_step = 1
step_size = 1000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epoch):
        avg_cost = 0.
        avg_accuracy = 0.
        for step in range(step_size):
            offset = (step*batch_size) % (y_train.shape[0] - batch_size)

            batch_data = x_train[offset:(offset + batch_size), :]
            batch_labels = y_train[offset:(offset + batch_size), :]

            _, c, a = sess.run([optimizer, cost, accuracy], feed_dict={X:batch_data, Y:batch_labels})

            avg_cost += c / step_size
            avg_accuracy += a / step_size

        if epoch % display_step ==0:
            print("Epoch : ", '%02d' % (epoch+1), "cost={:.4f}".format(avg_cost), "train accuracy={:.4f}".format(avg_accuracy))

    outputs = sess.run(predicted, feed_dict={X:x_test})
    submission = ['PassengerId,Survived']

    for id, prediction in zip(passId,outputs):
        submission.append('{0},{1}'.format(id, int(prediction)))

    submission = '\n'.join(submission)

    with open('submission.csv', 'w') as outfile:
        outfile.write(submission)

