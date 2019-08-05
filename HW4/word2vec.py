import tensorflow as tf
import numpy as np

WINDOW_SIZE = 5
batch_size = 256
negative_samples = 20
EMBEDDING_DIM = 25
epoch = 3

shahname = open('ferdosi.txt', 'r', encoding='utf-8').read()
shahname = shahname.replace('\n', ' , ')
shahname = shahname.lower()

all_tokens = set()
for token in shahname.split():
    if token == ',':
        continue
    all_tokens.add(token)
all_tokens = list(all_tokens)

tokens_idx = {}
num_tokens = len(all_tokens)

for i in range(len(all_tokens)):
    tokens_idx[all_tokens[i]] = i

# make an array of list of tokens. list of all mesras
all_mesras = shahname.split(',')
mesras = []
for sentence in all_mesras:
    mesras.append(sentence.split())

print(len(mesras))
counter = 0
for s in mesras:
    counter += len(s)
print(counter)
print(mesras[100])
print(len(all_tokens))

W = tf.Variable(tf.random_normal([num_tokens, EMBEDDING_DIM], stddev=0.00001))
W_prim = tf.Variable(tf.random_normal([num_tokens, EMBEDDING_DIM], stddev=0.00001))

target = tf.placeholder(tf.int32, shape=(None, 1))
context = tf.placeholder(tf.int32, shape=(None, 1))
negative = tf.placeholder(tf.int32, shape=(None, negative_samples, 1))

vi = tf.reshape(tf.gather_nd(W, indices=target), shape=[batch_size, EMBEDDING_DIM])
uo = tf.reshape(tf.gather_nd(W_prim, indices=context), shape=[batch_size, EMBEDDING_DIM])
neg_uo = tf.reshape(tf.gather_nd(W_prim, indices=negative), shape=[batch_size, negative_samples, EMBEDDING_DIM])

score = tf.reduce_sum(tf.multiply(vi, uo), axis=1)
neg_score = tf.reduce_sum(tf.multiply(tf.reshape(vi, shape=(batch_size, 1, EMBEDDING_DIM)), neg_uo), axis=2)

loss = -tf.reduce_sum(tf.log(tf.sigmoid(score)) + tf.reduce_sum(tf.log(tf.sigmoid(-neg_score)), axis=1), axis=0)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.0012
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           400, 0.99, staircase=True)

train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

p = [0 for _ in range(num_tokens)]
all_words = []
for mesra in mesras:
    all_words += mesra
    for word in mesra:
        p[tokens_idx[word]] += 1

for i in range(len(p)):
    p[i] /= len(all_words)

OM = np.zeros((num_tokens, num_tokens))

for mesra in mesras:
    for index, word in enumerate(mesra):
        for i in range(max(0, index - WINDOW_SIZE), min(len(mesra), index + WINDOW_SIZE + 1)):
            if i == index:
                continue
            OM[tokens_idx[mesra[index]], tokens_idx[mesra[i]]] += 1

OMP = np.zeros_like(OM)

for i in range(num_tokens):
    for j in range(num_tokens):
        if OM[i, j] == 0:
            OMP[i, j] = p[j] ** 0.7

OMP = OMP / np.sum(OMP, axis=1).reshape(-1, 1)

for _ in range(epoch):
    x = []
    y = []
    n = []
    counter = 0
    for mesra in mesras:
        for index, word in enumerate(mesra):
            for i in range(max(0, index - WINDOW_SIZE), min(len(mesra), index + WINDOW_SIZE + 1)):
                if i == index:
                    continue

                x.append(tokens_idx[mesra[index]])
                y.append(tokens_idx[mesra[i]])

                ttt = tokens_idx[mesra[index]]
                negatives = np.random.choice(np.arange(num_tokens), negative_samples, p=OMP[ttt])
                n.append(negatives.tolist())

                if len(x) >= batch_size:

                    x_train = np.asarray(x).reshape(batch_size, 1)
                    y_train = np.asarray(y).reshape(batch_size, 1)
                    n_train = np.asarray(n).reshape(batch_size, negative_samples, 1)

                    sess.run(train_step, feed_dict={target: x_train, context: y_train, negative: n_train})

                    if counter % 100 == 0:
                        print("*" * 10, " epoch ", epoch + 1, " batch ", counter, " ", "*" * 10)
                        print('loss = ',
                              sess.run(loss, feed_dict={target: x_train, context: y_train, negative: n_train}))
                    x = x[batch_size:]
                    y = y[batch_size:]
                    n = n[batch_size:]
                    counter += 1

vectors = sess.run(W)
from sklearn.metrics.pairwise import cosine_similarity


def nearest_vecs(term, n=10):
    index = tokens_idx[term]
    candidates = []
    for i, vector in enumerate(vectors):
        if i == index:
            continue

        score = cosine_similarity([vectors[index]], [vector])[0, 0]
        candidates.append([all_tokens[i], score])
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:n]
    candidates = [i[0] for i in candidates]
    return candidates


print(nearest_vecs('رستم'))
# outputs ['خسرو', 'زمان', 'پیران', 'بیژن', 'لشکر', 'سوار', 'باز', 'پهلوان', 'ایران', 'پیش']
print(nearest_vecs('سیستان'))
# outputs ['براشوبد', 'گشتاسب', 'گیابر', 'سبکتر', 'برتابی', 'بگشایم', 'نژادکیان', 'جداست', 'برداشتن', 'گرداورد']
print(nearest_vecs('خردمند'))
# outputs ['دانا', 'جهاندار', 'خوب', 'هوش', 'شاد', 'انک', 'کان', 'پیر', 'پرخرد', 'نیک']
print(nearest_vecs('ایران'))
# outputs ['سپاه', 'لشکر', 'پیش', 'باز', 'زمان', 'شهر', 'رستم', 'سوار', 'نزد', 'انجمن']
print(nearest_vecs('گلاب'))
# outputs ['کافور', 'زعفران', 'سرخ', 'نگونسار', 'عنبر', 'عود', 'رخان', 'خشک', 'ارغوان', 'زنگ']
