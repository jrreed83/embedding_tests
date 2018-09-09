import models.cbow as cbow 
import trainers.training as training
import data.dataset as d

import sklearn.manifold as manifold
import matplotlib.pyplot as  plt 
import numpy as np 

def main():
    # Grab the dataset
    print('Fetching the Dataset')
    pattern1 = 'a b a b a b a b a b a b a b a b a b a b a b a b a b a b a b a b'
    pattern2 = 's t s t s t s t s t s t s t s t s t s t s t s t s t s t s t s t'
    pattern3 = '0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1'

    dataset = d.TestDataSet(
        f'{pattern1} {pattern2} {pattern3}'
    )
    vocab_size = len(dataset.word2id)  

    # Build the model
    print('Initializing the model')
    model = cbow.CBOW(vocab_size = vocab_size, embedding_dim = 5)

    # Train the model
    print('Training')
    train = training.train
    losses = train(model=model, dataset=dataset, num_epochs=500, batch_size=1)

    plt.plot(losses)
    plt.show()

    X = model.get_embedding_weights()

    s = dataset.word2id['s']
    t = dataset.word2id['t']

    xs = X[s]
    xt = X[t]

    xs = xs / np.linalg.norm(xs)
    xt = xt / np.linalg.norm(xt)

    print(np.dot(xs, xt))

    Y = manifold.TSNE(n_components=2).fit_transform(X)
 
    a = [x for x, y in Y]
    b = [y for x, y in Y]


    fig, ax = plt.subplots()
    ax.scatter(a, b)
    for i in range(len(Y)):
        ax.annotate(dataset.id2word[i], (a[i], b[i]))
    plt.show()
if __name__ == '__main__':
    main()