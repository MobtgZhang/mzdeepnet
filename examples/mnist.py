import mzdeepnet as mz
import mzdeepnet.nn as nn
import mzdeepnet.optim as optim
import mzdeepnet.dataset as datasets
import numpy as np
def main():
    # Fetch MNIST data
    dataset = datasets.MNIST()
    x_train, y_train, x_test, y_test = dataset.arrays(flat=True, dp_dtypes=True)
    # Dataset Scaler
    scaler = mz.StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    # Prepare network seeds
    batch_size = 128
    train_feed = mz.SupervisedFeed(x_train,y_train,batch_size=batch_size)
    test_feed = mz.Feed(x_test)
    # Setup network
    weight_gain = 1.0
    weight_decay = 0.0005
    net = nn.Network(
        layers=[
            nn.Affine(
                n_out=1024,
                weights=mz.Parameter(mz.AutoFiller(weight_gain),
                                     weight_decay=weight_decay)
            ),
            nn.ReLU(),
            nn.Affine(
                n_out=1024,
                weights=mz.Parameter(mz.AutoFiller(weight_gain),
                                     weight_decay=weight_decay)
            ),
            nn.ReLU(),
            nn.Affine(
                n_out=dataset.n_classes,
                weights=mz.Parameter(mz.AutoFiller())
            )
        ],
        loss=mz.SoftmaxCrossEntropy(),
    )
    # Train network
    learn_rate = 0.05/batch_size
    optimizer = optim.Momentum(learn_rate)
    trainer = optim.GradientDescent(net, train_feed, optimizer)
    trainer.train_epochs(n_epochs=5)
    # Evaluate on test data
    error = np.mean(net.predict(test_feed) != y_test)
    print('Test error rate: %.4f' % error)
if __name__ == '__main__':
    main()


