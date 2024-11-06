import mxnet as mx
from mxnet.gluon import nn
from mxnet import gluon, autograd
from mxnet.gluon.data.vision import transforms
from sklearn.svm import OneClassSVM
from sklearn.metrics import pairwise_distances

# Define the DCNN architecture
class DCNN(nn.Block):
    def __init__(self, **kwargs):
        super(DCNN, self).__init__(**kwargs)
        self.net = nn.Sequential()
        
        # Block 1
        self.net.add(
            nn.Conv2D(32, kernel_size=3, strides=1, padding=1),  # Conv11
            nn.Conv2D(64, kernel_size=3, strides=1, padding=1),  # Conv12
            nn.Conv2D(64, kernel_size=3, strides=1, padding=1),  # Conv13
            nn.MaxPool2D(pool_size=2, strides=2)                  # Pool1
        )

        # Block 2
        self.net.add(
            nn.Conv2D(64, kernel_size=3, strides=1, padding=1),  # Conv21
            nn.Conv2D(128, kernel_size=3, strides=1, padding=1), # Conv22
            nn.Conv2D(128, kernel_size=3, strides=1, padding=1), # Conv23
            nn.MaxPool2D(pool_size=2, strides=2)                 # Pool2
        )

        # Block 3
        self.net.add(
            nn.Conv2D(96, kernel_size=3, strides=1, padding=1),  # Conv31
            nn.Conv2D(192, kernel_size=3, strides=1, padding=1), # Conv32
            nn.Conv2D(192, kernel_size=3, strides=1, padding=1), # Conv33
            nn.MaxPool2D(pool_size=2, strides=2)                 # Pool3
        )

        # Block 4
        self.net.add(
            nn.Conv2D(128, kernel_size=3, strides=1, padding=1), # Conv41
            nn.Conv2D(256, kernel_size=3, strides=1, padding=1), # Conv42
            nn.Conv2D(256, kernel_size=3, strides=1, padding=1), # Conv43
            nn.MaxPool2D(pool_size=2, strides=2)                 # Pool4
        )

        # Block 5
        self.net.add(
            nn.Conv2D(160, kernel_size=3, strides=1, padding=1), # Conv51
            nn.Conv2D(320, kernel_size=3, strides=1, padding=1), # Conv52
            nn.Conv2D(320, kernel_size=3, strides=1, padding=1), # Conv53
            nn.AvgPool2D(pool_size=6)                            # Pool5
        )

        # Fully connected layer and dropout
        self.net.add(
#             nn.GlobalAvgPool2D(),
            nn.Dropout(0.4),                 # Dropout 40%
            nn.Dense(10548),                 # FC6 (fully connected)
#             nn.Softmax()                     # Softmax layer for classification
        )

    def forward(self, x):
        return self.net(x)

# Initialize the network
net = DCNN()
net.initialize(mx.init.Xavier())

# Prepare the data iterator
data_iter = mx.image.ImageIter(
    batch_size=128,  
    data_shape=(3, 100, 100),  
    path_imgrec="ee798r/faces_webface_112x112/train.rec",
    path_imgidx="ee798r/faces_webface_112x112/train.idx",
    rand_crop=True,  
    rand_mirror=True,
)

trainer = gluon.Trainer(
    net.collect_params(),
    'sgd',
    {'learning_rate': 0.01, 'momentum': 0.9}
)

for name, param in net.collect_params().items():
    if 'dense' in name:  # Adjust this to match your final layer's name
        param.wd_mult = 5e-4
    else:
        param.wd_mult = 0  # No weight decay for convolutional layers

# Learning rate scheduler
lr_scheduler = mx.lr_scheduler.FactorScheduler(step=100000, factor=0.5)
# trainer.set_learning_rate(lr_scheduler)

# Training loop placeholder
def train(num_epochs):
    total_iterations = 0  # Counter for total iterations
    for epoch in range(num_epochs):
        data_iter.reset()
        for batch in data_iter:
            data = batch.data[0].as_in_context(mx.cpu())
            label = batch.label[0].as_in_context(mx.cpu())
            with autograd.record():
                output = net(data)
                loss = gluon.loss.SoftmaxCrossEntropyLoss()(output, label)
            loss.backward()
            trainer.step(batch.data[0].shape[0])  # This will also call the scheduler
            # Increment the total iteration count
            total_iterations += 1
            print(total_iterations)
            # Update learning rate based on the scheduler
            if total_iterations % 100000 == 0:
                new_lr = lr_scheduler.get_lr(total_iterations)
                trainer.set_learning_rate(new_lr)


        print(f'Epoch {epoch + 1} complete. Current Learning Rate: {trainer.learning_rate}')

# Example: Start training
train(num_epochs=1)
net.save_parameters("dcnn_trained.params")
