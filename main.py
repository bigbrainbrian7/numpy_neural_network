import numpy as np
import matplotlib.pyplot as plt

from layer import Linear, ReLU, BatchNorm
from model import Model
from loss import SoftmaxCrossEntropyLoss

model = Model(
    Linear(784, 256),
    BatchNorm(256),
    ReLU(),
    Linear(256, 128),
    BatchNorm(128),
    ReLU(),
    Linear(128, 10),
)

loss_fn = SoftmaxCrossEntropyLoss()

train_images = np.load("data/train_images.npy") / 255.0
train_labels = np.load("data/train_labels.npy")

test_images = np.load("data/test_images.npy") / 255.0
test_labels = np.load("data/test_labels.npy")

np.random.seed(0)

indices = np.random.permutation(len(train_images))
train_images = train_images[indices]
train_labels = train_labels[indices]

split = int(0.8 * len(train_images))

val_images  = train_images[split:]
val_labels  = train_labels[split:]
train_images = train_images[:split]
train_labels = train_labels[:split]

epochs = 10
batch_size = 32
learning_rate = 0.1

errors = []
val_errors_bruh = []

for epoch in range(epochs):
    learning_rate *= 0.9
    batch_indices = np.random.permutation(len(train_images))

    val_batch_indices = np.random.permutation(len(val_images))


    for batch_start in range(0, len(train_images), batch_size):
        model.zero_grad()
        
        batch_samples = batch_indices[batch_start:batch_start + batch_size]
        val_batch_samples = val_batch_indices[batch_start:batch_start + batch_size]


        X_batch = train_images[batch_samples]
        X_batch = X_batch.reshape(-1, 28*28)

        y_batch = train_labels[batch_samples]
        y_one_hot = np.zeros((y_batch.size, 10))
        y_one_hot[np.arange(y_batch.size), y_batch] = 1

        # print(f"{y_batch} -> {y_one_hot}")

        output = model.forward(X_batch)
        output: np.ndarray = loss_fn.forward(output, y_one_hot)

        errors.append(loss_fn.loss)

        # print(output.argmax(axis=1, keepdims=True).shape)
        # print(y_batch.reshape(-1, 1).shape)

        correct = (output.argmax(axis=1, keepdims=True) == y_batch.reshape(-1, 1))

        print(f"Epoch: {epoch}, Batch: {batch_start}, Accuracy: {correct.mean()}")

        upstream = loss_fn.backward(y_one_hot)
        model.backward(upstream)

        model.step(learning_rate)

        X_batch = val_images[val_batch_samples]
        X_batch = X_batch.reshape(-1, 28*28)

        y_batch = val_labels[val_batch_samples]
        y_one_hot = np.zeros((y_batch.size, 10))
        y_one_hot[np.arange(y_batch.size), y_batch] = 1

        # print(f"{y_batch} -> {y_one_hot}")

        output = model.forward(X_batch)
        output: np.ndarray = loss_fn.forward(output, y_one_hot)
        val_errors_bruh.append(loss_fn.loss)


#----Validation----

val_errors = []

batch_indices = np.random.permutation(len(val_images))

tot = 0

for batch_start in range(0, len(val_images), batch_size):
    batch_samples = batch_indices[batch_start:batch_start + batch_size]

    X_batch = val_images[batch_samples]
    X_batch = X_batch.reshape(-1, 28*28)

    y_batch = val_labels[batch_samples]
    y_one_hot = np.zeros((y_batch.size, 10))
    y_one_hot[np.arange(y_batch.size), y_batch] = 1

    # print(f"{y_batch} -> {y_one_hot}")

    output = model.forward(X_batch)
    output: np.ndarray = loss_fn.forward(output, y_one_hot)
    val_errors.append(loss_fn.loss)

    # print(output.argmax(axis=1, keepdims=True).shape)
    # print(y_batch.reshape(-1, 1).shape)

    tot += (output.argmax(axis=1, keepdims=True) == y_batch.reshape(-1, 1)).sum(axis=0)

print(f"Validation Accuracy: {tot / len(val_images)}")

print(model.layers[1].gamma)
print(model.layers[1].beta)


plt.plot(errors)
plt.plot(val_errors_bruh)
plt.xlabel("Batch #")
plt.ylabel("Loss")
plt.show()

def plot_first_layer_weights(model, n_cols=16):
    # first layer weights shape is (784, hidden_size) with your convention
    # each COLUMN is one neuron's weights — one per output node
    weights = model.layers[0].weights  # (784, hidden_size)
    
    n_neurons = weights.shape[1]       # number of output neurons
    n_rows = int(np.ceil(n_neurons / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows * 1.2))
    axes = axes.flat

    for i in range(n_neurons):
        ax = axes[i]
        img = weights[:, i].reshape(28, 28)  # slice column i, reshape to image

        # normalize each neuron independently so features are visible
        vmax = np.abs(img).max()
        ax.imshow(img, cmap='RdBu', vmin=-vmax, vmax=vmax)
        ax.axis('off')

    # hide any unused axes
    for i in range(n_neurons, len(axes)):
        axes[i].axis('off')

    plt.suptitle('First layer weights', y=1.01)
    plt.tight_layout()
    plt.show()

# plot_first_layer_weights(model)

