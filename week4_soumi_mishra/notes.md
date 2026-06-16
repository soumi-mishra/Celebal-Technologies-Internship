---
## 📚 Theory Notes — Deep Learning Concepts

> These notes explain each concept used in this project. Written while building the CIFAR-10 classifier.

---

### 🔵 1. Perceptron

A **perceptron** is the simplest unit of a neural network — basically one artificial neuron.

It takes multiple inputs, multiplies each by a weight, adds a bias, and passes the result through an activation function to give an output.

**Formula:**
```
output = activation( w1*x1 + w2*x2 + ... + wn*xn + b )
```

- `x` = inputs (pixel values in our case)
- `w` = weights (learned during training)
- `b` = bias (a constant that shifts the output)
- `activation` = some function like step, sigmoid, or ReLU

A single perceptron can only separate data that is **linearly separable** — meaning it can draw one straight line to split two classes.  
For CIFAR-10 with 10 classes, one perceptron is not enough at all — we need many of them stacked together (MLP).

**Example — single perceptron logic:**
```python
import numpy as np

def perceptron(x, w, b):
    z = np.dot(w, x) + b        # weighted sum
    return 1 if z >= 0 else 0   # step activation

# pretend x = [pixel1, pixel2], w = [0.5, -0.3], b = 0.1
x = np.array([0.8, 0.4])
w = np.array([0.5, -0.3])
b = 0.1
print("perceptron output:", perceptron(x, w, b))   # 1 or 0
```
---
### 🔵 2. MLP — Multi-Layer Perceptron

An **MLP** is a perceptron stacked in layers — that's why it's also called a **fully connected network** or **dense network**.

Structure:
```
Input Layer → Hidden Layer(s) → Output Layer
```

In our ANN model:
- **Input**: 3072 neurons (32×32×3 flattened)
- **Hidden**: 512 → 256 → 128 neurons with ReLU
- **Output**: 10 neurons with Softmax

Each neuron in one layer connects to **every** neuron in the next layer — that's why it's called "Dense" in Keras.

Why MLP fails on images:
- Flattening destroys spatial structure (a pixel's neighbors matter — MLP ignores that)
- Too many parameters for image input (3072 → 512 = ~1.5 million weights in just the first layer)
- That's why CNN is better — it uses spatial structure instead of destroying it

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# this is an MLP — all Dense layers
mlp = Sequential([
    Flatten(input_shape=(32, 32, 3)),   # 3072 inputs
    Dense(512, activation='relu'),       # hidden layer 1
    Dense(256, activation='relu'),       # hidden layer 2
    Dense(10,  activation='softmax')     # output layer
])
mlp.summary()
```
---
### 🔵 3. Forward Pass

The **forward pass** is when an input travels through the network from left to right (input → output) to make a prediction.

**For each layer:**
```
Z = W · X + b      # linear step (dot product of weights and input + bias)
A = activation(Z)  # non-linear step (apply activation function)
```

This output `A` becomes the input `X` for the next layer.

**In CIFAR-10 example (ANN):**
1. Image (32×32×3) → flattened to 3072 values
2. Dense(512): Z = W·X + b, A = ReLU(Z)
3. Dense(256): same thing, smaller
4. Dense(10):  Z = W·X + b, A = Softmax(Z) → gives 10 probabilities

The final output is a vector like `[0.02, 0.01, 0.05, 0.60, 0.01, 0.08, 0.01, 0.01, 0.02, 0.19]`  
The class with the highest probability (index 3 = cat) is the model's prediction.

```python
import numpy as np

# manual forward pass for one layer
X = np.random.randn(3072)         # one flattened image
W = np.random.randn(512, 3072)    # weight matrix
b = np.zeros(512)                 # bias

Z = W @ X + b                     # linear step
A = np.maximum(0, Z)              # ReLU activation

print("Z shape:", Z.shape)        # (512,)
print("A shape:", A.shape)        # (512,) — input to next layer
```
---
### 🔵 4. Backpropagation

After the **forward pass** gives a prediction, we compare it to the true label using a **loss function**.  
**Backpropagation** is how the network learns — it figures out which weights caused the error and adjusts them.

Steps:
1. Compute **loss** (how wrong the prediction was)
2. Compute **gradient** of the loss with respect to each weight using the chain rule
3. Update weights using **gradient descent**: `W = W - lr * gradient`

The "back" in backpropagation means the gradient flows **backward** — from output layer → hidden layers → input layer.

**Chain Rule (the math behind it):**
```
∂L/∂W1 = ∂L/∂A3 × ∂A3/∂Z3 × ∂Z3/∂A2 × ∂A2/∂Z2 × ∂Z2/∂W1
```
Each `×` is multiplying derivatives layer by layer going backward.

**In Keras — this is all automatic:**
```python
model.compile(optimizer='adam', loss='categorical_crossentropy')
# when model.fit() runs, Keras:
# 1. does forward pass to get predictions
# 2. computes loss
# 3. runs backpropagation to get gradients
# 4. updates weights using Adam optimizer
```

We don't write backpropagation by hand — Keras and TensorFlow do it automatically via autograd.
---
### 🔵 5. Activation Functions — Sigmoid and Tanh

Activation functions add **non-linearity** to the network. Without them, stacking layers is pointless (many linear layers = one linear layer).

#### Sigmoid
```
σ(x) = 1 / (1 + e^(-x))
```
- Output range: **0 to 1**
- Used for **binary classification output** (one node, predict yes/no)
- Problem: **vanishing gradient** — for very large or small x, gradient ≈ 0, so learning stops

#### Tanh
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```
- Output range: **-1 to 1**
- Zero-centered (better than sigmoid for hidden layers)
- Still suffers from **vanishing gradient** for extreme values

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-6, 6, 200)

sigmoid = 1 / (1 + np.exp(-x))
tanh    = np.tanh(x)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x, sigmoid, color='steelblue', lw=2)
plt.axhline(0, color='k', lw=0.5); plt.axvline(0, color='k', lw=0.5)
plt.title('Sigmoid'); plt.xlabel('x'); plt.ylabel('output')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x, tanh, color='tomato', lw=2)
plt.axhline(0, color='k', lw=0.5); plt.axvline(0, color='k', lw=0.5)
plt.title('Tanh'); plt.xlabel('x'); plt.ylabel('output')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('activations_sigmoid_tanh.png', dpi=100, bbox_inches='tight')
plt.show()
print("Sigmoid at x=0:", round(sigmoid[100], 4))   # 0.5
print("Tanh at x=0:",    round(tanh[100], 4))       # 0.0
```

**Why we don't use sigmoid in hidden layers for our CIFAR model:**  
ReLU is faster and doesn't have the vanishing gradient problem.
---
### 🔵 6. ReLU Family

**ReLU = Rectified Linear Unit** — most popular activation for hidden layers in CNNs today.

#### ReLU
```
ReLU(x) = max(0, x)
```
- If x > 0 → output = x (pass through)
- If x ≤ 0 → output = 0 (kill)
- Output range: **0 to ∞**
- **Fast** to compute, **no vanishing gradient** for positive values
- Problem: **Dying ReLU** — neurons that get x < 0 always output 0 and stop learning

#### Leaky ReLU
```
LeakyReLU(x) = x if x > 0 else 0.01 * x
```
- Fixes dying ReLU — negative values still pass a small gradient through
- The `0.01` is a hyperparameter (called alpha)

#### ELU — Exponential Linear Unit
```
ELU(x) = x if x > 0 else α(e^x - 1)
```
- Smoother than Leaky ReLU for negative values
- Can produce negative outputs, making it zero-centered

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4, 4, 200)

relu       = np.maximum(0, x)
leaky_relu = np.where(x > 0, x, 0.01 * x)
elu        = np.where(x > 0, x, 1.0 * (np.exp(x) - 1))

plt.figure(figsize=(12, 4))
for i, (name, y, color) in enumerate(zip(
        ['ReLU', 'Leaky ReLU', 'ELU'],
        [relu, leaky_relu, elu],
        ['steelblue', 'tomato', 'mediumseagreen'])):
    plt.subplot(1, 3, i+1)
    plt.plot(x, y, color=color, lw=2)
    plt.axhline(0, color='k', lw=0.5); plt.axvline(0, color='k', lw=0.5)
    plt.title(name); plt.xlabel('x'); plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('activations_relu_family.png', dpi=100, bbox_inches='tight')
plt.show()
print("ReLU examples: ReLU(-2)=", max(0,-2), "  ReLU(3)=", max(0,3))
```

**In our CNN model:** We use `activation='relu'` in all Conv2D and Dense hidden layers.  
Final output layer uses `softmax` (not ReLU) because we need probabilities that sum to 1.
---
### 🔵 7. Loss Functions (Deep Learning)

A **loss function** measures how wrong the model's predictions are. The goal of training is to **minimize** the loss.

#### Categorical Cross-Entropy (used in our model)
Used for **multi-class classification** (like our 10-class CIFAR-10 problem).

```
Loss = -Σ y_true * log(y_pred)
```

- `y_true` = one-hot vector like [0,0,0,1,0,0,0,0,0,0] (cat)
- `y_pred` = softmax output like [0.01, 0.02, 0.03, 0.85, ...]
- Only the term where y_true=1 survives — so loss = -log(predicted probability for correct class)
- If model is very confident and correct → loss ≈ 0
- If model is confident but wrong → loss is very high (punished heavily)

#### Binary Cross-Entropy
Used for **binary classification** (dog vs cat, spam vs not-spam)
```
Loss = -(y * log(p) + (1-y) * log(1-p))
```

#### Mean Squared Error (MSE)
Used in **regression** — not for classification
```
Loss = (1/n) * Σ (y_true - y_pred)²
```

```python
import numpy as np

# manual categorical cross-entropy
y_true = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])   # cat (class 3)
y_pred = np.array([0.01, 0.01, 0.02, 0.85, 0.01, 0.03, 0.01, 0.01, 0.03, 0.02])

# only the correct class (index 3) contributes
loss = -np.sum(y_true * np.log(y_pred + 1e-8))
print(f"Cross-entropy loss: {loss:.4f}")   # small = model was right and confident

y_pred_wrong = np.array([0.01, 0.01, 0.02, 0.05, 0.01, 0.80, 0.01, 0.01, 0.05, 0.03])
loss_wrong = -np.sum(y_true * np.log(y_pred_wrong + 1e-8))
print(f"Cross-entropy loss (wrong prediction): {loss_wrong:.4f}")  # large = model was wrong
```
---
### 🔵 8. Convolution Layer

The **Conv2D** layer is the heart of a CNN. Instead of connecting every pixel to every neuron (like Dense), it slides a small **filter/kernel** over the image and detects local patterns.

**How it works:**
- A filter is a small matrix like 3×3 or 5×5
- It slides (convolves) across the image
- At each position: element-wise multiply filter with image patch, then sum
- This gives one number in the output (feature map)
- One filter detects **one type of pattern** (e.g., a horizontal edge)
- Using 32 filters → 32 feature maps → the model learns 32 different patterns

**Why Conv2D is better than Dense for images:**
- Dense: 32×32×3 → 512 = ~1.5 million weights in one layer
- Conv2D: one 3×3×3 filter = only 27 weights, shared across the whole image (called **weight sharing**)

```python
import numpy as np

# manual 2D convolution (simplified — no padding, 1 channel)
def conv2d_manual(image, kernel):
    ih, iw = image.shape
    kh, kw = kernel.shape
    out_h = ih - kh + 1
    out_w = iw - kw + 1
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = image[i:i+kh, j:j+kw]
            output[i, j] = np.sum(patch * kernel)
    return output

# a simple 4x4 image
image = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 8, 7, 6],
                  [5, 4, 3, 2]], dtype=float)

# horizontal edge detector kernel
kernel = np.array([[-1, -1, -1],
                   [ 0,  0,  0],
                   [ 1,  1,  1]], dtype=float)

feature_map = conv2d_manual(image, kernel)
print("Input image shape:", image.shape)
print("Kernel shape:", kernel.shape)
print("Feature map (output):\n", feature_map)
print("Feature map shape:", feature_map.shape)   # (2, 2) because no padding
```

**In our CNN model:**
- Block 1: `Conv2D(32, (3,3))` → 32 filters of size 3×3 → learns 32 low-level patterns (edges)
- Block 2: `Conv2D(64, (3,3))` → 64 filters → learns 64 mid-level patterns (shapes)
- Block 3: `Conv2D(128, (3,3))` → 128 filters → learns 128 high-level patterns (object parts)
---
### 🔵 9. Pooling and Stride

#### Pooling
**MaxPooling** reduces the spatial size of the feature map. It takes the **maximum value** from each region.

- `MaxPooling2D(2, 2)` → takes the max of every 2×2 block
- Feature map goes from 32×32 → 16×16 → reduces computation
- Keeps the strongest signal (most active feature) from each region
- Also makes the model slightly more **translation invariant** (if a feature moves slightly, it's still detected)

#### Stride
**Stride** controls how many pixels the filter moves each step.

- stride=1 → move 1 pixel at a time (default, output ≈ same size as input)
- stride=2 → move 2 pixels at a time (output = half the size)
- Using stride=2 in Conv2D can replace MaxPooling for downsampling

```python
import numpy as np

def maxpool2d(feature_map, pool_size=2):
    h, w = feature_map.shape
    out_h = h // pool_size
    out_w = w // pool_size
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            region = feature_map[i*pool_size:(i+1)*pool_size,
                                 j*pool_size:(j+1)*pool_size]
            output[i, j] = np.max(region)
    return output

# example 4x4 feature map
fm = np.array([[1, 3, 2, 4],
               [5, 6, 1, 2],
               [3, 2, 8, 1],
               [4, 5, 6, 7]], dtype=float)

pooled = maxpool2d(fm, pool_size=2)
print("Feature map (4x4):\n", fm)
print("\nAfter MaxPool 2x2 → shape becomes 2x2:\n", pooled)
# top-left 2x2 max = 6, top-right 2x2 max = 4, etc.
```

**In our CNN:**  
Each block ends with `MaxPooling2D(2, 2)`:  
- After Block 1: 32×32 → 16×16  
- After Block 2: 16×16 → 8×8  
- After Block 3: 8×8 → 4×4  
Then Flatten: 4×4×128 = 2048 values fed into Dense layers
---
### 🔵 10. Padding

When a 3×3 filter slides over a 32×32 image without padding, the output size becomes 30×30 — the edges are cut.  
**Padding** adds zeros around the image border so the output size stays the same.

#### Types:
- **`padding='valid'`** (default) — no padding, output is smaller than input
  - input 32×32, filter 3×3, stride 1 → output = 30×30
- **`padding='same'`** — pads with zeros to keep the same output size
  - input 32×32, filter 3×3, stride 1 → output = 32×32

**Output size formula (no padding):**
```
output_size = (input_size - kernel_size) / stride + 1
```

**Why we use `padding='same'` in our model:**
- Without it, the feature map shrinks with every conv layer
- With small inputs like 32×32, we can't afford to lose size too early
- We control downsampling only through MaxPooling (more intentional)

```python
import numpy as np

def output_size(input_size, kernel_size, stride=1, padding=0):
    return (input_size - kernel_size + 2*padding) // stride + 1

# valid padding (padding=0)
print("valid padding:", output_size(32, 3, stride=1, padding=0))   # 30

# same padding: padding = (kernel_size - 1) / 2
pad = (3 - 1) // 2
print("same padding:", output_size(32, 3, stride=1, padding=pad))  # 32

# with stride=2 (like downsampling)
print("stride=2, valid:", output_size(32, 3, stride=2, padding=0))   # 15
```

**In our CNN model:** all Conv2D layers use `padding='same'` to preserve spatial dimensions. Downsampling only happens through MaxPooling.
---
### 🔵 11. CNN Architectures

A CNN is typically made of repeated blocks of:  
**[Conv2D → BatchNorm → Conv2D → MaxPool → Dropout]**

Well-known architectures (all built on this idea):

| Architecture | Year | Key Idea | CIFAR-10 Accuracy |
|---|---|---|---|
| LeNet-5 | 1998 | first practical CNN | ~75% |
| AlexNet | 2012 | deep CNN + ReLU + Dropout | ~82% |
| VGG-16 | 2014 | very deep (16 layers), simple 3×3 filters | ~93% |
| ResNet | 2015 | skip connections to fix vanishing gradient | ~95% |
| DenseNet | 2016 | dense skip connections | ~96% |

**Our custom CNN (3 blocks) is closest to AlexNet/VGGNet in structure:**

```
Block 1: Conv2D(32) → BN → Conv2D(32) → MaxPool → Dropout
Block 2: Conv2D(64) → BN → Conv2D(64) → MaxPool → Dropout
Block 3: Conv2D(128) → BN → MaxPool → Dropout
Flatten → Dense(256) → Dropout → Dense(10, softmax)
```

**BatchNormalization** (BN):
- Normalizes the output of a layer during training
- Each mini-batch: subtract mean, divide by std
- Speeds up training, allows higher learning rate, acts as mild regularizer

```python
# ResNet's key idea — skip connection (residual connection)
# instead of learning F(x), learn F(x) + x (the residual)
# this lets gradients flow directly backward, solving vanishing gradient

# Keras example of a residual block (simplified):
from tensorflow.keras.layers import Add, Input, Conv2D, BatchNormalization, Activation
from tensorflow.keras.models import Model

inputs = Input(shape=(32, 32, 32))   # 32 feature maps
x = Conv2D(32, (3,3), padding='same')(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(32, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Add()([x, inputs])              # add input directly to output — skip connection!
x = Activation('relu')(x)
print("Residual block built successfully")
```
---
### 🔵 12. Transfer Learning

**Transfer Learning** = take a model already trained on a huge dataset, and fine-tune it on your smaller dataset.

Why it works:
- Big models (VGG, ResNet, EfficientNet) trained on ImageNet (1.2 million images, 1000 classes)
- They already know how to detect edges, textures, shapes, object parts
- We can reuse those weights instead of learning from scratch
- Even though CIFAR-10 images are only 32×32, transfer learning can still help

#### Two approaches:

**1. Feature extraction** — freeze all layers, only train the final classification head
- Fast, good when your dataset is small
- The pre-trained weights are not changed at all

**2. Fine-tuning** — unfreeze some top layers and retrain them too
- Slower, good when your dataset is large enough
- Adapts the model more to your specific data

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import Resizing

# load VGG16 pre-trained on ImageNet, without the top classification layer
base_model = VGG16(
    weights='imagenet',
    include_top=False,      # remove the final Dense layers
    input_shape=(96, 96, 3) # VGG needs at least 48x48, we upscale CIFAR-10
)

# STEP 1 — Feature extraction: freeze the base model
base_model.trainable = False

transfer_model = Sequential([
    Resizing(96, 96),            # upscale 32x32 → 96x96 for VGG
    base_model,                  # pre-trained feature extractor
    GlobalAveragePooling2D(),    # replaces Flatten — more efficient
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')  # our 10 CIFAR classes
])

transfer_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
transfer_model.summary()

# STEP 2 — Fine-tuning: unfreeze top layers of base model
# base_model.trainable = True
# for layer in base_model.layers[:-4]:   # freeze all but last 4 layers
#     layer.trainable = False
# then recompile with a much smaller learning rate (1e-5)
# and re-train

print("\nTrainable parameters:", sum(p.numpy().size for p in transfer_model.trainable_weights))
print("Non-trainable parameters:", sum(p.numpy().size for p in transfer_model.non_trainable_weights))
```

**Transfer learning usually gives 80–90%+ on CIFAR-10 vs our CNN's ~75–82%**  
It's the practical choice in real industry projects — very rarely train from scratch.
