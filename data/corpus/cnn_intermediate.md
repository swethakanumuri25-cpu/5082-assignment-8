# Convolutional Neural Networks

## Why CNNs are used for images
Convolutional Neural Networks are designed to work well with grid-like data such as images. Unlike a standard ANN that treats every input feature independently, a CNN preserves spatial structure. This makes it effective for recognizing patterns like edges, textures, and object parts. CNNs use local receptive fields and shared weights, which greatly reduce the number of parameters compared with fully connected layers. That makes them more efficient and better suited for image tasks.

## Convolution, Filters, and Feature Maps
A convolution layer applies a small learnable filter across the input image. As the filter slides over the input, it computes dot products that produce a feature map. Early filters often capture simple features such as edges or corners, while deeper layers learn more complex patterns. Because the same filter is reused across the whole image, CNNs exploit translation-related structure and need fewer parameters than dense architectures processing the same input.

## Pooling and Hierarchical Representation
Pooling layers reduce the spatial size of feature maps, which lowers computation and helps the network focus on the most important signals. Max pooling is common because it preserves the strongest activation in a region. CNNs create hierarchical representations: shallow layers detect basic visual features, middle layers combine them into motifs, and deeper layers represent higher-level concepts such as object parts. This layered feature extraction is a major reason CNNs perform well in computer vision.
