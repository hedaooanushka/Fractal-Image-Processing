# Fractal Image Compression Using Discrete Cosine Transform with Domain Classification and Hexagonal Partitioning

## Inspiration
We aimed to design a fractal image compression method using hexagonal partitioning to reduce computational complexity through parallel computing. In this method a range block is compared with every other domain block. The domain block having the least RMS error is mapped to the range block and replaced with it storing the original contrast and brightness settings. In this manner, an encoding file is created which stores information of all the mappings, and then the original image is decoded iteratively until the output resembles the original image. It was a complicated and error-prone task, but I'm glad to mention that the experiments succeeded as we've achieved the desired compression rate.

## Publication
Springer, International Conference on Power Engineering and Intelligent Systems (PEIS) · Oct 29, 2024
https://link.springer.com/chapter/10.1007/978-981-97-6710-6_32

