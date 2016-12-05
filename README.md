# Ethz-Bootstrapped-Captioning
(Under construction) Code for "bootstrap, review, decode: using out-of-domain textual data to improve image captioning"

# MS-COCO Test Server

| Model | LLH | CS-1 | CS-2 | CS-3 | CS-4 | CS-5 |
|----|----|----|----|----|----|----|
ATT-LSTM-EXT (Ours) | **73.4** | **56.3** | **42.3** | **31.7** | **96.4** | **25.4** |
ATT~\cite{you2016image} | 73.1 | 56.5 | 42.4| 31.6 | 94.3 | 25.0 |
Google~\cite{vinyals2015show} | 71.3 | 54.2 | 40.7 | 30.9 | 94.3 | 25.4 |
kimiyoung~\cite{yang2016encode} | 72.0 | 55.0 | 41.4 | 31.3 | 96.5 | 25.6 |

# Overview
Our model is mainly based on "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention", "Review Networks for Caption Generation", "Image Captioning with Semantic Attention" and "From Captions to Visual Concepts and Back", our implementation is mainly based on "arctic-caption"[1] and "visual-concept"[2].

# Requirements
This code is written in python and caffe. To use it you will need:
* Python 2.7
* [NumPy](http://www.numpy.org/)
* [Theano](http://deeplearning.net/software/theano/)
* [Caffe](http://caffe.berkeleyvision.org/)
* [h5py](http://www.h5py.org/)

# Data Preparation
Our model can be run on MS-COCO[3] and Flickr30K[4] dataset

# Reference

If you use this code as part of any published research, please acknowledge the
following paper (it encourages researchers who publish their code!):

**"bootstrap, review, decode: using out-of-domain textual data to improve image captioning"**  
Chen, Wenhu and Lucchi, Aurelien and Hofmann, Thomas. *Submitted to CVPR (2017)*

    @article{chen2016bootstrap,
      title={Bootstrap, Review, Decode: Using Out-of-Domain Textual Data to Improve Image Captioning},
      author={Chen, Wenhu and Lucchi, Aurelien and Hofmann, Thomas},
      journal={arXiv preprint arXiv:1611.05321},
      year={2016}
    }

## License

The code is released under a [revised (3-clause) BSD License](http://directory.fsf.org/wiki/License:BSD_3Clause).

[1]: https://github.com/kelvinxu/arctic-captions
[2]: https://github.com/s-gupta/visual-concepts
[3]: http://mscoco.org/
[4]: http://shannon.cs.illinois.edu/DenotationGraph/
