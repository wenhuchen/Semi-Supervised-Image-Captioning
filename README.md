# ETHZ-Bootstrapped-Captioning
Code for "bootstrap, review, decode: using out-of-domain textual data to improve image captioning", implemented with Theano, Caffe.

## MS-COCO Test Server

| Model | B-1 | B-2 | B-3 | B-4 | CIDEr | METEOR |
|----|----|----|----|----|----|----|
ATT-LSTM-EXT (Ours) | **73.4** | **56.3** | **42.3** | **31.7** | **96.4** | **25.4** |
ATT | 73.1 | 56.5 | 42.4| 31.6 | 94.3 | 25.0 |
Google | 71.3 | 54.2 | 40.7 | 30.9 | 94.3 | 25.4 |
kimiyoung | 72.0 | 55.0 | 41.4 | 31.3 | 96.5 | 25.6 |

## Overview
Our model is mainly based on "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention", "Review Networks for Caption Generation", "Image Captioning with Semantic Attention" and "From Captions to Visual Concepts and Back", our implementation is mainly based on [arctic-caption](1) and [visual-concept](2).

## Requirements
This code is written in python and caffe. To use it you will need:

* Python 2.7
* [NumPy](http://www.numpy.org/)
* [Theano](http://deeplearning.net/software/theano/)
* [Caffe](http://caffe.berkeleyvision.org/)
* [h5py](http://www.h5py.org/)
* [opencv](http://opencv.org/)
* [scikit-image](http://scikit-image.org/)

## Data Preparation
Our model can be run on [MS-COCO](3) and [Flickr30K](4) dataset, the bootstrapped learning can be done on [NewsCrawl](5) dataset. The whole process takes three steps, the first step is to detect the salient visual concepts and their corresponding regional features. The second step is to pre-train the model on out-of-domain data and get a good starting points, the final step is to finetune the model on the indomain pairwise dataset.

### Config file
The config.py needs to be specified to run experiments. You need to edit the config by setting the coco/flickr image folder to your local location, also the Multi-Instance Caffe and Standard Caffe toolkit need to be specified. The standard caffe toolkit is used for layer feature extraction, while Multi-Instance Caffe used for visual concept detection

	coco_image_base=""
	flickr_image_base=""
	caffe_mil="/Users/wenhuchen/caffe-mil/python"
	caffe_standard="/Users/wenhuchen/caffe-standard/python"

### Visual concept Detection
This step is done via the visual concept detector, the details of implementation is in folder "visual-concepts".
	
	cd visual-concepts
	python eval.py --test_json ../Data/coco/cocotalk_val.json --dataset coco --split val --order 30
This step will generate concepts and regional features, which are neccessary for the next step.

### Bootstrapping
This is an optional step, if you bootstrap the model by pre-training, the model will gain around 1.0~2.0 BLEU-4 improvements. The first step is to "fake" visual concepts from the commoncrawl text files or "fake" regional features from the commoncrawl text files.
	
	cd attention_generator
	python commoncrawl --splits train val test --type commoncrawl --cutoff 15
The commonad will produce files in ../Data/{type} folder, with these generated files, we can start bootstrap pre-training
	
	python train.py --saveto commoncraw_pretrained --dataset commoncrawl --cutoff 15
Parameter list

	--saveto indicates the file name to store the experiment settings and snapshots, it's required in every experiment
	--dataset indicates which dataset to be used and validated on in the training process
	--cutoff indicates only the top n concpets from visual concept files are needed for the training. 
The training will be fired in this way, early stop criterion will smartly stop training when BLEU-4 score on validation set converges.

### In-Domain Finetunning
After the bootstrapping step, the in-domain training can be started. But before that, the semantic and visual representation data needs to be prepared via extract_features.py and extract_semantics.py

	python extract_semantics.py --dataset coco --split train val test
	python extract_features.py --dataset coco --concepts ../Data/coco/visual_concept_val.json --type resnet --split val --mapping ../Data/coco/id2path_mapping.pkl
All the generated results will be saved to caption_{split}.json and Feat\_{split}.h5. These two files are the main inputs to the training procedure.
We can simply start training our model by train.py based on the pre-trained weights from commoncraw_pretrained_bestll.npz.
	
	python train.py --saveto coco_resnet_cnninit_review_ext_standard --dataset coco --cutoff 30 --cnn_type resnet --pretrained commoncraw_pretrained_bestll.npz
Parameter list
	
	--saveto is the name of the experiment
	--dataset is the dataset for training
	--cutoff indicates the top n words selected for input
	--type indicates the cnn type for decoder initialization
	--pretrained starts the training from specified bootstrapped weights
	--region use regional features as input, if not specified, the model uses semantic features instead
The normal training takes around 24 hours to finish, the best model is saved under {saveto}_bestll.npz. Now we could use it to test model performance.

### Decoding
We implement both single model decoding and ensemble model decoding here, if you specify --pkl_name --model parameters more than one models, then ensemble decoding will be called automatically.
	
	python generate --pkl_name coco_resnet_cnninit_review_ext_standard.pkl --model coco_resnet_cnninit_review_ext_standard_bestll.npz --split test

## Reference

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
[5]: http://www.statmt.org/wmt11/translation-task.html#download
