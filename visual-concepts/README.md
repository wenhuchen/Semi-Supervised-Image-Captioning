### Visual Concept Detector 
Many thanks to [visual concept](1) repository, our visual concept detector is largely contributed to their marvelous work.


#### Installation Instructions ####
Please refer to [visual-concept](1) for more detail, you can directly download their pre-trained model and use it in our project.
	
	wget ftp://ftp.cs.berkeley.edu/pub/projects/vision/im2cap-cvpr15b/trained-coco.v2.tgz
You can extrac the tarball file and move the pre-trained caffee model to output/vgg/ folder for further step.
 
#### Visual concept and regional features ####
Our model can extract both the visual concepts and regional features out of the given images. 
	
	python eval.py --test_json ../Data/coco/cocotalk_val.json --dataset coco --split val --order 30

Parameter list

	--test_json refers to the json file containing id and path of all images
	--dataset refers to the dataset you want to use, COCO|Flickr30K 
	--split chooses the split to run the detection
	--order generates the top n words based on their mil probabilities
The result files will be put in ../Data/{dataset} folder, the h5 file name will be "Feats_{split}.h5" and the json file name will be "visual\_concept\_{split}.json".


#### Training, Testing the model ####
All the relevant commands are stored in the file ``scripts/scripts_all.py``, make sure to have GPU memory larger than 10G to train the model. 

#### Citing
If you find this codebase useful in your research, please consider citing the following paper:

    @InProceedings{Fang_2015_CVPR,
      author = {Fang, Hao and Gupta, Saurabh and Iandola, Forrest and Srivastava, Rupesh K. and Deng, Li and Dollar, Piotr and Gao, Jianfeng and He, Xiaodong and Mitchell, Margaret and Platt, John C. and Lawrence Zitnick, C. and Zweig, Geoffrey},
      title = {From Captions to Visual Concepts and Back},
      booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      month = {June},
      year = {2015}
    }

[1]:https://github.com/s-gupta/visual-concepts