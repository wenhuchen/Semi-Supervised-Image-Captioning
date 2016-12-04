import matplotlib
matplotlib.use('Agg')
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
from json import encoder
import os.path as osp
import numpy as np
import operator
import os
current_path = osp.dirname(osp.realpath(__file__))

class cocoEvaluation(object):
    def __init__(self, dataset='coco'):
        self.dataType='val2014'
        if 'coco' in dataset:
            annFile=osp.join(current_path, 'annotations/captions_%s.json'%self.dataType)
            self.dataset = 'coco'
            self.imgDir='$WORK/coco-dataset/val2014/'
        elif 'commoncrawl' in dataset:
            annFile=osp.join(current_path, 'annotations/%s_captions_%s.json'%('commoncrawl', self.dataType))
        elif 'commonvisual' in dataset:
            annFile=osp.join(current_path, 'annotations/%s_captions_%s.json'%('commonvisual', self.dataType))
        elif 'flickr' in dataset:
            annFile=osp.join(current_path, 'annotations/%s_captions_%s.json'%('flickr', self.dataType))
            self.dataset = 'flickr'
            self.imgDir='$WORK/flick30k-dataset/flickr30k-images/'

            
        self.coco = COCO(annFile)
        self.subtypes=['results', 'evalImgs', 'eval']

    def transform(self, imgId):
        if self.dataset == 'coco':
            return self.imgDir + 'COCO_val2014_' + '%.12d'% imgId + '.jpg'
        elif self.dataset == 'flickr':
            return self.imgDir + str(imgId) + '.jpg'

    def evaluate(self, resDir, display=False):
        [resFile, evalImgsFile, evalFile]= ['%s/captions_%s_%s.json'%(resDir, self.dataType, subtype) \
                                                                        for subtype in self.subtypes]
        cocoRes = self.coco.loadRes(resFile)
        cocoEval = COCOEvalCap(self.coco, cocoRes)
        cocoEval.params['image_id'] = cocoRes.getImgIds()
        cocoEval.evaluate()

        # demo how to use evalImgs to retrieve low score result
        indexes = np.argsort([eva['CIDEr'] for eva in cocoEval.evalImgs], axis=0)[:5]
        imgIds = [cocoEval.evalImgs[index]['image_id'] for index in indexes]
    
        if display:
            for i, imgId in zip(indexes, imgIds):
                print 'image Id %d, CIDEr score %f'%(imgId, cocoEval.evalImgs[i]['CIDEr'])
                annIds = self.coco.getAnnIds(imgIds=imgId)
                anns = self.coco.loadAnns(annIds)
                self.coco.showAnns(anns)
                print " "
                annIds = cocoRes.getAnnIds(imgIds=imgId)
                anns = cocoRes.loadAnns(annIds)
                cocoRes.showAnns(anns)
                os.system('display ' + self.transform(imgId))
                raw_input("Press Enter to continue...")

        # save evaluation results to ./results folder
        json.dump(cocoEval.evalImgs, open(evalImgsFile, 'w'))
        json.dump(cocoEval.eval,     open(evalFile, 'w'))
        return cocoEval.eval

    def compare(self, resDir1, resDir2):
        [resFile1, evalImgsFile1, evalFile2]= ['%s/captions_%s_%s.json'%(resDir1, self.dataType, subtype) \
                                                                        for subtype in self.subtypes]
        [resFile2, evalImgsFile2, evalFile2]= ['%s/captions_%s_%s.json'%(resDir2, self.dataType, subtype) \
                                                                        for subtype in self.subtypes]
        cocoRes1 = self.coco.loadRes(resFile1)
        cocoEval1 = COCOEvalCap(self.coco, cocoRes1)
        cocoEval1.params['image_id'] = cocoRes1.getImgIds()
        cocoEval1.evaluate()

        cocoRes2 = self.coco.loadRes(resFile2)
        cocoEval2 = COCOEvalCap(self.coco, cocoRes2)
        cocoEval2.params['image_id'] = cocoRes2.getImgIds()
        cocoEval2.evaluate()

        cider_map1 = {eva['image_id']:eva['CIDEr'] for eva in cocoEval1.evalImgs}
        cider_map2 = {eva['image_id']:eva['CIDEr'] for eva in cocoEval2.evalImgs}
    
        counts, diff_map = [0, 0, 0], {}
        for k in cider_map1:
            diff_map[k] = cider_map1[k]-cider_map2[k]
            if diff_map[k] == 0:
                counts[0] += 1
            elif diff_map[k] < 0:
                counts[1] += 1
            else:
                counts[2] += 1
        print "same score: %d, worse score: %d, better score: %d"%(counts[0], counts[1], counts[2])
        sorted_map = sorted(diff_map.items(), key=operator.itemgetter(1))[:50]

        for imgId, cider_diff in sorted_map:
            annIds = self.coco.getAnnIds(imgIds=imgId)
            anns = self.coco.loadAnns(annIds)
            self.coco.showAnns(anns)
            print " "
            annIds = cocoRes1.getAnnIds(imgIds=imgId)
            anns = cocoRes1.loadAnns(annIds)
            cocoRes1.showAnns(anns) 
            print " "
            annIds = cocoRes2.getAnnIds(imgIds=imgId)
            anns = cocoRes2.loadAnns(annIds)
            cocoRes2.showAnns(anns)
            print " "
            print "imgid:%d cider1:%0.2f cider2:%0.2f"%(imgId, cider_map1[imgId], cider_map2[imgId])
            os.system('display ' + self.imgDir + 'COCO_val2014_' + '%.12d'% imgId + '.jpg')
            raw_input("Press Enter to continue...")
