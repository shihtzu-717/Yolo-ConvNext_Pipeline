
###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object detections applying the following metrics:     #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012)                  #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012)                  #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: Updated with applied patch                                    #
###########################################################################################

import argparse
import glob
import os
import shutil
import sys
import math

import _init_paths
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils import BBFormat

# Validate formats
def ValidateFormats(argFormat, argName, errors):
    if argFormat == 'xywh':
        return BBFormat.XYWH
    elif argFormat == 'xyrb':
        return BBFormat.XYX2Y2
    elif argFormat is None:
        return BBFormat.XYWH  # default when nothing is passed
    else:
        errors.append(f"argument {argName}: invalid value. It must be either 'xywh' or 'xyrb'")

# Get bounding boxes from files
def getBoundingBoxes(directory, isGT, bbFormat, coordType, allBoundingBoxes=None, allClasses=None, imgSize=(0, 0)):
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []
    os.chdir(directory)
    files = glob.glob("*.txt")
    files.sort()

    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(f, "r")
        for line in fh1:
            line = line.strip()
            if line == '':
                continue
            splitLine = line.split(" ")
            if isGT:
                idClass = splitLine[0]  
                x, y, w, h = map(float, splitLine[1:5])
                bb = BoundingBox(nameOfImage, idClass, x, y, w, h, coordType, imgSize, BBType.GroundTruth, format=bbFormat)
            else:
                idClass = splitLine[0]
                confidence = float(splitLine[1])
                x, y, w, h = map(float, splitLine[2:6])
                bb = BoundingBox(nameOfImage, idClass, x, y, w, h, coordType, imgSize, BBType.Detected, confidence, format=bbFormat)
            allBoundingBoxes.addBoundingBox(bb)

            if idClass not in allClasses:
                allClasses.append(idClass)
        fh1.close()
    
    # print(f"Loaded {len(allBoundingBoxes.getBoundingBoxes())} bounding boxes from {directory}")
    if isGT:
        print(f"Loaded {len(allBoundingBoxes.getBoundingBoxesByType(BBType.GroundTruth))} bounding boxes from {directory}")
    else:
        print(f"Loaded {len(allBoundingBoxes.getBoundingBoxesByType(BBType.Detected))} bounding boxes from {directory}")
    return allBoundingBoxes, allClasses

# Main processing function
def process_evaluation(gtFolder, detFolder, class_num, threshold=0.25, iouThreshold=0.5, savePath="results", showPlot=True):
    try:
        os.makedirs(savePath, exist_ok=True)
        print(f"Directory '{savePath}' created successfully or already exists.")
    except Exception as e:
        print(f"Error creating directory '{savePath}': {e}")
        return

    allBoundingBoxes, allClasses = getBoundingBoxes(gtFolder, True, BBFormat.XYWH, CoordinatesType.Absolute)
    allBoundingBoxes, allClasses = getBoundingBoxes(detFolder, False, BBFormat.XYWH, CoordinatesType.Absolute, allBoundingBoxes, allClasses)
    
    evaluator = Evaluator()
    detections = evaluator.PlotPrecisionRecallCurve(allBoundingBoxes, conf_threshold=threshold, IOUThreshold=iouThreshold,
                                                    method=MethodAveragePrecision.EveryPointInterpolation, showAP=True, showInterpolatedPrecision=True, savePath=savePath, showGraphic=False)
    
    ap_values = [m['AP'] for m in detections]
    total_tp = [m['total TP'] for m in detections]
    total_fp = [m['total FP'] for m in detections]
    total_fn = [m['total FN'] for m in detections]

    # Filter out nan values
    ap_values = [ap for ap in ap_values if not math.isnan(ap)]

    cur_precison = float(sum(total_tp) / (sum(total_tp) + sum(total_fp)))
    cur_recall = float(sum(total_tp) / (sum(total_tp) + sum(total_fn)))
    f1_score = 2 * cur_precison * cur_recall / (cur_precison + cur_recall)
    mAP = sum(ap_values) / class_num if ap_values else 0
    error_rate_FN = float((sum(total_fp) + sum(total_fn)) / (sum(total_tp) + sum(total_fp) + sum(total_fn))) if float(sum(total_tp) + sum(total_fp) + sum(total_fn)) > 0  else 0.0
    FDR = float((sum(total_fp)) / (sum(total_tp) + sum(total_fp))) if float(
        sum(total_tp) + sum(total_fp)) > 0 else 0.0
    accuracy = float(sum(total_tp) / (sum(total_tp) + sum(total_fn))) if sum(total_tp) + sum(total_fn) > 0 else 0.0

    print(f" for conf_thresh = {threshold}, precision = {cur_precison:.2}, recall = {cur_recall:.2}, F1-score = {f1_score:.2}")
    print(f" for conf_thresh = {threshold}, TP = {int(sum(total_tp))}, FP = {int(sum(total_fp))}, FN = {int(sum(total_fn))}")
    print(f" mAP@{iouThreshold} = {mAP:.2%}")
    print(f" mean_FDR = {FDR:.2%}")

    return mAP

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate object detection using Pascal VOC metrics")
    parser.add_argument("-gt", "--gtfolder", default="groundtruths", help="Folder containing ground truth bounding boxes")
    parser.add_argument("-det", "--detfolder", default="detections", help="Folder containing detected bounding boxes")
    parser.add_argument("-t", "--threshold", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("-iou", "--iou_threshold", type=float, default=0.5, help="IOU threshold (default: 0.5)")
    parser.add_argument("-sp", "--savepath", required=True, help="Folder to save results")
    parser.add_argument("-np", "--noplot", action="store_false", help="Disable plotting")
    parser.add_argument("-cl", "--class_num", type=int, default=10, help="Disable plotting")

    args = parser.parse_args()
    process_evaluation(args.gtfolder, args.detfolder, args.class_num, args.threshold, args.iou_threshold, args.savepath, args.noplot)
