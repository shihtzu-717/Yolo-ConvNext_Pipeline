import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from BoundingBox import *
from BoundingBoxes import *
from utils import *

class Evaluator:
    def GetPascalVOCMetrics(
        self,
        boundingboxes,
        conf_threshold=0.25,            # TP/FP 계산 시 적용할 confidence threshold
        IOUThreshold=0.5,               # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation
    ):
        """
        Args:
            boundingboxes: BoundingBoxes 객체 (GT + Det)
            conf_threshold (float): TP/FP 계산용 confidence threshold
            IOUThreshold (float): IOU가 이 값 이상이면 TP로 간주
            method: AP 계산 방식 (EveryPointInterpolation or ElevenPointInterpolation)
        Returns:
            클래스별 평가 결과 list
        """

        ret = []
        groundTruths = []
        detections  = []
        classes = []

        # 1) 모든 바운딩 박스를 순회하며 GT / 검출을 분리
        for bb in boundingboxes.getBoundingBoxes():
            if bb.getBBType() == BBType.GroundTruth:
                groundTruths.append([
                    bb.getImageName(),
                    bb.getClassId(),
                    1,  # Confidence=1 (GT)
                    bb.getAbsoluteBoundingBox(BBFormat.XYWH)
                ])
            else:
                detections.append([
                    bb.getImageName(),
                    bb.getClassId(),
                    bb.getConfidence(),
                    bb.getAbsoluteBoundingBox(BBFormat.XYWH)
                ])
            if bb.getClassId() not in classes:
                classes.append(bb.getClassId())
        classes = sorted(classes)

        # 출력용 헤더
        print(f"{'Class ID':<10} {'AP (%)':>7} {'TP':>5} {'FP':>5} {'Precision (%)':>15} {'Recall (%)':>11}")
        print("=" * 60)

        # 2) 클래스별 평가
        for c in classes:
            # 2-1) 해당 클래스의 GT / 검출만 모으기
            dects = [d for d in detections if d[1] == c]
            gts   = {}
            npos  = 0  # GT(Positive) 총 개수

            for g in groundTruths:
                if g[1] == c:
                    npos += 1
                    gts[g[0]] = gts.get(g[0], []) + [g]

            # confidence가 높은 순으로 소팅 (전체 검출)
            dects = sorted(dects, key=lambda x: x[2], reverse=True)

            # ---------------------------------------------
            # (A) AP 계산용: "모든" 검출 사용 (threshold 미적용)
            # ---------------------------------------------
            ap_TP = np.zeros(len(dects))
            ap_FP = np.zeros(len(dects))
            # 각 이미지별로 GT가 몇 개 있고, 매칭 여부(0=미매칭,1=매칭) 관리
            ap_det = {key: np.zeros(len(gts[key])) for key in gts}

            for i in range(len(dects)):
                imgName, _, detConf, detBox = dects[i]
                gtArr = gts[imgName] if imgName in gts else []

                iouMax = sys.float_info.min
                jmax   = -1
                for j in range(len(gtArr)):
                    iouVal = Evaluator.iou_yolo(detBox, gtArr[j][3])
                    if iouVal > iouMax:
                        iouMax = iouVal
                        jmax   = j

                if iouMax >= IOUThreshold:
                    if ap_det[imgName][jmax] == 0:
                        ap_TP[i] = 1
                        ap_det[imgName][jmax] = 1
                    else:
                        ap_FP[i] = 1
                else:
                    ap_FP[i] = 1

            # ap_TP / ap_FP 누적
            acc_AP_FP = np.cumsum(ap_FP)
            acc_AP_TP = np.cumsum(ap_TP)

            if npos == 0:
                # 해당 클래스에 GT가 아예 없다면
                rec = np.zeros_like(acc_AP_TP)
            else:
                rec = acc_AP_TP / npos
            prec = np.divide(acc_AP_TP, (acc_AP_FP + acc_AP_TP))

            # AP 계산
            if method == MethodAveragePrecision.EveryPointInterpolation:
                [ap, mpre, mrec, ii] = Evaluator.CalculateAveragePrecision(rec, prec)
            else:
                [ap, mpre, mrec, _] = Evaluator.ElevenPointInterpolatedAP(rec, prec)

            # ----------------------------------------------------
            # (B) Precision/Recall/TP/FP 계산용: threshold=0.25 이상
            # ----------------------------------------------------
            tp_fp_dects = [d for d in dects if d[2] >= conf_threshold]  # 필터링
            TP = np.zeros(len(tp_fp_dects))
            FP = np.zeros(len(tp_fp_dects))

            # 별도의 매칭 딕셔너리 (ap_det와 별개)
            det_25 = {key: np.zeros(len(gts[key])) for key in gts}

            for i in range(len(tp_fp_dects)):
                imgName2, _, detConf2, detBox2 = tp_fp_dects[i]
                gtArr2 = gts[imgName2] if imgName2 in gts else []

                iouMax2 = sys.float_info.min
                jmax2   = -1
                for j in range(len(gtArr2)):
                    iouVal2 = Evaluator.iou_yolo(detBox2, gtArr2[j][3])
                    if iouVal2 > iouMax2:
                        iouMax2 = iouVal2
                        jmax2   = j

                if iouMax2 >= IOUThreshold:
                    if det_25[imgName2][jmax2] == 0:
                        TP[i] = 1
                        det_25[imgName2][jmax2] = 1
                    else:
                        FP[i] = 1
                else:
                    FP[i] = 1

            TP_sum = np.sum(TP)
            FP_sum = np.sum(FP)
            FN_sum = npos - TP_sum

            # 최종 Precision/Recall 계산 (단일 threshold=0.25 기준)
            class_precision = (
                TP_sum / (TP_sum + FP_sum) if (TP_sum + FP_sum) > 0 else 0.0
            )
            class_recall = (
                TP_sum / (TP_sum + FN_sum) if npos > 0 else 0.0
            )

            # 출력 및 결과 저장
            print(
                f"{c:<10} {ap*100:>6.2f}% {int(TP_sum):>5} {int(FP_sum):>5}"
                f" {class_precision*100:>13.2f}% {class_recall*100:>10.2f}%"
            )

            r = {
                "class": c,
                "precision": prec,              # (A)에서 구한 전체 곡선의 prec (참고용)
                "recall": rec,                  # (A)에서 구한 전체 곡선의 rec  (참고용)
                "AP": ap,                       # 전체 detections로 구한 AP
                "interpolated precision": mpre, # AP 계산 시 보간된 정밀도
                "interpolated recall": mrec,    # AP 계산 시 보간된 재현율
                "total positives": npos,
                "total TP": TP_sum,             # (B) 단일 threshold=0.25일 때 TP
                "total FP": FP_sum,             # (B) 단일 threshold=0.25일 때 FP
                "total FN": FN_sum,             # (B) 단일 threshold=0.25일 때 FN
            }
            ret.append(r)

        print("=" * 60)
        return ret

    def PlotPrecisionRecallCurve(
        self,
        boundingBoxes,
        conf_threshold=0.25,
        IOUThreshold=0.45,
        method=MethodAveragePrecision.EveryPointInterpolation,
        showAP=False,
        showInterpolatedPrecision=False,
        savePath=None,
        showGraphic=True
    ):
        """Precision-Recall Curve를 그리고 저장"""

        results = self.GetPascalVOCMetrics(
            boundingBoxes,
            conf_threshold=conf_threshold,  # ← (B) 부분에서 0.25 적용
            IOUThreshold=IOUThreshold,
            method=method
        )

        if not results:
            print("⚠️ No classes found. Skipping PR curve generation.")
            return results

        for result in results:
            if result is None:
                continue

            classId = result.get("class", None)
            precision = result["precision"]                   # (A)에서 구한 전체 PR 곡선
            recall    = result["recall"]
            average_precision = result["AP"]
            mpre = result["interpolated precision"]
            mrec = result["interpolated recall"]

            # 그래프 그리기
            plt.close()

            if showInterpolatedPrecision:
                if method == MethodAveragePrecision.EveryPointInterpolation:
                    plt.plot(mrec, mpre, "--r", label="Interpolated (every point)")
                elif method == MethodAveragePrecision.ElevenPointInterpolation:
                    nrec, nprec = [], []
                    for idx in range(len(mrec)):
                        r = mrec[idx]
                        if r not in nrec:
                            idxEq = np.argwhere(mrec == r)
                            nrec.append(r)
                            nprec.append(max([mpre[int(id)] for id in idxEq]))
                    plt.plot(nrec, nprec, "or", label="11-point interpolated")

            plt.plot(recall, precision, label="Precision")
            plt.xlabel("recall")
            plt.ylabel("precision")

            if showAP:
                ap_str = "{0:.2f}%".format(average_precision * 100)
                plt.title(f"Precision x Recall curve\nClass: {classId}, AP: {ap_str}")
            else:
                plt.title(f"Precision x Recall curve\nClass: {classId}")

            plt.legend(shadow=True)
            plt.grid()

            # 결과 저장
            if savePath is not None:
                filename = f"{classId}.png" if classId is not None else "no_class.png"
                plt.savefig(os.path.join(savePath, filename))

            if showGraphic:
                plt.show()
                plt.pause(0.05)

        return results

    @staticmethod
    def CalculateAveragePrecision(rec, prec):
        """ Every-Point Interpolation 방식으로 AP 계산 """
        mrec = [0] + list(rec) + [1]
        mpre = [0] + list(prec) + [0]

        # Precision 보정 (단조 감소하도록)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        # 실제 AP 계산
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[i + 1] != mrec[i]:
                ii.append(i + 1)

        ap = 0
        for i in ii:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]

        # mpre, mrec에서 처음/끝(0,1) 제외
        return [ap, mpre[1:-1], mrec[1:-1], ii]

    @staticmethod
    def ElevenPointInterpolatedAP(rec, prec):
        """ VOC 2007 11-point 방식으로 AP 계산 """
        mrec = list(rec)
        mpre = list(prec)
        recallValues = np.linspace(0, 1, 11)[::-1]
        rhoInterp = []
        recallValid = []

        for r in recallValues:
            # r 이상인 recall 지점의 precision 중 최댓값
            argGreaterRecalls = np.argwhere(mrec[:] >= r)
            pmax = 0
            if argGreaterRecalls.size != 0:
                pmax = max(mpre[argGreaterRecalls.min():])
            recallValid.append(r)
            rhoInterp.append(pmax)

        ap = sum(rhoInterp) / 11.0

        # 그래프용 포인트 생성
        rvals = [recallValid[0]] + recallValid + [0]
        pvals = [0] + rhoInterp + [0]
        cc = []
        for i in range(len(rvals)):
            p1 = (rvals[i], pvals[i - 1])
            p2 = (rvals[i], pvals[i])
            if p1 not in cc:
                cc.append(p1)
            if p2 not in cc:
                cc.append(p2)

        recallGraph = [i[0] for i in cc]
        precGraph   = [i[1] for i in cc]
        return [ap, precGraph, recallGraph, None]

    # ---------------------------
    #    iou, iou_yolo 함수들
    # ---------------------------
    @staticmethod
    def iou(boxA, boxB):
        """ boxA, boxB: [x1, y1, x2, y2] """
        if Evaluator._boxesIntersect(boxA, boxB) is False:
            return 0
        interArea = Evaluator._getIntersectionArea(boxA, boxB)
        union     = Evaluator._getUnionAreas(boxA, boxB, interArea=interArea)
        return interArea / union if union > 0 else 0

    @staticmethod
    def iou_yolo(boxA, boxB):
        """ boxA, boxB: [x, y, w, h] (YOLO 형식) """
        if Evaluator._boxesIntersect_yolo(boxA, boxB) is False:
            return 0
        interArea = Evaluator._getIntersectionArea_yolo(boxA, boxB)
        union     = Evaluator._getUnionAreas_yolo(boxA, boxB, interArea=interArea)
        return interArea / union if union > 0 else 0

    @staticmethod
    def _boxesIntersect(boxA, boxB):
        if boxA[0] > boxB[2]: return False
        if boxB[0] > boxA[2]: return False
        if boxA[3] < boxB[1]: return False
        if boxA[1] > boxB[3]: return False
        return True

    @staticmethod
    def _boxesIntersect_yolo(boxA, boxB):
        # boxA: (x_c, y_c, w, h)
        if boxA[0] - (boxA[2]/2) > boxB[0] + (boxB[2]/2): return False
        if boxA[0] + (boxA[2]/2) < boxB[0] - (boxB[2]/2): return False
        if boxA[1] + (boxA[3]/2) < boxB[1] - (boxB[3]/2): return False
        if boxA[1] - (boxA[3]/2) > boxB[1] + (boxB[3]/2): return False
        return True

    @staticmethod
    def _getIntersectionArea(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        return (xB - xA + 1) * (yB - yA + 1)

    @staticmethod
    def _getIntersectionArea_yolo(boxA, boxB):
        xA = max(boxA[0] - (boxA[2]/2), boxB[0] - (boxB[2]/2))
        yA = max(boxA[1] - (boxA[3]/2), boxB[1] - (boxB[3]/2))
        xB = min(boxA[0] + (boxA[2]/2), boxB[0] + (boxB[2]/2))
        yB = min(boxA[1] + (boxA[3]/2), boxB[1] + (boxB[3]/2))
        return (xB - xA) * (yB - yA)

    @staticmethod
    def _getUnionAreas(boxA, boxB, interArea=None):
        areaA = Evaluator._getArea(boxA)
        areaB = Evaluator._getArea(boxB)
        if interArea is None:
            interArea = Evaluator._getIntersectionArea(boxA, boxB)
        return float(areaA + areaB - interArea)

    @staticmethod
    def _getUnionAreas_yolo(boxA, boxB, interArea=None):
        areaA = Evaluator._getArea_yolo(boxA)
        areaB = Evaluator._getArea_yolo(boxB)
        if interArea is None:
            interArea = Evaluator._getIntersectionArea_yolo(boxA, boxB)
        return float(areaA + areaB - interArea)

    @staticmethod
    def _getArea(box):
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

    @staticmethod
    def _getArea_yolo(box):
        return box[2] * box[3]
