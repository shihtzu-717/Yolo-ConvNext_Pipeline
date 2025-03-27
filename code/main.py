import sys
import glob
import os
import json
import shutil

from tqdm import tqdm

import detection
import classification
import mAP

# 설정 파일 경로
CONFIG_PATH = "conf.json"

def load_config(config_path):
    """JSON 설정 파일 로드"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

# === 실행 가능한 단계 목록 ===
STEPS = {
    "edge": "Edge Model 탐지",
    "server": "Server Model 탐지",
    "classification": "Classification Model 실행"
}

def run_edge_model(config):
    """Edge Model 탐지 실행"""
    print("🚀 Edge Model 탐지 실행 중...")
    try:
        detection.yolo_inference(config, 'edge')
        print("✅ Edge Model 탐지 완료!")
        return True
    except Exception as e:
        print(f"❌ Edge Model 실행 중 오류 발생: {str(e)}")
        return False

def run_server_model(config, filelist=None):
    """Server Model 탐지 실행"""
    print("🚀 Server Model 탐지 실행 중...")
    try:
        detection.yolo_inference(config, 'server', filelist)
        print("✅ Server Model 탐지 완료!")
        return True
    except Exception as e:
        print(f"❌ Server Model 실행 중 오류 발생: {str(e)}")
        return False

def run_classification_model(config):
    """Classification Model 실행"""
    print("🚀 Classification Model 실행 중...")
    classification.convnext_inference(config, 'classification')
    print("✅ Classification Model 완료!")

    # classification 결과 파일 읽기
    classification_results_path = os.path.join(config["output_dir"], 'classification', "classification_pred.json")
    if not os.path.exists(classification_results_path):
        print("❌ Classification 결과 파일을 찾을 수 없습니다.")
        return None
        
    with open(classification_results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
        
    return results

def run_evaluation(gt_data_path, det_data_path, config, step):
    """성능 평가 실행 (항상 마지막에 실행)"""
    print("🚀 성능 평가 실행 중...")

    results_data_path = os.path.join(config["output_dir"], "evaluate")
    class_num = 10
    results = mAP.process_evaluation(gt_data_path, det_data_path, class_num, threshold=float(config['thresh']), iouThreshold=float(config['iou_thresh']), savePath=results_data_path, showPlot=False)
    print("✅ 성능 평가 완료!")

def remove_non_object_bboxes(config, annot_path):
    """
    탐지된 객체가 있는 어노테이션 파일을 찾고, 해당 이미지 경로 리스트를 반환하는 함수.

    Args:
        config (dict): 설정 파일 (input_path 포함)
        annot_path (str): 어노테이션 파일이 저장된 경로

    Returns:
        list: 존재하는 이미지 경로 리스트
    """
    remove_image_list = []
    valid_image_paths = []
    edge_annot_list = glob.glob(os.path.join(annot_path, "**/*.txt"), recursive=True)

    for edge_annot in edge_annot_list:
        with open(edge_annot, 'r') as f:
            lines = f.readlines()
            if not lines:  # 빈 파일이면 스킵
                remove_image_list.append(edge_annot)
                continue

        # 어노테이션 파일명에서 이미지 파일명 생성
        annot_filename = os.path.basename(edge_annot)  # "example.txt"
        image_filename = os.path.splitext(annot_filename)[0]  # "example"

        # 가능한 이미지 확장자
        possible_extensions = ["jpg", "png"]
        for ext in possible_extensions:
            image_path = os.path.join(config["input_path"], "images", f"{image_filename}.{ext}")
            if os.path.exists(image_path):  # 실제 존재하는 파일만 추가
                valid_image_paths.append(image_path)
                break  # 한 개만 추가하면 되므로 중단
            
    return valid_image_paths, remove_image_list  # 탐지된 객체가 있는 이미지 경로 리스트, 없는 경로 리스트 둘 다 반환


def remove_non_pothole_bboxes(config, annot_dir, new_annot_dir, classification_results=None, debug_mode=False):
    """
    Args:
        classification_results (list, optional): 이미 로드된 classification 결과. 
            None인 경우 파일에서 읽음
    """
    if classification_results is None:
        json_path = os.path.join(config["output_dir"], 'classification', 'classification_pred.json')
        if not os.path.exists(json_path):
            print(f"❌ JSON 파일을 찾을 수 없습니다: {json_path}")
            return
        try:
            with open(json_path, "r") as f:
                classification_results = json.load(f)
        except json.JSONDecodeError:
            print("❌ JSON 파일을 읽는 중 오류 발생!")
            return

    if not os.path.exists(new_annot_dir):
        os.makedirs(new_annot_dir, exist_ok=True)

    # 2️⃣ is_pothole=False && class_id == 0 인 바운딩 박스 목록 생성
    non_pothole_bboxes = {}
    for entry in classification_results:
        if entry["class_id"] == 0 and not entry["is_pothole"]:
            img_filename = os.path.splitext(entry["img_filename"])[0]  # 확장자 제거
            bbox = entry["bbox"]
            if img_filename not in non_pothole_bboxes:
                non_pothole_bboxes[img_filename] = []
            non_pothole_bboxes[img_filename].append(bbox)

    # 3️⃣ 해당 bbox가 포함된 어노테이션 txt 파일 수정
    modified_files = set()  # 수정된 파일 목록 저장
    for img_name, bbox_list in non_pothole_bboxes.items():
        txt_path = os.path.join(annot_dir, f"{img_name}.txt")
        new_txt_path = os.path.join(new_annot_dir, f"{img_name}.txt")

        if not os.path.exists(txt_path):
            if debug_mode:
                print(f"⚠️ TXT 어노테이션 파일이 존재하지 않음: {txt_path}")
            continue

        before_lines = []  # 기존 어노테이션 저장
        new_lines = []  # 수정된 어노테이션 저장

        with open(txt_path, "r") as f:
            for line in f:
                before_lines.append(line.strip())  # 기존 내용 저장
                parts = line.strip().split()
                if len(parts) != 6:
                    continue

                cls_id = int(parts[0])
                bbox = list(map(float, parts[2:]))  # [x_center, y_center, width, height]

                # **0번 클래스 중 is_pothole=False인 바운딩 박스 제거**
                if cls_id == 0 and bbox in bbox_list:
                    if debug_mode:
                        print(f"🚨 [제거됨] {txt_path} → {bbox}") # 제거 대상이면 저장하지 않음
                    continue

                new_lines.append(line.strip())

        # 4️⃣ 기존 어노테이션과 비교하여 출력 (디버그 모드에서만)
        if debug_mode:
            print("\n====================================")
            print(f"📂 {img_name}.txt 변경 내용")
            print("📌 [Before] 기존 어노테이션:")
            for line in before_lines:
                print(f"   {line}")

            print("\n📌 [After] 수정된 어노테이션:")
            if new_lines:
                for line in new_lines:
                    print(f"   {line}")
            else:
                print("   🚫 모든 0번 클래스 바운딩 박스 제거됨")

            print("====================================\n")

        # 5️⃣ 수정된 어노테이션 파일 저장 (빈 파일이라도 생성)
        with open(new_txt_path, "w") as f:
            if new_lines:
                f.write("\n".join(new_lines) + "\n")
            else:
                f.write("")  # 빈 파일 생성 (0KB 유지)

        modified_files.add(img_name)  # 수정된 파일 기록
    print(annot_dir)
    # 6️⃣ 수정되지 않은 파일을 그대로 복사
    for txt_file in os.listdir(annot_dir):
        img_name, ext = os.path.splitext(txt_file)
        if ext == ".txt" and img_name not in modified_files:  # 수정된 파일이 아니면 복사
            src_path = os.path.join(annot_dir, txt_file)
            dest_path = os.path.join(new_annot_dir, txt_file)
            shutil.copy2(src_path, dest_path)
            if debug_mode:
                print(f"✅ [복사됨] {txt_file} → 수정하지 않은 파일은 server/annots에서 class/annots로 파일 복사")

    if debug_mode:
        print(f"✅ 모든 TXT 어노테이션 파일 업데이트 완료!")


def validate_paths(config):
    """필요한 경로들이 존재하는지 확인"""
    paths = {
        "input_path": config["input_path"],
        "output_dir": config["output_dir"],
        # "edge_output": os.path.join(config["output_dir"], 'edge'),
        # "server_output": os.path.join(config["output_dir"], 'server'),
        # "classification_output": os.path.join(config["output_dir"], 'classification')
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            try:
                os.makedirs(path)
                print(f"✅ 생성됨: {path}")
            except Exception as e:
                print(f"❌ 경로 생성 실패 ({name}): {path}")
                print(f"   오류: {str(e)}")
                return False
    return True

def check_previous_step_results(config, step):
    """이전 단계의 결과가 존재하는지 확인"""
    if step == "server":
        edge_annot_path = os.path.join(config["output_dir"], 'edge', "annotations")
        if not os.path.exists(edge_annot_path):
            print(f"❌ Edge 모델의 결과가 없습니다: {edge_annot_path}")
            return False
            
    elif step == "classification":
        server_annot_path = os.path.join(config["output_dir"], 'server', "annotations")
        if not os.path.exists(server_annot_path):
            print(f"❌ Server 모델의 결과가 없습니다: {server_annot_path}")
            return False
    return True


def load_edge_none_object_data(remove_file_list, server_data_path):
    print(f"📂 성능 지표 계산을 위해 Edge 모델이 미탐지한 데이터 {len(remove_file_list)} 개를 불러옵니다.")

    for remove_file in tqdm(remove_file_list):
        shutil.copy2(remove_file, server_data_path)

def main():
    # 1️⃣ 설정 로드
    remove_file_list = []
    config = load_config(CONFIG_PATH)
    
    # 경로 검증
    if not validate_paths(config):
        print("⚠️ 필요한 경로 생성에 실패했습니다. 프로그램을 종료합니다.")
        return

    gt_data_path = os.path.join(config["input_path"], "annotations")
    edge_data_path = os.path.join(config["output_dir"], 'edge', "annotations")
    server_data_path = os.path.join(config["output_dir"], 'server', "annotations")
    classification_data_path = os.path.join(config["output_dir"], 'classification', "annotations")

    # 실행할 단계 확인
    selected_steps = config.get("model_pipeline", [])
    selected_steps = [step for step in selected_steps if step in STEPS]

    if not selected_steps:
        print("⚠️ 실행할 단계가 설정되지 않았습니다. 프로그램을 종료합니다.")
        return

    print(f"🔹 실행할 단계: {', '.join(selected_steps)}")

    # 2️⃣ 선택된 단계 실행
    for step in selected_steps:
        try:
            # 각 단계별 선행 조건 확인
            if step == "classification":  # classification 단계만 체크
                if not check_previous_step_results(config, step):
                    print(f"⚠️ Classification 단계 실행을 건너뜁니다 (Server 모델 결과 없음)")
                    continue

            if step == "edge":
                if not os.path.exists(edge_data_path):
                    os.makedirs(edge_data_path, exist_ok=True)

                result = run_edge_model(config)
                if not result:
                    print("❌ Edge 모델 실행 실패")
                    continue

            elif step == "server":
                if not os.path.exists(server_data_path):
                    os.makedirs(server_data_path, exist_ok=True)

                # edge 모델 결과가 있으면 사용, 없으면 전체 데이터셋에 대해 실행
                image_folder = os.path.join(config["input_path"], "images")
                num_gt_data = len([f for f in glob.glob(os.path.join(image_folder, "*.jpg"))])

                if os.path.exists(edge_data_path):
                    file_list, remove_file_list = remove_non_object_bboxes(config, edge_data_path)
                    if len(remove_file_list) == 0 and len(file_list) == num_gt_data:
                        print("⚠️  Edge 모델 결과 모든 이미지에서 객체가 탐지되어 전체 데이터셋을 처리합니다.")
                        file_list = None
                    elif len(remove_file_list) == 0 and len(file_list) != num_gt_data:
                        print(f"⚠️ Edge 모델 결과: {len(file_list)}개, 전체 데이터셋: {num_gt_data}개")
                        print("⚠️ Edge 모델 결과와 전체 입력 데이터의 수가 다릅니다. 전체 데이터셋을 처리합니다.")
                        file_list = None
                    elif len(remove_file_list) > 0:
                        if len(remove_file_list) == num_gt_data:
                            print("⚠️ Edge 모델 결과, 모든 입력 데이터가 탐지되지 않아 성능 평가를 종료합니다.")
                            sys.exit(1)
                        else:
                            print("⚠️ Edge 모델 결과 아래 이미지에서 객체가 탐지되지 않아 제외한 나머지 데이터셋을 처리합니다.")
                            for f in remove_file_list:
                                print(f)

                else:
                    print("⚠️ Edge 모델 결과가 없어 전체 데이터셋을 처리합니다.")
                    file_list = None
                
                success = run_server_model(config, file_list)

                # edge 모델이 미탐지한 데이터 가져오기
                if os.path.exists(edge_data_path) and remove_file_list:
                    load_edge_none_object_data(remove_file_list, server_data_path)

                if not success:
                    print("❌ Server 모델 실행 실패")
                    continue

            elif step == "classification":
                if not os.path.exists(classification_data_path):
                    os.makedirs(classification_data_path, exist_ok=True)
                result = run_classification_model(config)
                if not result:
                    print("❌ Classification 모델 실행 실패")
                    continue
                
                # 이미 읽은 결과를 전달
                remove_non_pothole_bboxes(config, server_data_path, classification_data_path, classification_results=result, debug_mode=True)

            # 각 단계마다 evaluation 실행 여부 확인
            if config.get("evaluation_per_step", False):
                curr_annot_path = {
                    "edge": edge_data_path,
                    "server": server_data_path,
                    "classification": classification_data_path
                }[step]
                
                if os.path.exists(curr_annot_path):
                    run_evaluation(gt_data_path, curr_annot_path, config, step)
                else:
                    print(f"⚠️ {step} 단계의 평가를 건너뜁니다. 결과 파일이 없습니다.")

        except Exception as e:
            print(f"❌ {step} 단계 실행 중 오류 발생: {str(e)}")
            continue
    
    if not config.get("evaluation_per_step", False):
        final_annot_path = classification_data_path if "classification" in selected_steps else (
            server_data_path if "server" in selected_steps else edge_data_path)
        if os.path.exists(final_annot_path):
            run_evaluation(gt_data_path, final_annot_path, config, "final")
        else:
            print("⚠️ 최종 평가를 건너뜁니다. 결과 파일이 없습니다.")

    print("\n✅ 모든 선택된 단계를 실행 완료!")

if __name__ == "__main__":
    main()
