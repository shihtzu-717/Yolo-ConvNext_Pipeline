import os
import json
import detection
import classification
import pascalvoc_updated

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
    yolo_results = detection.yolo_inference(config, 'edge')
    print("✅ Edge Model 탐지 완료!")
    return yolo_results

def run_server_model(config):
    """Server Model 탐지 실행"""
    print("🚀 Server Model 탐지 실행 중...")
    yolo_results = detection.yolo_inference(config, 'server')
    print("✅ Server Model 탐지 완료!")
    return yolo_results

def run_classification_model(config):
    """Classification Model 실행"""
    print("🚀 Classification Model 실행 중...")
    convnext_results = classification.convnext_inference(config, 'classification')
    print("✅ Classification Model 완료!")

    # 결과 저장
    results_path = os.path.join("results", "final_results.json")
    os.makedirs("results", exist_ok=True)  # 결과 폴더가 없으면 생성
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(convnext_results, f, indent=4)
    print(f"✅ 최종 결과 저장 완료: {results_path}")
    return convnext_results

def run_evaluation(config, step):
    """성능 평가 실행 (항상 마지막에 실행)"""
    print("🚀 성능 평가 실행 중...")
    gt_data_path = os.path.join(config["input_path"], "annotations")
    det_data_path = os.path.join(config["output_dir"], step, "annotations")
    results_data_path = os.path.join(config["output_dir"], "evaluate")
    results = pascalvoc_updated.process_evaluation(gt_data_path, det_data_path, 10, iouThreshold=0.5, savePath=results_data_path, showPlot=True)
    print("✅ 성능 평가 완료!")


def main():
    # 1️⃣ 설정 로드
    config = load_config(CONFIG_PATH)

    # 실행할 단계 확인
    selected_steps = config.get("model_pipeline", [])

    # 유효한 단계만 필터링 (evaluation 제외)
    selected_steps = [step for step in selected_steps if step in STEPS]

    if not selected_steps:
        print("⚠️ 실행할 단계가 설정되지 않았습니다. 프로그램을 종료합니다.")
        exit()

    # 평가 실행 방식 확인
    evaluation_per_step = config.get("evaluation_per_step", False)

    # 2️⃣ 선택된 단계 실행
    print(f"\n🔹 실행할 단계: {', '.join(selected_steps)}\n")

    for step in selected_steps:
        if step == "edge":
            run_edge_model(config)
        elif step == "server":
            run_server_model(config)
        elif step == "classification":
            run_classification_model(config)

        # 각 단계마다 evaluation 실행 여부 확인
        if evaluation_per_step:
            run_evaluation(config, step)

    # 3️⃣ "evaluation" 단계는 한 번만 실행하는 경우 (evaluation_per_step=False)
    if not evaluation_per_step:
            run_evaluation(config, step)

    print("\n✅ 모든 선택된 단계를 실행 완료!")

if __name__ == "__main__":
    main()
