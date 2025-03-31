import os
import json
import sys
import random
import glob
import paramiko

from sshmanager import SSHManager

AI_APPLICATION_VERION = "INFERENCE DETECTION MODEL"
line_clean_count = 140
batch_size = 2000

def yolo_inference(info, model_type, filelist=None):
    host = info["detection_host"]
    timeout = info["timeout"]
    username = info["detection_username"]
    password = info["detection_password"]
    model_framework = info[f'{model_type}_model_framework']
    model_name = info[f"{model_type}_model_name"]
    iou_thresh = info["iou_thresh"]
    thresh = info["thresh"]
    input_path = info["input_path"]
    output_dir = info["output_dir"]
    gpu_num = info["gpu_num"]

    print("\n\n==================================================================================================================")
    print(f"                                         * {AI_APPLICATION_VERION} - {(model_type).upper()} *")
    print("- Configuration Information")
    print(f"- Version : {AI_APPLICATION_VERION}")
    print(f"- Model Framework: {model_framework}")
    print(f"- threshold: {thresh}")
    print(f"- IoU-threshold: {iou_thresh}")
    print(f"- Model Name : {model_name}")
    print(f"- GPU Number : {gpu_num}")
    print(f"- Input Directory : {input_path}")
    print(f"- Output Directory : {output_dir}/{model_type}")
    print(f"- Host IP : {host}")
    print("==================================================================================================================\n")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=username, password=password, timeout=timeout)
    stdin, stdout, stderr = ssh.exec_command('cd /data/model/ ; ls')
    server_model_list = (''.join(stdout.readlines()).split('\n'))
    ssh.close()

    if '' in server_model_list:
        server_model_list.remove('')

    if model_name not in server_model_list:
        print("모델명을 확인해주세요.")
        print(f'지정한 모델 : "{model_name}"')
        print('서버 저장된 모델 :', server_model_list)
        sys.exit()

    ssh_manager = SSHManager()
    print( f"---------->   Create ssh client : {host}" )
    ssh_manager.create_ssh_client(host, username, password, timeout) # 세션생성

    if not filelist:
        print(f'input_path = {input_path}')
        filelist = glob.glob(input_path+'\\**\\*.jpg', recursive=True) + glob.glob(input_path+'\\**\\*.png', recursive=True)
    else:
        print(f'input_path = {os.path.dirname(filelist[0])}')

    remote_input_dir_num = int(random.random() * 10000000000)
    remote_input_dir = "/tmp/"+model_framework+"/input" + "_" + str(remote_input_dir_num) + '/'
    remote_input_dir_images = "/tmp/" + model_framework + "/input" + "_" + str(remote_input_dir_num) + '/images/'
    remote_input_dir_annotations = "/tmp/" + model_framework + "/input" + "_" + str(remote_input_dir_num) + '/annotations/'
    remote_input_list_file = "input_" + str(remote_input_dir_num) + ".txt"

    print(f"remote dir = {remote_input_dir_images}, {remote_input_list_file}")

    ssh_manager.send_command(f'mkdir -p {remote_input_dir_images}')
    ssh_manager.send_command(f'mkdir -p {remote_input_dir_images}' + '/results')
    ssh_manager.send_command(f'mkdir -p {remote_input_dir_annotations}')

    sending_cnt = 0
    img_cnt = 0
    partial_send_file_list = []
    print(f'total image count : {len(filelist)}')
    for img in filelist:
        img_cnt = img_cnt + 1
        sending_cnt = sending_cnt + 1
        print ('\r', " " * line_clean_count, end='\r')
        send_count_str = str(img_cnt).zfill(10)
        print (f"\r{send_count_str} : {img} sending....", end='\r')
        image_base_name = os.path.basename(img)
        ssh_manager.send_file(img, remote_input_dir_images+image_base_name) # 파일전송
        partial_send_file_list.append(image_base_name)


        if sending_cnt % batch_size == 0:
            sending_cnt = 0
            print ('\r', " " * line_clean_count, end='\r')
            print (f"---------->   Input Image Count : {img_cnt}")
            ssh_manager.send_command(f'cd /home/daree/dev/darknet ; python make_input_file.py --base_path {remote_input_dir_images}')
            print("---------->   Success uploading all image files complete")
            print("---------->   Model inference start")

            # 1. darknet - 중간 배치 구간
            if model_framework == 'darknet':
                ssh_manager.send_command_long_time(
                    f'cd /home/daree/dev/darknet ; /home/daree/dev/darknet/AI_pipeline.sh {model_name} {remote_input_dir_images} {iou_thresh} 0.005 -save_labels {gpu_num}')

            elif model_framework == 'darknet255':
                ssh_manager.send_command_long_time(
                    f'cd /home/daree/dev/darknet ; /home/daree/dev/darknet/AI_pipeline_255.sh {model_name} {remote_input_dir_images} {iou_thresh} 0.005 -save_labels {gpu_num}')

            print("---------->   Model inference end")
            output_images_path = os.path.join(output_dir, model_type, 'images')
            output_annotations_path = os.path.join(output_dir, model_type, 'annotations')
            if not os.path.exists(output_images_path):
                os.makedirs(output_images_path)
            if not os.path.exists(output_annotations_path):
                os.makedirs(output_annotations_path)

            print("---------->   Model Result file Receiving")
            img_recv_cnt = 0
            for filename in partial_send_file_list:
                img_recv_cnt += 1
                # ssh_manager.get_file(remote_input_dir_images + 'results/' + filename[:-3] + 'jpg',
                #                     os.path.join(output_images_path, filename[:-3] + 'jpg'))  # download images file
                ssh_manager.get_file(remote_input_dir_annotations + filename[:-3] + 'txt',
                                    os.path.join(output_annotations_path, filename[:-3] + 'txt'))  # download annotations file

                print ('\r', " " * line_clean_count, end='\r')
                print (f"{filename} Receiving....\r", end=' ')

            print ('\r', " " * line_clean_count, end='\r')
            print (f"---------->   Result Image Count : {img_recv_cnt}")

            print("---------->   Cleaning Server data")
            partial_send_file_list.clear()
            print(f"---------->   partial_send_file_list count = {len(partial_send_file_list)}")
            ssh_manager.send_command('rm -rf '+ remote_input_dir)
            ssh_manager.send_command(f'mkdir -p {remote_input_dir_images}')
            ssh_manager.send_command(f'mkdir -p {remote_input_dir_images}' + '/results')
            ssh_manager.send_command(f'mkdir -p {remote_input_dir_annotations}')

    if sending_cnt > 0:
        sending_cnt = 0
        print ('\r', " " * line_clean_count, end='\r')
        print (f"---------->   Input Image Count : {img_cnt}")
        ssh_manager.send_command(f'cd /home/daree/dev/darknet ; python make_input_file.py --base_path {remote_input_dir_images}')
        print("---------->   Success uploading all image files complete")
        print("---------->   Model inference start")

        # 2. darknet - 마지막 남은 배치 처리 구간
        if model_framework == 'darknet':
            ssh_manager.send_command_long_time(
                f'cd /home/daree/dev/darknet ; /home/daree/dev/darknet/AI_pipeline.sh {model_name} {remote_input_dir_images} {iou_thresh} 0.005 -save_labels {gpu_num}')

        elif model_framework == 'darknet255':
            ssh_manager.send_command_long_time(
                f'cd /home/daree/dev/darknet ; /home/daree/dev/darknet/AI_pipeline_255.sh {model_name} {remote_input_dir_images} {iou_thresh} 0.005 -save_labels {gpu_num}')

        print("---------->   Model inference end")

        output_images_path = os.path.join(output_dir, model_type, 'images')
        output_annotations_path = os.path.join(output_dir, model_type, 'annotations')

        if not os.path.exists(output_images_path):
            os.makedirs(output_images_path)
        if not os.path.exists(output_annotations_path):
            os.makedirs(output_annotations_path)

        print("---------->   Model Result file Receiving")
        img_recv_cnt = 0
        for filename in partial_send_file_list:
            img_recv_cnt += 1
            # ssh_manager.get_file(remote_input_dir_images + 'results/' + filename[:-3] + 'jpg',
            #                     os.path.join(output_images_path,filename[:-3] + 'jpg'))  # download images file
            ssh_manager.get_file(remote_input_dir_annotations + filename[:-3] + 'txt',
                                os.path.join(output_annotations_path, filename[:-3] + 'txt'))  # download annotations file

            print ('\r', " " * line_clean_count, end='\r')
            print (f"{filename} Receiving....\r", end=' ')
        print ('\r', " " * line_clean_count, end='\r')
        print (f"---------->   Result Image Count : {img_recv_cnt}")

        print("---------->   Cleaning Server data")
        partial_send_file_list.clear()
        print(f"---------->   partial_send_file_list count = {len(partial_send_file_list)}")

        ssh_manager.send_command('rm -rf '+ remote_input_dir)

    print("---------->   Session Closing")
    ssh_manager.close_ssh_client() # 세션종료

    print(f"---------->   {model_name} Model Inference Complete")
    return True


if __name__=="__main__":
    yolo_inference()