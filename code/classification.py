import os
import json
import sys
import random
import glob
import paramiko

from sshmanager import SSHManager

AI_APPLICATION_VERION = "INFERENCE CLASSIFICATION MODEL"
line_clean_count = 140
batch_size = 2000

def convnext_inference(info, mode_type):
    host = info["classification_host"]
    timeout = info["timeout"]
    username = info["classification_username"]
    password = info["classification_password"]
    model_type = info['classification_model_type']
    model_name = info["classification_model_name"]
    input_path = info["input_path"]
    output_dir = info["output_dir"]

    print("\n\n==================================================================================================================")
    print(f"                                         * {AI_APPLICATION_VERION} *\n")
    print("- Configuration Information")
    print(f"- Version : {AI_APPLICATION_VERION}")
    print(f"- Model Type: {model_type}")
    print(f"- Model Name : {model_name}")
    print(f"- Input Directory : {input_path}")
    print(f"- Output Directory : {output_dir}_{model_name}")
    print(f"- Host IP : {host}")
    print("==================================================================================================================\n")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=username, password=password, timeout=timeout)
    stdin, stdout, stderr = ssh.exec_command(f'cd /data/classifier/model_pipeline/best_model ; ls')
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
    ssh_manager.create_ssh_client(host, username, password,timeout) # 세션 생성

    img_path = input_path
    annot_path = os.path.join(output_dir, 'server', 'annotations')
    print(f'img_path = {img_path}\n annot_path= {annot_path}')

    img_filelist = sorted(glob.glob(img_path+'\\**\\*.jpg', recursive=True) + glob.glob(img_path+'\\**\\*.png', recursive=True))
    annot_filelist = sorted(glob.glob(annot_path+'\\**\\*.txt', recursive=True))
    img_filelist = sorted(img_filelist, key=lambda x: os.path.basename(x))
    annot_filelist = sorted(annot_filelist, key=lambda x: os.path.basename(x))

    remote_input_dir_num = int(random.random() * 10000000000)

    remote_input_dir = f"/data/classifier/model_pipeline/tmp/{model_type}/input_{remote_input_dir_num}"
    remote_input_list_file = f"input_{remote_input_dir_num}.txt"
    print(f"remote dir = {remote_input_dir}, {remote_input_list_file}")

    # ssh_manager.send_command(f'mkdir -p {remote_input_dir}')
    ssh_manager.send_command(f'mkdir -p {remote_input_dir}' + '/images')
    ssh_manager.send_command(f'mkdir -p {remote_input_dir}' + '/annotations')
    ssh_manager.send_command(f'mkdir -p {remote_input_dir}' + '/results')

    sending_cnt = 0
    img_cnt = 0
    partial_send_file_list = []

    print(f"🖼️ Image File List: {len(img_filelist)}")
    print(f"📑 Annotation File List: {len(annot_filelist)}")

    for img, annot in zip(img_filelist, annot_filelist):
        img_cnt = img_cnt + 1
        sending_cnt = sending_cnt + 1
        print ('\r', " " * line_clean_count, end='\r')
        send_count_str = str(img_cnt).zfill(10)
        print (f"\r{send_count_str} : {img} sending....", end='\r')

        image_base_name = os.path.basename(img)
        annot_base_name = os.path.basename(annot)

        ssh_manager.send_file(img, f"{remote_input_dir}/images/{image_base_name}") # images 파일 전송
        ssh_manager.send_file(annot, f"{remote_input_dir}/annotations/{annot_base_name}") # annotations 파일 전송
        partial_send_file_list.append((image_base_name, annot_base_name))


        if sending_cnt % batch_size == 0:
            sending_cnt = 0

            print ('\r', " " * line_clean_count, end='\r')
            print (f"---------->   Input Image Count : {img_cnt}")
            print("---------->   Success uploading all image files complete")
            print("---------->   Model inference start")

            if model_type == 'pothole':
                ssh_manager.send_command_long_time(
                    f"cd /data/classifier/model_pipeline ; \
                    python inference.py \
                    --model_type {model_type} \
                    --resume best_model/{model_name}/checkpoint-best_weights.pth  \
                    --input_img_path {remote_input_dir}/images \
                    --input_annot_path {remote_input_dir}/annotations \
                    --output_path {remote_input_dir}")

            elif model_type == 'debris':
                ssh_manager.send_command_long_time(
                    f"cd /data/classifier/model_pipeline ; \
                    python inference.py \
                    --model_type {model_type} \
                    --resume best_model/{model_name}/checkpoint-best_weights.pth  \
                    --input_img_path {remote_input_dir}/images \
                    --input_annot_path {remote_input_dir}/annotations \
                    --output_path {remote_input_dir}")


            print("---------->   Model inference end")

            output_images_path = os.path.join(output_dir, 'classification')
            if not os.path.exists(output_images_path):
                os.makedirs(output_images_path)

            print("---------->   Model Result file Receiving")
            ssh_manager.get_file(f"{remote_input_dir}/classification_pred.json", output_images_path)  # download reslut json file

            print("---------->   Cleaning Server data")
            partial_send_file_list.clear()
            print(f"---------->   partial_send_file_list count = {len(partial_send_file_list)}")


            ssh_manager.send_command(f'rm -rf {remote_input_dir}/images')
            ssh_manager.send_command(f'rm -rf {remote_input_dir}/annotations')
            ssh_manager.send_command(f'rm -rf {remote_input_dir}/results')

            ssh_manager.send_command(f'mkdir -p {remote_input_dir}' + '/images')
            ssh_manager.send_command(f'mkdir -p {remote_input_dir}' + '/annotations')
            ssh_manager.send_command(f'mkdir -p {remote_input_dir}' + '/results')


    if sending_cnt > 0:
        sending_cnt = 0
        print ('\r', " " * line_clean_count, end='\r')
        print (f"---------->   Input Image Count : {img_cnt}")
        print("---------->   Success uploading all image files complete")
        print("---------->   Model inference start")

        if model_type == 'pothole':
            ssh_manager.send_command_long_time(
                f"cd /data/classifier/model_pipeline ; \
                python inference.py \
                --model_type {model_type} \
                --resume best_model/{model_name}/checkpoint-best_weights.pth  \
                --input_img_path {remote_input_dir}/images \
                --input_annot_path {remote_input_dir}/annotations \
                --output_path {remote_input_dir}")

        elif model_type == 'debris':
            ssh_manager.send_command_long_time(
                f"cd /data/classifier/model_pipeline ; \
                python inference.py \
                --model_type {model_type} \
                --resume best_model/{model_name}/checkpoint-best_weights.pth  \
                --input_img_path {remote_input_dir}/images \
                --input_annot_path {remote_input_dir}/annotations \
                --output_path {remote_input_dir}")

        print("---------->   Model inference end")

        output_images_path = os.path.join(output_dir, 'classification')
        if not os.path.exists(output_images_path):
            os.makedirs(output_images_path)

        print("---------->   Model Result file Receiving")
        ssh_manager.get_file(f"{remote_input_dir}/classification_pred.json", output_images_path)  # download reslut json file

        print("---------->   Cleaning Server data")
        partial_send_file_list.clear()
        print(f"---------->   partial_send_file_list count = {len(partial_send_file_list)}")

        ssh_manager.send_command('rm -rf ' + remote_input_dir)

    print("---------->   Session Closing")
    ssh_manager.close_ssh_client() # 세션종료

    print(f"---------->   {model_name} Model Application Complete")

if __name__=="__main__":
    convnext_inference()