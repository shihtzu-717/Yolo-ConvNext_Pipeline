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
    if len(sys.argv) != 2:
        print("\n오류 : 입력값이 부족합니다.\n")
        print("사용법 : python ./app.py [json 설정파일명]")
        sys.exit()

    if os.path.isfile(sys.argv[1]):
        file_name = sys.argv[1]
    else:
        print (f"\n오류 : {sys.argv[1]} 파일이 존재하지 않습니다.")
        sys.exit()


    current_path = os.path.abspath(os.getcwd())
    json_file = os.path.join(current_path,file_name)
    with open(json_file,"r", encoding='utf-8-sig') as f:
        info = json.load(f)
    host = info["host"]
    port = info["port"]
    timeout = info["timeout"]
    username = info["username"]
    password = info["password"]
    model_framework = info['classification_model_type']
    model_name = info["classification_model_name"]
    iou_thresh = info["iou_thresh"]
    thresh = info["thresh"]
    input_path = info["input_path"]
    output_dir = info["output_dir"]

    print("\n\n==================================================================================================================")
    print(f"                                         * {AI_APPLICATION_VERION} *\n")
    print("- Configuration Information\n")
    print(f"- Version : {AI_APPLICATION_VERION}\n")
    print(f"- Model Framework: {model_framework}\n")
    print(f"- threshold: {thresh}\n")
    print(f"- IoU-threshold: {iou_thresh}\n")
    print(f"- Model Name : {model_name}\n")
    print(f"- Input Directory : {input_path}\n")
    print(f"- Output Directory : {output_dir}_{model_name}\n")
    print(f"- Host IP : {host} -p {port}\n")
    print("==================================================================================================================\n")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=username, password=password, timeout=timeout, port=port)
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
    ssh_manager.create_ssh_client(host, username, password,timeout) # 세션생성

    current_path = os.path.abspath(os.getcwd())
    #filelist = os.listdir(os.path.join(current_path,input_path))
    base_path = os.path.join(current_path, input_path)
    print(f'base_path = {base_path}')
    filelist = glob.glob(base_path+'\\**\\*.jpg', recursive=True) + glob.glob(base_path+'\\**\\*.png', recursive=True)


    remote_input_dir_num = int(random.random() * 10000000000)

    # remote_input_dir = "/home/daree/dev/yolov7/input" + "_" + str(remote_input_dir_num) + '/'
    #remote_input_dir = "/home/daree/dev/"+model_framework+"/input" + "_" + str(remote_input_dir_num) + '/'
    if save_labels:
        remote_input_dir = "/tmp/"+model_framework+"/input" + "_" + str(remote_input_dir_num) + '/'
        remote_input_dir_images = "/tmp/" + model_framework + "/input" + "_" + str(remote_input_dir_num) + '/images/'
        remote_input_dir_annotations = "/tmp/" + model_framework + "/input" + "_" + str(remote_input_dir_num) + '/annotations/'
        remote_input_list_file = "input_" + str(remote_input_dir_num) + ".txt"
    else:
        remote_input_dir = remote_input_dir_images = "/tmp/"+model_framework+"/input" + "_" + str(remote_input_dir_num) + '/'
        remote_input_list_file = "input_" + str(remote_input_dir_num) + ".txt"
    print(f"remote dir = {remote_input_dir_images}, {remote_input_list_file}")

    #sys.exit()
    #ssh_manager.send_command('mkdir /home/daree/dev/darknet/input')
    ssh_manager.send_command(f'mkdir -p {remote_input_dir_images}')
    ssh_manager.send_command(f'mkdir -p {remote_input_dir_images}' + '/results')
    if save_labels:
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
        #ssh_manager.send_file(os.path.join(current_path,input_path,img), "/home/daree/dev/darknet/input/"+img) # 파일전송
        #ssh_manager.send_file(os.path.join(current_path,input_path,img), remote_input_dir+img) # 파일전송
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

            if model_framework == 'darknet':
                if save_labels:
                    ssh_manager.send_command_long_time(
                        'cd /home/daree/dev/darknet ; /home/daree/dev/darknet/AI_application.sh ' + model_name + ' ' + remote_input_dir_images + ' ' + iou_thresh + ' ' + thresh + ' -save_labels')
                else:
                    ssh_manager.send_command_long_time(
                        'cd /home/daree/dev/darknet ; /home/daree/dev/darknet/AI_application.sh ' + model_name + ' ' + remote_input_dir_images + ' ' + iou_thresh + ' ' + thresh + ' ')  # model shell file arg 추가

            elif model_framework == 'darknet255':
                if save_labels:
                    ssh_manager.send_command_long_time(
                        'cd /home/daree/dev/darknet ; /home/daree/dev/darknet/AI_application_255.sh ' + model_name + ' ' + remote_input_dir_images + ' ' + iou_thresh + ' ' + thresh + ' -save_labels')
                else:
                    ssh_manager.send_command_long_time(
                        'cd /home/daree/dev/darknet ; /home/daree/dev/darknet/AI_application_255.sh ' + model_name + ' ' + remote_input_dir_images + ' ' + iou_thresh + ' ' + thresh + ' ')  # model shell file arg 추가

            elif model_framework == 'yolov7':
                ssh_manager.send_command_long_time(
                    'cd /home/daree/dev/yolov7 ; sh AI_application.sh '+model_name+' '+ remote_input_dir_images) # model shell file arg 추가

            print("---------->   Model inference end")
            # output_path = f"{output_dir}_{model_framework}_{model_name}"
            # base_path = os.path.dirname(output_dir)
            # sub_path = f"output_{model_name}_{model_framework}/" + os.path.basename(output_dir) + "/images"
            # output_path = base_path + "/" + sub_path
            # #print(f'output_path = {output_path}')

            if save_labels:
                if save_labels_only:
                    output_annotations_path = os.path.join('output', model_name, output_dir, 'annotations')
                else:
                    output_images_path = os.path.join('output', model_name, output_dir, 'images')
                    output_annotations_path = os.path.join('output', model_name, output_dir, 'annotations')
                    if not os.path.exists(output_images_path):
                        os.makedirs(output_images_path)
                if not os.path.exists(output_annotations_path):
                    os.makedirs(output_annotations_path)
            else:
                output_images_path = os.path.join('output', model_name, output_dir)
                if not os. path.exists(output_images_path):
                    os.makedirs(output_images_path)

            print("---------->   Model Result file Receiving")
            img_recv_cnt = 0
            #filenum = len(filelist)
            filenum = len(partial_send_file_list)
            #for i in range(1,filenum+1):
            for filename in partial_send_file_list:
                img_recv_cnt += 1
                #filename = "predictions"+str(i)+".jpg"
                if save_labels:
                    if save_labels_only:
                        ssh_manager.get_file(remote_input_dir_annotations + filename[:-3] + 'txt',
                                             os.path.join(current_path, output_annotations_path, filename[:-3] + 'txt'))  # download annotations file
                    else:
                        ssh_manager.get_file(remote_input_dir_images + 'results/' + filename[:-3] + 'jpg',
                                            os.path.join(current_path, output_images_path, filename[:-3] + 'jpg'))  # download images file
                        ssh_manager.get_file(remote_input_dir_annotations + filename[:-3] + 'txt',
                                            os.path.join(current_path, output_annotations_path, filename[:-3] + 'txt'))  # download annotations file
                else:
                    ssh_manager.get_file(remote_input_dir_images+'results/'+filename[:-3]+'jpg',
                                         os.path.join(current_path, output_images_path, filename[:-3]+'jpg')) # download images file

                print ('\r', " " * line_clean_count, end='\r')
                print (f"{filename} Receiving....\r", end=' ')
            print ('\r', " " * line_clean_count, end='\r')
            print (f"---------->   Result Image Count : {img_recv_cnt}")

            print("---------->   Cleaning Server data")
            partial_send_file_list.clear()
            print(f"---------->   partial_send_file_list count = {len(partial_send_file_list)}")
            if save_labels:
                ssh_manager.send_command('rm -rf '+ remote_input_dir)
                ssh_manager.send_command(f'mkdir -p {remote_input_dir_images}')
                ssh_manager.send_command(f'mkdir -p {remote_input_dir_images}' + '/results')
                ssh_manager.send_command(f'mkdir -p {remote_input_dir_annotations}')
            else:
                ssh_manager.send_command('rm -rf '+ remote_input_dir_images)
                ssh_manager.send_command(f'mkdir -p {remote_input_dir_images}')
                ssh_manager.send_command(f'mkdir -p {remote_input_dir_images}' + '/results')

    if sending_cnt > 0:
        sending_cnt = 0
        print ('\r', " " * line_clean_count, end='\r')
        print (f"---------->   Input Image Count : {img_cnt}")
        ssh_manager.send_command(f'cd /home/daree/dev/darknet ; python make_input_file.py --base_path {remote_input_dir_images}')
        print("---------->   Success uploading all image files complete")
        print("---------->   Model inference start")

        if model_framework == 'darknet':
            if save_labels:
                ssh_manager.send_command_long_time(
                    'cd /home/daree/dev/darknet ; /home/daree/dev/darknet/AI_pipeline.sh ' + model_name + ' ' + remote_input_dir_images + ' ' + iou_thresh + ' ' + thresh + ' -save_labels')
            else:
                ssh_manager.send_command_long_time(
                    'cd /home/daree/dev/darknet ; /home/daree/dev/darknet/AI_pipeline.sh ' + model_name + ' ' + remote_input_dir_images + ' ' + iou_thresh + ' ' + thresh + ' ')  # model shell file arg 추가

        elif model_framework == 'darknet255':
            if save_labels:
                ssh_manager.send_command_long_time(
                    'cd /home/daree/dev/darknet ; /home/daree/dev/darknet/AI_application_255.sh ' + model_name + ' ' + remote_input_dir_images + ' ' + iou_thresh + ' ' + thresh + ' -save_labels')
            else:
                ssh_manager.send_command_long_time(
                    'cd /home/daree/dev/darknet ; /home/daree/dev/darknet/AI_application_255.sh ' + model_name + ' ' + remote_input_dir_images + ' ' + iou_thresh + ' ' + thresh + ' ')  # model shell file arg 추가

        elif model_framework == 'yolov7':
            ssh_manager.send_command_long_time(
                'cd /home/daree/dev/yolov7 ; sh AI_application.sh ' + model_name + ' ' + remote_input_dir_images)  # model shell file arg 추가

        print("---------->   Model inference end")
        # output_path = f"{output_dir}_{model_framework}_{model_name}"
        # base_path = os.path.dirname(output_dir)
        # sub_path = f"output_{model_name}_{model_framework}/" + os.path.basename(output_dir) + "/images"
        # output_path = base_path + "/" + sub_path
        # #print(f'output_path = {output_path}')
        if save_labels:
            if save_labels_only:
                output_annotations_path = os.path.join('output', model_name, output_dir, 'annotations')
            else:
                output_images_path = os.path.join('output', model_name, output_dir, 'images')
                output_annotations_path = os.path.join('output', model_name, output_dir, 'annotations')
                if not os.path.exists(output_images_path):
                    os.makedirs(output_images_path)
            if not os.path.exists(output_annotations_path):
                os.makedirs(output_annotations_path)
        else:
            output_images_path = os.path.join('output', model_name, output_dir)
            if not os.path.exists(output_images_path):
                os.makedirs(output_images_path)

        print("---------->   Model Result file Receiving")
        img_recv_cnt = 0
        #filenum = len(filelist)
        filenum = len(partial_send_file_list)
        #for i in range(1,filenum+1):
        for filename in partial_send_file_list:
            img_recv_cnt += 1
            if save_labels:
                if save_labels_only:
                    ssh_manager.get_file(remote_input_dir_annotations + filename[:-3] + 'txt',
                                         os.path.join(current_path, output_annotations_path, filename[:-3] + 'txt'))  # download annotations file
                else:
                    ssh_manager.get_file(remote_input_dir_images + 'results/' + filename[:-3] + 'jpg',
                                         os.path.join(current_path, output_images_path,filename[:-3] + 'jpg'))  # download images file
                    ssh_manager.get_file(remote_input_dir_annotations + filename[:-3] + 'txt',
                                         os.path.join(current_path, output_annotations_path, filename[:-3] + 'txt'))  # download annotations file
            else:
                ssh_manager.get_file(remote_input_dir_images + 'results/' + filename[:-3] + 'jpg',
                                     os.path.join(current_path, output_images_path, filename[:-3] + 'jpg'))  # download images file

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

    print(f"---------->   {model_name} Model Application Complete")


if __name__=="__main__":
    convnext_inference()