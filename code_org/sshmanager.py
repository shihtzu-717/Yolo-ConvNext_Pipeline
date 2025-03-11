import sys
import socket
import select
import paramiko
from paramiko.ssh_exception import BadHostKeyException, AuthenticationException, SSHException
from scp import SCPClient, SCPException


line_clean_count = 140
class SSHManager:
    def __init__(self):
        self.ssh_client = None
    def create_ssh_client(self, hostname, username, password, timeout, port=22):
        """Create SSH client session to remote server"""
        if self.ssh_client is None:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                self.ssh_client.connect(hostname, username=username, password=password, timeout=timeout, port=port)
            except (BadHostKeyException, AuthenticationException, SSHException, socket.error) as e:
                print (f"ssh connection error exeption!!!!!!!!!!!!")
                print(e)
                sys.exit()
        else: print("SSH client session exist.")

    def close_ssh_client(self):
        """Close SSH client session"""
        self.ssh_client.close()
    def send_file(self, local_path, remote_path):
        """Send a single file to remote path"""
        try:
            with SCPClient(self.ssh_client.get_transport()) as scp:
                scp.put(local_path, remote_path, preserve_times=True)
        except SCPException:
            raise SCPException.message
    def get_file(self, remote_path, local_path):
        """Get a single file from remote path"""
        try:
            with SCPClient(self.ssh_client.get_transport()) as scp:
                scp.get(remote_path, local_path)
        except SCPException as e:
            print(e)

    def send_command(self, command):
        """Send a single command"""
        try:
            stdin, stdout, stderr = self.ssh_client.exec_command(command)
            return stdout.readlines()
        except SCPException as e:
            print(e)


    def send_command_long_time(self, command):
        """Send a single command"""
        channel = self.ssh_client.get_transport().open_session()
        channel.exec_command(command)
        model_cnt = 0
        while True:
            if channel.exit_status_ready():
                break
            rl, wl, xl = select.select([channel], [], [], 0.0)
            if len(rl) > 0:
                print('\r', " " * line_clean_count, end='\r')
                print(f"---------->   Modeling {model_cnt} \r", end=' ')
                model_cnt = model_cnt + 1
                # print (f"{str(channel.recv(1024)).rstrip()}\r", end=' ')
        print('\r', " " * line_clean_count, end='\r')