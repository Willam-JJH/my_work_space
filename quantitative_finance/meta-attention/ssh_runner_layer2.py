"""Upload subset data + Layer-2 script to 4090 and run."""
import paramiko, time, os

HOST='ai.bygpu.com'; PORT=52902; USER='user2'; PWD='Lingyi234'
REMOTE='/home/user2/meta_attn'
LOCAL_DATA='D:/code/data/remote_subset'
SCRIPT='D:/code/quantitative_finance/meta-attention/second_attention_test.py'

print("Connecting...")
ssh=paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, PORT, USER, PWD)

# Create dirs
ssh.exec_command(f'mkdir -p {REMOTE}/data/us_market {REMOTE}/data/cn_market'); time.sleep(1)

# Upload data (40MB total)
sftp=ssh.open_sftp()
for root,dirs,files in os.walk(LOCAL_DATA):
    for f in files:
        local=os.path.join(root,f).replace('\\','/')
        rel=os.path.relpath(local, LOCAL_DATA).replace('\\','/')
        remote=f'{REMOTE}/data/{rel}'
        os.makedirs(os.path.dirname(local), exist_ok=True)
        ssh.exec_command(f'mkdir -p {os.path.dirname(remote)}'); time.sleep(0.2)
        size=os.path.getsize(local)/1024/1024
        print(f"Uploading {rel} ({size:.0f}MB)...")
        sftp.put(local, remote)
sftp.close()

# Upload and fix script
remote_script=f'{REMOTE}/second_attention_test.py'
sftp=ssh.open_sftp(); sftp.put(SCRIPT, remote_script); sftp.close()
ssh.exec_command(f"sed -i \"s|D:/code/data/|{REMOTE}/data/|g\" {remote_script}"); time.sleep(1)

# Check CUDA
sin,sout,serr=ssh.exec_command('python3 -c "import torch; print(torch.cuda.is_available())"')
print(f"CUDA: {sout.read().decode().strip()}")

# Run
print("Starting (nohup)...")
ssh.exec_command(f'cd {REMOTE} && nohup python3 second_attention_test.py > output_layer2.log 2>&1 &')
time.sleep(3)
sin,sout,serr=ssh.exec_command('ps aux | grep second_attention | grep -v grep')
print(f"Process:\n{sout.read().decode()}")

# Poll
for i in range(25):
    time.sleep(30)
    sin,sout,serr=ssh.exec_command(f'tail -3 {REMOTE}/output_layer2.log 2>/dev/null')
    last=sout.read().decode().strip()
    if last: print(f"[{30*(i+1)}s] {last[:300]}")
    sin,sout,serr=ssh.exec_command('ps aux | grep second_attention | grep -v grep')
    if not sout.read().decode().strip():
        print("Done!")
        sin,sout,serr=ssh.exec_command(f'cat {REMOTE}/output_layer2.log')
        print(sout.read().decode())
        break

ssh.close()
