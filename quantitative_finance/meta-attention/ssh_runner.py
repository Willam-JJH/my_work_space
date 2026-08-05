"""Upload and run public benchmarks on remote GPU via nohup."""
import paramiko, time

HOST='ai.bygpu.com'; PORT=52902; USER='user2'; PWD='Lingyi234'
REMOTE_DIR='/home/user2/meta_attn'
LOCAL_FILE='D:/code/quantitative_finance/meta-attention/public_benchmarks_full.py'

print("Connecting...")
ssh=paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, PORT, USER, PWD)
ssh.exec_command(f'mkdir -p {REMOTE_DIR}'); time.sleep(0.5)

# Upload
print("Uploading script...")
sftp=ssh.open_sftp(); sftp.put(LOCAL_FILE, f'{REMOTE_DIR}/public_benchmarks_full.py'); sftp.close()

# Install deps
print("Installing deps...")
sin,sout,serr=ssh.exec_command('pip3 install pandas numpy scikit-learn scipy torch pyarrow 2>&1 | tail -3')
sout.channel.recv_exit_status()

# Run with nohup
print("Starting benchmark (nohup)...")
cmd=f'cd {REMOTE_DIR} && nohup python3 public_benchmarks_full.py > output.log 2>&1 &'
ssh.exec_command(cmd); time.sleep(1)

# Check it started
sin,sout,serr=ssh.exec_command('ps aux | grep public_benchmarks')
procs=sout.read().decode()
print(f"Running processes:\n{procs}")

# Wait and poll
print("Waiting for completion (polling every 30s)...")
for i in range(20):  # max 10 minutes
    time.sleep(30)
    sin,sout,serr=ssh.exec_command(f'tail -3 {REMOTE_DIR}/output.log 2>/dev/null')
    last=sout.read().decode().strip()
    if last:
        print(f"[{30*(i+1)}s] {last[:200]}")
    sin,sout,serr=ssh.exec_command('ps aux | grep public_benchmarks | grep -v grep')
    if not sout.read().decode().strip():
        print("Done! Fetching results...")
        sin,sout,serr=ssh.exec_command(f'tail -30 {REMOTE_DIR}/output.log')
        print(sout.read().decode())
        break

ssh.close()
print("Complete.")
