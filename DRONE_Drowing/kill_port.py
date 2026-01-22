#!/usr/bin/env python3
import os
import signal
import subprocess

def kill_process_on_port(port=5000):
    try:
        # Find PID using port 5000
        result = subprocess.run(
            ['lsof', '-ti', f'tcp:{port}'],
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    print(f"Killing process {pid} on port {port}")
                    os.kill(int(pid), signal.SIGTERM)
        else:
            print(f"No process found on port {port}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    kill_process_on_port(5000)