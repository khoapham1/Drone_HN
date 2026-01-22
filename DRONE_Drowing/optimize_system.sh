#!/bin/bash
# Increase USB buffer size
sudo sh -c 'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb'

# Increase network buffer
sudo sysctl -w net.core.rmem_max=26214400
sudo sysctl -w net.core.wmem_max=26214400

# Increase process limits
ulimit -n 65536
ulimit -u 65536

# Disable power saving for USB
for usb in /sys/bus/usb/devices/*/power/control; do
    echo on > $usb 2>/dev/null
done

echo "System optimized for drone control"