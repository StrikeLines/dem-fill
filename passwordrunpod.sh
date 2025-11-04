#!/bin/bash

# Runpod SSH Setup Script with Password Authentication
# Repository: https://github.com/justinwlin/Runpod-SSH-Password
# Usage: wget https://raw.githubusercontent.com/justinwlin/Runpod-SSH-Password/main/passwordrunpod.sh && chmod +x passwordrunpod.sh && ./passwordrunpod.sh

# Detect current user (or default to root if run with sudo)
CURRENT_USER="${SUDO_USER:-$(whoami)}"

# Function to print in color
print_color() {
    COLOR=$1
    TEXT=$2
    case $COLOR in
        "green") echo -e "\e[32m$TEXT\e[0m" ;;
        "red") echo -e "\e[31m$TEXT\e[0m" ;;
        "yellow") echo -e "\e[33m$TEXT\e[0m" ;;
        "blue") echo -e "\e[34m$TEXT\e[0m" ;;
        *) echo "$TEXT" ;;
    esac
}

# Function to check if user has password already set
check_password_exists() {
    local username=$1
    # Check if password is locked or empty
    if sudo passwd -S "$username" 2>/dev/null | grep -qE "L|NP"; then
        return 1  # No password set
    else
        return 0  # Password exists
    fi
}

# Function to check if SSH keys are configured
check_ssh_keys() {
    local username=$1
    local home_dir

    # Get home directory for user
    if [ "$username" = "root" ]; then
        home_dir="/root"
    else
        home_dir=$(eval echo "~$username")
    fi

    # Check for SSH keys
    if [ -d "$home_dir/.ssh" ] && [ -f "$home_dir/.ssh/authorized_keys" ] && [ -s "$home_dir/.ssh/authorized_keys" ]; then
        return 0  # SSH keys exist
    else
        return 1  # No SSH keys
    fi
}

# Function to prompt for password
get_password() {
    local username=$1
    while true; do
        print_color "blue" "Enter a password for $username user:"
        read -s user_password
        echo

        print_color "blue" "Confirm password:"
        read -s confirm_password
        echo

        if [ "$user_password" = "$confirm_password" ]; then
            print_color "green" "Password confirmed successfully."
            break
        else
            print_color "red" "Passwords do not match. Please try again."
        fi
    done
}

# Check for OS Type
print_color "blue" "Detecting Linux Distribution..."
os_info=$(cat /etc/*release)
print_color "yellow" "OS Detected: $os_info"

# Check for SSH Server and install if necessary
if ! command -v sshd >/dev/null; then
    print_color "yellow" "SSH server not found. Installing..."
    if [[ $os_info == *"debian"* || $os_info == *"ubuntu"* ]]; then
        apt-get update && apt-get install -y openssh-server
    elif [[ $os_info == *"redhat"* || $os_info == *"centos"* ]]; then
        yum install -y openssh-server
    else
        print_color "red" "Unsupported Linux distribution for automatic SSH installation."
        exit 1
    fi
    print_color "green" "SSH Server Installed Successfully."
else
    print_color "green" "SSH Server is already installed."
fi

# Detect which user to configure
print_color "blue" "Detected user: $CURRENT_USER"

# Check for SSH keys
if check_ssh_keys "$CURRENT_USER"; then
    print_color "green" "✓ SSH keys detected for $CURRENT_USER"
    print_color "blue" "SSH key authentication is already configured and will continue to work."
    print_color "blue" "This script will enable password authentication alongside your SSH keys."
    echo ""
fi

# Check if password already exists
if check_password_exists "$CURRENT_USER"; then
    print_color "yellow" "Password already exists for user $CURRENT_USER"
    print_color "yellow" "Do you want to:"
    print_color "yellow" "  1) Keep existing password (just enable SSH password auth)"
    print_color "yellow" "  2) Set a new password (will replace existing)"
    read -p "Enter choice (1/2): " password_choice

    if [ "$password_choice" = "1" ]; then
        print_color "green" "Keeping existing password. Only configuring SSH access."
        SKIP_PASSWORD_CHANGE=true
    else
        print_color "blue" "Will set new password for $CURRENT_USER"
        SKIP_PASSWORD_CHANGE=false
    fi
else
    print_color "blue" "No password set for $CURRENT_USER. You'll need to create one."
    SKIP_PASSWORD_CHANGE=false
fi

# Configure SSH to allow password authentication
print_color "blue" "Configuring SSH to allow password authentication..."

# Enable password authentication
sed -i 's/#\?PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config

# If current user is root, ensure PermitRootLogin is enabled
if [ "$CURRENT_USER" = "root" ]; then
    sed -i 's/#\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
fi

# Restart SSH service
service ssh restart
print_color "green" "SSH Configuration Updated."

# Set password if needed
if [ "$SKIP_PASSWORD_CHANGE" = false ]; then
    get_password "$CURRENT_USER"

    print_color "blue" "Setting password for $CURRENT_USER..."
    echo "$CURRENT_USER:$user_password" | chpasswd
    echo $user_password > /workspace/${CURRENT_USER}_password.txt
    print_color "green" "Password set and saved in /workspace/${CURRENT_USER}_password.txt"
else
    user_password="<existing password>"
fi

# Check if environment variables are set
print_color "blue" "Checking environment variables..."
if [ -z "$RUNPOD_PUBLIC_IP" ] || [ -z "$RUNPOD_TCP_PORT_22" ]; then
    print_color "red" "Environment variables RUNPOD_PUBLIC_IP or RUNPOD_TCP_PORT_22 are missing."
    exit 1
fi
print_color "green" "Environment variables are set."

# Create connection script for Windows (.bat)
print_color "blue" "Creating connection script for Windows..."
echo "@echo off" > /workspace/connect_windows.bat
echo "echo ========================================" >> /workspace/connect_windows.bat
echo "echo SSH CONNECTION" >> /workspace/connect_windows.bat
echo "echo ========================================" >> /workspace/connect_windows.bat
echo "echo User: $CURRENT_USER" >> /workspace/connect_windows.bat
echo "echo Password: $user_password" >> /workspace/connect_windows.bat
echo "echo." >> /workspace/connect_windows.bat
echo "echo Connect via SSH:" >> /workspace/connect_windows.bat
echo "echo ssh $CURRENT_USER@$RUNPOD_PUBLIC_IP -p $RUNPOD_TCP_PORT_22" >> /workspace/connect_windows.bat
echo "echo." >> /workspace/connect_windows.bat
echo "echo ========================================" >> /workspace/connect_windows.bat
echo "echo FILE TRANSFER OPTIONS" >> /workspace/connect_windows.bat
echo "echo ========================================" >> /workspace/connect_windows.bat
echo "echo." >> /workspace/connect_windows.bat
echo "echo OPTION 1: RSYNC (Recommended)" >> /workspace/connect_windows.bat
echo "echo WARNING: Requires rsync on BOTH machines!" >> /workspace/connect_windows.bat
echo "echo Install: apt-get update && apt-get install -y rsync" >> /workspace/connect_windows.bat
echo "echo." >> /workspace/connect_windows.bat
echo "echo Copy file TO pod:" >> /workspace/connect_windows.bat
echo "echo rsync -rltvzP --no-perms --no-owner --no-group -e \"ssh -p $RUNPOD_TCP_PORT_22\" yourfile.txt $CURRENT_USER@$RUNPOD_PUBLIC_IP:/workspace/" >> /workspace/connect_windows.bat
echo "echo." >> /workspace/connect_windows.bat
echo "echo Copy file FROM pod:" >> /workspace/connect_windows.bat
echo "echo rsync -rltvzP --no-perms --no-owner --no-group -e \"ssh -p $RUNPOD_TCP_PORT_22\" $CURRENT_USER@$RUNPOD_PUBLIC_IP:/workspace/yourfile.txt ." >> /workspace/connect_windows.bat
echo "echo." >> /workspace/connect_windows.bat
echo "echo Copy folder TO pod:" >> /workspace/connect_windows.bat
echo "echo rsync -rltvzP --no-perms --no-owner --no-group -e \"ssh -p $RUNPOD_TCP_PORT_22\" yourfolder/ $CURRENT_USER@$RUNPOD_PUBLIC_IP:/workspace/" >> /workspace/connect_windows.bat
echo "echo." >> /workspace/connect_windows.bat
echo "echo OPTION 2: SCP (Standard - Works everywhere)" >> /workspace/connect_windows.bat
echo "echo Copy file TO pod:" >> /workspace/connect_windows.bat
echo "echo scp -P $RUNPOD_TCP_PORT_22 yourfile.txt $CURRENT_USER@$RUNPOD_PUBLIC_IP:/workspace/" >> /workspace/connect_windows.bat
echo "echo." >> /workspace/connect_windows.bat
echo "echo Copy file FROM pod:" >> /workspace/connect_windows.bat
echo "echo scp -P $RUNPOD_TCP_PORT_22 $CURRENT_USER@$RUNPOD_PUBLIC_IP:/workspace/yourfile.txt ." >> /workspace/connect_windows.bat
echo "echo." >> /workspace/connect_windows.bat
echo "echo Copy folder TO pod (small folders):" >> /workspace/connect_windows.bat
echo "echo scp -P $RUNPOD_TCP_PORT_22 -r yourfolder $CURRENT_USER@$RUNPOD_PUBLIC_IP:/workspace/" >> /workspace/connect_windows.bat
echo "echo." >> /workspace/connect_windows.bat
echo "echo Copy folder with many small files (recommended - zip first):" >> /workspace/connect_windows.bat
echo "echo zip -r folder.zip yourfolder && scp -P $RUNPOD_TCP_PORT_22 folder.zip $CURRENT_USER@$RUNPOD_PUBLIC_IP:/workspace/ && rm folder.zip" >> /workspace/connect_windows.bat
echo "echo Then extract on destination:" >> /workspace/connect_windows.bat
echo "echo ssh $CURRENT_USER@$RUNPOD_PUBLIC_IP -p $RUNPOD_TCP_PORT_22 \"cd /workspace && unzip folder.zip && rm folder.zip\"" >> /workspace/connect_windows.bat
echo "echo ========================================" >> /workspace/connect_windows.bat
print_color "green" "Windows connection script created in /workspace."

# Create connection script for Linux/Mac (.sh)
print_color "blue" "Creating connection script for Linux/Mac..."
echo "#!/bin/bash" > /workspace/connect_linux.sh
echo "echo '========================================'" >> /workspace/connect_linux.sh
echo "echo 'SSH CONNECTION'" >> /workspace/connect_linux.sh
echo "echo '========================================'" >> /workspace/connect_linux.sh
echo "echo 'User: $CURRENT_USER'" >> /workspace/connect_linux.sh
echo "echo 'Password: $user_password'" >> /workspace/connect_linux.sh
echo "echo ''" >> /workspace/connect_linux.sh
echo "echo 'Connect via SSH:'" >> /workspace/connect_linux.sh
echo "echo 'ssh $CURRENT_USER@$RUNPOD_PUBLIC_IP -p $RUNPOD_TCP_PORT_22'" >> /workspace/connect_linux.sh
echo "echo ''" >> /workspace/connect_linux.sh
echo "echo '========================================'" >> /workspace/connect_linux.sh
echo "echo 'FILE TRANSFER OPTIONS'" >> /workspace/connect_linux.sh
echo "echo '========================================'" >> /workspace/connect_linux.sh
echo "echo ''" >> /workspace/connect_linux.sh
echo "echo 'OPTION 1: RSYNC (Recommended)'" >> /workspace/connect_linux.sh
echo "echo 'WARNING: Requires rsync on BOTH machines!'" >> /workspace/connect_linux.sh
echo "echo 'Install: apt-get update && apt-get install -y rsync'" >> /workspace/connect_linux.sh
echo "echo ''" >> /workspace/connect_linux.sh
echo "echo 'Copy file TO pod:'" >> /workspace/connect_linux.sh
echo "echo 'rsync -rltvzP --no-perms --no-owner --no-group -e \"ssh -p $RUNPOD_TCP_PORT_22\" yourfile.txt $CURRENT_USER@$RUNPOD_PUBLIC_IP:/workspace/'" >> /workspace/connect_linux.sh
echo "echo ''" >> /workspace/connect_linux.sh
echo "echo 'Copy file FROM pod:'" >> /workspace/connect_linux.sh
echo "echo 'rsync -rltvzP --no-perms --no-owner --no-group -e \"ssh -p $RUNPOD_TCP_PORT_22\" $CURRENT_USER@$RUNPOD_PUBLIC_IP:/workspace/yourfile.txt .'" >> /workspace/connect_linux.sh
echo "echo ''" >> /workspace/connect_linux.sh
echo "echo 'Copy folder TO pod:'" >> /workspace/connect_linux.sh
echo "echo 'rsync -rltvzP --no-perms --no-owner --no-group -e \"ssh -p $RUNPOD_TCP_PORT_22\" yourfolder/ $CURRENT_USER@$RUNPOD_PUBLIC_IP:/workspace/'" >> /workspace/connect_linux.sh
echo "echo ''" >> /workspace/connect_linux.sh
echo "echo 'OPTION 2: SCP (Standard - Works everywhere)'" >> /workspace/connect_linux.sh
echo "echo 'Copy file TO pod:'" >> /workspace/connect_linux.sh
echo "echo 'scp -P $RUNPOD_TCP_PORT_22 yourfile.txt $CURRENT_USER@$RUNPOD_PUBLIC_IP:/workspace/'" >> /workspace/connect_linux.sh
echo "echo ''" >> /workspace/connect_linux.sh
echo "echo 'Copy file FROM pod:'" >> /workspace/connect_linux.sh
echo "echo 'scp -P $RUNPOD_TCP_PORT_22 $CURRENT_USER@$RUNPOD_PUBLIC_IP:/workspace/yourfile.txt .'" >> /workspace/connect_linux.sh
echo "echo ''" >> /workspace/connect_linux.sh
echo "echo 'Copy folder TO pod (small folders):'" >> /workspace/connect_linux.sh
echo "echo 'scp -P $RUNPOD_TCP_PORT_22 -r yourfolder $CURRENT_USER@$RUNPOD_PUBLIC_IP:/workspace/'" >> /workspace/connect_linux.sh
echo "echo ''" >> /workspace/connect_linux.sh
echo "echo 'Copy folder with many small files (recommended - zip first):'" >> /workspace/connect_linux.sh
echo "echo 'zip -r folder.zip yourfolder && scp -P $RUNPOD_TCP_PORT_22 folder.zip $CURRENT_USER@$RUNPOD_PUBLIC_IP:/workspace/ && rm folder.zip'" >> /workspace/connect_linux.sh
echo "echo 'Then extract on destination:'" >> /workspace/connect_linux.sh
echo "echo 'ssh $CURRENT_USER@$RUNPOD_PUBLIC_IP -p $RUNPOD_TCP_PORT_22 \"cd /workspace && unzip folder.zip && rm folder.zip\"'" >> /workspace/connect_linux.sh
echo "echo '========================================'" >> /workspace/connect_linux.sh
chmod +x /workspace/connect_linux.sh
print_color "green" "Linux/Mac connection script created in /workspace."

print_color "green" "Setup Completed Successfully!"
echo ""

# Check and display SSH key status
if check_ssh_keys "$CURRENT_USER"; then
    print_color "green" "========================================"
    print_color "green" "SSH AUTHENTICATION METHODS"
    print_color "green" "========================================"
    print_color "green" "✓ SSH Key authentication: ENABLED"
    print_color "green" "✓ Password authentication: ENABLED"
    echo ""
    print_color "blue" "You can connect using either method:"
    print_color "blue" "  - SSH key (if you have the private key)"
    print_color "blue" "  - Password (shown below)"
    echo ""
fi

print_color "yellow" "========================================"
print_color "yellow" "SSH CONNECTION"
print_color "yellow" "========================================"
print_color "yellow" "Connect via SSH:"
echo "ssh $CURRENT_USER@$RUNPOD_PUBLIC_IP -p $RUNPOD_TCP_PORT_22"
echo ""
print_color "yellow" "User: $CURRENT_USER"
print_color "yellow" "Password: $user_password"
echo ""
print_color "blue" "========================================"
print_color "blue" "FILE TRANSFER OPTIONS"
print_color "blue" "========================================"
echo ""
print_color "green" "Option 1: RSYNC (Recommended - Fastest with compression & progress)"
print_color "yellow" "⚠ IMPORTANT: Requires rsync on BOTH source and destination!"
print_color "yellow" "  Install on source: apt-get update && apt-get install -y rsync"
print_color "yellow" "  Install on destination: apt-get update && apt-get install -y rsync"
echo ""
print_color "blue" "Copy file TO pod:"
echo "rsync -rltvzP --no-perms --no-owner --no-group -e 'ssh -p $RUNPOD_TCP_PORT_22' yourfile.txt $CURRENT_USER@$RUNPOD_PUBLIC_IP:/workspace/"
echo ""
print_color "blue" "Copy file FROM pod:"
echo "rsync -rltvzP --no-perms --no-owner --no-group -e 'ssh -p $RUNPOD_TCP_PORT_22' $CURRENT_USER@$RUNPOD_PUBLIC_IP:/workspace/yourfile.txt ."
echo ""
print_color "blue" "Copy entire folder TO pod:"
echo "rsync -rltvzP --no-perms --no-owner --no-group -e 'ssh -p $RUNPOD_TCP_PORT_22' yourfolder/ $CURRENT_USER@$RUNPOD_PUBLIC_IP:/workspace/"
echo ""
echo ""
print_color "green" "Option 2: SCP (Standard - Works everywhere)"
print_color "blue" "Copy file TO pod:"
echo "scp -P $RUNPOD_TCP_PORT_22 yourfile.txt $CURRENT_USER@$RUNPOD_PUBLIC_IP:/workspace/"
echo ""
print_color "blue" "Copy file FROM pod:"
echo "scp -P $RUNPOD_TCP_PORT_22 $CURRENT_USER@$RUNPOD_PUBLIC_IP:/workspace/yourfile.txt ."
echo ""
print_color "blue" "Copy folder TO pod (small folders):"
echo "scp -P $RUNPOD_TCP_PORT_22 -r yourfolder $CURRENT_USER@$RUNPOD_PUBLIC_IP:/workspace/"
echo ""
print_color "blue" "Copy folder with many small files (recommended - zip first):"
echo "zip -r folder.zip yourfolder && scp -P $RUNPOD_TCP_PORT_22 folder.zip $CURRENT_USER@$RUNPOD_PUBLIC_IP:/workspace/ && rm folder.zip"
print_color "blue" "Then extract on destination:"
echo "ssh $CURRENT_USER@$RUNPOD_PUBLIC_IP -p $RUNPOD_TCP_PORT_22 'cd /workspace && unzip folder.zip && rm folder.zip'"
echo ""
echo ""
print_color "green" "Option 3: Migration Tool (Interactive - Auto-handles everything)"
echo "wget https://raw.githubusercontent.com/justinwlin/Runpod-SSH-Password/refs/heads/main/SCPMigration -O scp_migration.py && python3 scp_migration.py"
echo ""
print_color "green" "Connection scripts saved in /workspace/connect_windows.bat and /workspace/connect_linux.sh"
