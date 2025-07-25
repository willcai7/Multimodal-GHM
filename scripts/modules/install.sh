#!/bin/bash

install_msmtp() {
    echo "Installing msmtp..."
    
    # Check if we're on a Debian/Ubuntu system
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y msmtp msmtp-mta
    # Check if we're on a Red Hat/Fedora system
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y msmtp
    elif command -v yum &> /dev/null; then
        sudo yum install -y msmtp
    # Check if we're on an Arch system
    elif command -v pacman &> /dev/null; then
        sudo pacman -S --noconfirm msmtp
    # Check if we're on a macOS system with Homebrew
    elif command -v brew &> /dev/null; then
        brew install msmtp
    else
        echo "Error: Could not determine package manager. Please install msmtp manually."
        return 1
    fi
    
    # Create a basic msmtp configuration file if it doesn't exist
    if [ ! -f "$HOME/.msmtprc" ]; then
        echo "Creating default msmtp configuration file at $HOME/.msmtprc"
        cat > "$HOME/.msmtprc" << EOF
# Default settings for all accounts
defaults
auth           on
tls            on
tls_trust_file /etc/ssl/certs/ca-certificates.crt
logfile        ~/.msmtp.log

# Gmail account
account        gmail
host           smtp.gmail.com
port           587
from           willcai754@gmail.com
user           willcai754@gmail.com
password       rtmzvjwvditdawah 

# Set a default account
account default : gmail
EOF
        chmod 600 "$HOME/.msmtprc"
        echo "Please edit $HOME/.msmtprc with your email credentials"
    fi
    
    echo "msmtp installation completed"
}

# Run the installation function
install_msmtp
# Function to install mail utilities
install_mail() {
    echo "Installing mail utilities..."
    
    # Check which package manager is available
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y mailutils
    # Check if we're on a Red Hat/Fedora system
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y mailx
    elif command -v yum &> /dev/null; then
        sudo yum install -y mailx
    # Check if we're on an Arch system
    elif command -v pacman &> /dev/null; then
        sudo pacman -S --noconfirm mailutils
    # Check if we're on a macOS system with Homebrew
    elif command -v brew &> /dev/null; then
        brew install mailutils
    else
        echo "Error: Could not determine package manager. Please install mail utilities manually."
        return 1
    fi
    
    # Test if mail command is available
    if command -v mail &> /dev/null; then
        echo "Mail utilities installation completed successfully"
    else
        echo "Warning: Mail command not found after installation. You may need to install it manually."
        return 1
    fi
    
    # Configure mail to use msmtp if needed
    if [ -f "$HOME/.msmtprc" ]; then
        echo "Configuring mail to use msmtp..."
        if [ ! -f "$HOME/.mailrc" ]; then
            echo "set sendmail=\"/usr/bin/msmtp -t\"" > "$HOME/.mailrc"
            chmod 600 "$HOME/.mailrc"
        fi
    fi
    
    return 0
}

