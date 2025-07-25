#!/bin/bash
send_email() {
    local to="willcai754@gmail.com"
    local subject="$1"
    local body="$2"
    local from="willcai754@gmail.com"  # Set your email address here
    
    {
        echo "From: $from"
        echo "To: $to"
        echo "Subject: $subject"
        echo ""
        echo "$body"
    } | msmtp --file=/home/jovyan/local/msmtp.conf -t "$to"
}

send_email "Experiment completed" "The experiment has completed successfully."
