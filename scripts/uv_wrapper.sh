#!/bin/bash

# Check if a script file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <script_file> [additional_arguments]"
    exit 1
fi

SCRIPT_FILE="$1"
shift  # Remove the script file from the arguments

# Check if the script file exists
if [ ! -f "$SCRIPT_FILE" ]; then
    echo "Error: Script file '$SCRIPT_FILE' not found."
    exit 1
fi

# Create a temporary file
TEMP_FILE=$(mktemp)

# Replace 'python' with 'uv run' in the script
sed 's/python /uv run /g' "$SCRIPT_FILE" > "$TEMP_FILE"

# Make the temporary file executable
chmod +x "$TEMP_FILE"

# Execute the modified script with any additional arguments
"$TEMP_FILE" "$@"

# Clean up the temporary file
rm "$TEMP_FILE"

echo "Script execution completed."
