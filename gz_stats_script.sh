#!/bin/bash

# Run gz stats and redirect output to a file
OUTPUT_FILE="/tmp/gazebo_rtf_output.txt"

# Kill any existing gz stats process
pkill -f "gz stats"

# Clear the output file
> $OUTPUT_FILE

# Run gz stats in the background and redirect output to the file
gz stats > $OUTPUT_FILE &

echo "Started gz stats with PID: $!"
echo "Output is being written to $OUTPUT_FILE"
echo "To stop, run: pkill -f 'gz stats'"