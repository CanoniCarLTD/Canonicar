FROM tensorflow/tensorflow:latest

# Copy logs into container
COPY logs/ /logdir

# Expose TensorBoard port
EXPOSE 6006

# Start TensorBoard when container runs
CMD ["tensorboard", "--logdir=/logdir", "--bind_all", "--port=6006"]

FROM tensorflow/tensorflow:latest

# Copy logs into container (only for standalone use, not used with bind mount)
COPY logs/ /logdir

# Expose TensorBoard port
EXPOSE 6006

# Set entrypoint to dynamically find latest log folder
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["latest=$(find /logdir -maxdepth 2 -type d -name 'run_*' | sort | tail -n 1) && \
  echo \"Launching TensorBoard with: $latest\" && \
  tensorboard --logdir=$latest --bind_all --port=6006"]
