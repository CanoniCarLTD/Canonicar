FROM tensorflow/tensorflow:latest

# Copy logs into container
COPY logs/ /logdir

# Expose TensorBoard port
EXPOSE 6006

CMD ["tensorboard", "--logdir=/logdir/v3.0.0/run_20250328_0001/tensorboard", "--bind_all", "--port=6006"]

