FROM tensorflow/tensorflow:latest

# Copy logs into container
COPY logs/ /logdir

# Expose TensorBoard port
EXPOSE 6006

CMD ["tensorboard", "--logdir=/logdir", "--bind_all", "--port=6006"]
