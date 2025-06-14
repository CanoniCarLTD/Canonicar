FROM tensorflow/tensorflow:latest

COPY logs/ /logdir

EXPOSE 6006

ENV TENSORBOARD_LOGDIR="/logdir/v3.0.0/run_20250328_0001/tensorboard"

CMD ["tensorboard", "--logdir=${TENSORBOARD_LOGDIR}", "--bind_all", "--port=6006"]