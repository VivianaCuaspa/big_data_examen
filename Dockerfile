FROM ubuntu:latest
LABEL authors="Beez"

ENTRYPOINT ["top", "-b"]