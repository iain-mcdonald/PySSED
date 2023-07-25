FROM python:3.8

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /sda

COPY ./ ./

ENV dash_port=8050
ENV dash_debug="False"
EXPOSE 8050

ENTRYPOINT [ "python3" ]
CMD [ "sda_gtomo.py" ]
