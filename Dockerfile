From python:latest

COPY . .

RUN pip install -r requirements.txt
ENV PASSWORD=${PASSWORD}
ENTRYPOINT [ "cd ./src/ && python3 main.py" ]